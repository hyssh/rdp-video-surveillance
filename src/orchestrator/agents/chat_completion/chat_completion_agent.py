import os
import yaml
import asyncio
import base64
import logging
import re
import random
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from dotenv import load_dotenv
from typing import List, Optional, Union, Tuple, Dict, Any
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from pydantic import BaseModel
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.image_content import ImageContent
# from semantic_kernel.agents import AgentResponseItem

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global service instance for singleton pattern
_service_instance = None

# Add these constants at the top of the file
MAX_RETRY_ATTEMPTS = 5
MIN_RETRY_WAIT_SECONDS = 1
MAX_RETRY_WAIT_SECONDS = 60
RATE_LIMIT_PATTERN = re.compile(r"Rate limit is exceeded. Try again in (\d+) seconds")

# Create a custom exception class for rate limiting
class AzureRateLimitExceededException(Exception):
    """Exception raised when Azure AI service rate limits are exceeded."""
    def __init__(self, message, retry_after_seconds=None):
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        super().__init__(self.message)

# Add a function to check if an exception is due to rate limiting
def is_rate_limit_error(exception):
    """Check if the exception is due to rate limiting and extract wait time."""
    if not str(exception):
        return False, None
        
    error_msg = str(exception).lower()
    if "rate limit" in error_msg or "too many requests" in error_msg:
        # Try to extract the wait time from the error message
        match = RATE_LIMIT_PATTERN.search(str(exception))
        retry_after = int(match.group(1)) if match else None
        return True, retry_after
    return False, None

# Add a rate limiting tracker to manage concurrent requests
class RateLimitTracker:
    """Tracks rate limiting across the application."""
    
    def __init__(self):
        self.last_rate_limit_time = None
        self.backoff_until = None
        self.consecutive_failures = 0
        
    def record_rate_limit(self, retry_after_seconds=None):
        """Record a rate limit hit and calculate backoff time."""
        self.last_rate_limit_time = datetime.now()
        self.consecutive_failures += 1
        
        # If we have a specific retry-after time, use it
        if retry_after_seconds:
            backoff_seconds = retry_after_seconds
        else:
            # Otherwise use exponential backoff with jitter
            base_backoff = min(MAX_RETRY_WAIT_SECONDS, 
                              MIN_RETRY_WAIT_SECONDS * (2 ** (self.consecutive_failures - 1)))
            # Add jitter to prevent thundering herd problem
            jitter = random.uniform(0, 0.1 * base_backoff)
            backoff_seconds = base_backoff + jitter
            
        self.backoff_until = self.last_rate_limit_time + timedelta(seconds=backoff_seconds)
        return backoff_seconds
        
    def should_throttle(self):
        """Check if requests should be throttled based on recent rate limits."""
        if not self.backoff_until:
            return False
            
        return datetime.now() < self.backoff_until
        
    def get_wait_time(self):
        """Get seconds to wait if throttling is active."""
        if not self.backoff_until:
            return 0
            
        wait_seconds = (self.backoff_until - datetime.now()).total_seconds()
        return max(0, wait_seconds)
        
    def reset(self):
        """Reset after successful operation."""
        self.consecutive_failures = 0
        self.backoff_until = None

# Create a global rate limit tracker
_rate_limit_tracker = RateLimitTracker()

# Create a retry decorator for Azure operations with rate limit handling
def with_azure_rate_limit_retry(max_attempts=MAX_RETRY_ATTEMPTS):
    """Decorator to add rate limit-aware retry logic to Azure operations."""
    def decorator(func):
        @retry(
            retry=retry_if_exception_type(AzureRateLimitExceededException),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT_SECONDS, max=MAX_RETRY_WAIT_SECONDS),
            before_sleep=before_sleep_log(logger, logging.INFO)
        )
        async def wrapper(*args, **kwargs):
            global _rate_limit_tracker
            
            # Check if we're in a backoff period
            if _rate_limit_tracker.should_throttle():
                wait_time = _rate_limit_tracker.get_wait_time()
                logger.info(f"Rate limiting active, waiting {wait_time:.2f} seconds before trying")
                await asyncio.sleep(wait_time)
            
            try:
                result = await func(*args, **kwargs)
                # Success - reset the rate limit tracker
                _rate_limit_tracker.reset()
                return result
            except Exception as e:
                is_rate_limit, retry_after = is_rate_limit_error(e)
                if is_rate_limit:
                    # Track the rate limit and calculate backoff
                    backoff_seconds = _rate_limit_tracker.record_rate_limit(retry_after)
                    logger.warning(f"Rate limit exceeded. Backing off for {backoff_seconds:.2f} seconds.")
                    # Raise custom exception for the retry mechanism
                    raise AzureRateLimitExceededException(str(e), retry_after)
                else:
                    # Not a rate limit error, re-raise
                    raise
        return wrapper
    return decorator

class ImageAnalyzerService:
    """Service class to manage the lifecycle of Azure AI Agent instances"""
    
    def __init__(self):
        self.client = None
        self.agent = None
        self.credential = None
        
    async def initialize(self):
        """Initialize the Azure AI Agent client and resources"""
        try:
            # Use DefaultAzureCredential for managed identity support in production
            self.credential = DefaultAzureCredential()
            
            # Configure agent settings with appropriate timeout and retry policies
            # settings = AzureAIAgentSettings(
            #     endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
            #     model_deployment_name=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
            #     # timeout=120,  # Increase timeout for production workloads
            # )
            
            # Create client with explicit settings
            # self.client = AzureAIAgent.create_client(
            #     credential=self.credential,
            #     settings=settings
            # )
            
            # Get or create the agent
            self.agent = await self._get_or_create_agent()
            
            return self.agent
        except Exception as e:
            logger.error(f"Failed to initialize ImageAnalyzerService: {e}")
            # Cleanup resources in case of initialization failure
            await self.close()
            raise


    def load_prompts(self, prompt_path: str="agents/chat_completion/agent.yml") -> str:
        """
        Load a prompt from a file.

        Args:
            prompt_path (str): Path to the prompt YAML file.        

        Returns:
            str: The image analyzer prompt.
        """
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)

            image_analyzer_prompt = prompts.get('image_analyzer_prompt', '')

            if not image_analyzer_prompt:
                raise ValueError("image_analyzer_prompt must be provided in the YAML file.")
            
            return image_analyzer_prompt
        except Exception as e:
            logger.error(f"Error loading prompts from {prompt_path}: {str(e)}")
            raise
    
    async def _get_or_create_agent(self) -> ChatCompletionAgent:
        """Get existing agent or create a new one"""
        try:
            # # Check for existing agent
            # agent_list = self.client.agents.list_agents()
            
            # async for existing_agent in agent_list:
            #     if existing_agent.name == "security_image_analyzer_agent":
            #         agent_definition = await self.client.agents.get_agent(existing_agent.id)
            #         logger.info(f"Found existing agent: {existing_agent.id}")
            #         return AzureAIAgent(client=self.client, definition=agent_definition)

            if self.agent:
                logger.info("Using existing agent instance")
                return self.agent
            
            agent = ChatCompletionAgent(
                service=AzureChatCompletion(instruction_role="system"),
                name="security_image_analyzer_agent",
                instructions=self.load_prompts(),  # Load the prompt from YAML
            )

            return agent

        except Exception as e:
            logger.error(f"Error getting or creating agent: {e}")
            raise
    
    @with_azure_rate_limit_retry(max_attempts=5)
    async def chat(self, 
                    base64_image: str, 
                    omniparser_response: Dict[str, Any],
                    thread: Optional[ChatHistoryAgentThread] = None,
                    custom_prompt: Optional[str] = None) -> str:
        """
        Analyze an image using the Azure AI Agent with rate limit handling
        
        Args:
            base64_image: Base64 encoded image data
            omniparser_response: Response from omniparser with image analysis
            thread: Optional thread ID for continuing a conversation
            custom_prompt: Optional custom prompt to guide the analysis
            
        Returns:
            The analysis result from the agent
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        # Build the prompt
        prompt = custom_prompt or "Analyze the image for any suspicious activities or security threats."
        full_prompt = f"{prompt}\n\nUse the given image data:\nAnd the omniparser response:\n{omniparser_response}"

        chat_message_contents = [
            ChatMessageContent(
                role="user",
                items=[
                    ImageContent(data_format="base64", data=base64_image, mime_type="image/jpeg")
                ]
            ),
            ChatMessageContent(
                role="user",
                items=[
                    TextContent(text=full_prompt),
                ]
            ),
        ]
        
        try:
            # Get response with proper error handling            
            response = await self.agent.get_response(messages=chat_message_contents, thread=thread, top_p=0.1, temperature=0.1)
            return response
        except Exception as e:
            # Check if this is a rate limit error
            is_rate_limit, retry_after = is_rate_limit_error(e)
            if is_rate_limit:
                logger.warning(f"Azure AI rate limit exceeded. Retry suggested after {retry_after} seconds.")
                # The decorator will handle retries for AzureRateLimitExceededException
                raise AzureRateLimitExceededException(str(e), retry_after)
            
            # Handle other errors
            logger.error(f"Error analyzing image: {e}")
            if "HTTP transport has already been closed" in str(e):
                logger.error("Connection error: The Azure AI service connection was closed.")
            raise
    
    async def close(self):
        """Close and cleanup resources"""
        try:
            if self.credential:
                await self.credential.close()
                self.credential = None
        except Exception as e:
            logger.error(f"Error closing credential: {e}")

async def get_chat_completion_agent():
    """
    Create or retrieve an Azure AI Agent for image analysis.
    
    Returns:
        AzureAIAgent: The initialized agent instance
    """
    global _service_instance
    
    # Create service if it doesn't exist
    if _service_instance is None:
        _service_instance = ImageAnalyzerService()
    
    # Initialize and return the agent
    return await _service_instance.initialize()

@with_azure_rate_limit_retry(max_attempts=3)
async def chat_completion_agent_run(agent: ChatCompletionAgent, 
                                   thread: Optional[ChatHistoryAgentThread] = None, 
                                   user_input: Optional[str] = None, 
                                   base64_image: str = "IMAGE IS MISSING", 
                                   omniparser_response: set = ("type:Unkown", "interactivity:Unknown","content:Unknown")):
    """
    Run the image analyzer agent with the given parameters and rate limit handling.
    
    Args:
        agent: The Azure AI Agent instance
        thread: Optional thread ID for continuing a conversation
        user_input: Optional custom prompt
        base64_image: Base64 encoded image data
        omniparser_response: Response from omniparser with image analysis
        
    Returns:
        The agent's response
    """
    global _service_instance
    
    # Verify agent type with proper error handling
    if not isinstance(agent, ChatCompletionAgent):
        raise TypeError(f"agent must be an instance of AzureAIAgent, got {type(agent).__name__}")
    
    # Verify service is initialized
    if _service_instance is None:
        logger.warning("Service not initialized. Creating a new service instance.")
        _service_instance = ImageAnalyzerService()
        await _service_instance.initialize()
    
    try:
        # Use the service to analyze the image
        return await _service_instance.chat(
            base64_image=base64_image,
            omniparser_response=omniparser_response,
            thread=thread,
            custom_prompt=user_input
        )
    except AzureRateLimitExceededException as e:
        # This will be caught by the decorator and retried
        raise
    except Exception as e:
        # Improved error handling with more detailed diagnostics
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error running image analyzer agent: {e}\n{error_details}")
        
        # For HTTP transport closed errors, provide a more helpful message
        if "HTTP transport has already been closed" in str(e):
            logger.error("The HTTP transport was closed. This typically happens when using an agent outside its context manager.")
            logger.error("Recommendation: Ensure all agent operations happen within the same async context.")
        
        # Re-raise for proper error propagation
        raise

# async def get_or_create_thread_with_backoff(agent: ChatCompletionAgent,) -> Optional[ChatHistoryAgentThread]:
#     """
#     Get or create a thread with rate limit handling.
    
#     Args:
#         agent: The Azure AI Agent instance
#         metadata: Optional metadata for the thread
        
#     Returns:
#         The thread or None if creation fails
#     """
#     global _rate_limit_tracker
    
#     # Check if we're in a backoff period
#     if _rate_limit_tracker.should_throttle():
#         wait_time = _rate_limit_tracker.get_wait_time()
#         logger.info(f"Rate limiting active, waiting {wait_time:.2f} seconds before creating thread")
#         await asyncio.sleep(wait_time)
    
#     # try:
#         # Try to create the thread
#         # properties = {}
#         # if metadata:
#         #     properties["metadata"] = metadata
            
#         # thread_response = await agent#client.agents.create_thread(properties=properties)
#         # logger.info(f"Created new thread: {thread_response.id}")
        
#         # # Reset rate limit tracking on success
#         # _rate_limit_tracker.reset()
        
#         # return AzureAIAgentThread(thread_id=thread_response.id, client=agent.client)
#     # except Exception as e:
#     #     is_rate_limit, retry_after = is_rate_limit_error(e)
#     #     if is_rate_limit:
#     #         # Record the rate limit and log
#     #         backoff_seconds = _rate_limit_tracker.record_rate_limit(retry_after)
#     #         logger.warning(f"Rate limit hit when creating thread. Backing off for {backoff_seconds} seconds.")
#     #         # Try again with backoff if needed
#     #         return None
#     #     else:
#     #         logger.error(f"Error creating thread: {e}")
#     #         return None

async def cleanup_resources():
    """Clean up all Azure resources properly"""
    global _service_instance
    if _service_instance:
        await _service_instance.close()
        _service_instance = None
        logger.info("Azure resources cleaned up")

if __name__ == "__main__":
    async def main():
        try:
            agent = await get_chat_completion_agent()
            logger.info(f"Agent initialized with ID: {agent.id}")
            
            # Test the agent
            result = await chat_completion_agent_run(
                agent=agent,
                user_input="This is a test. Analyze this image for testing purposes.",
                base64_image="TEST_IMAGE_DATA",
                omniparser_response={"MSG": "THIS IS A TEST"}
            )
            
            logger.info(f"Test result: {result}")
        finally:
            # Always clean up resources
            await cleanup_resources()
    
    asyncio.run(main())