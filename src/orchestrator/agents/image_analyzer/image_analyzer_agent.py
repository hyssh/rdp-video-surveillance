import os
import yaml
import asyncio
# import base64
import logging
# import re
# import random
# from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.image_content import ImageContent
# from semantic_kernel.agents import AgentResponseItem

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

    
async def get_security_reviewer_agent_thread():
    """Initialize the Azure AI Agent client and resources"""
    try:
        # Use DefaultAzureCredential for managed identity support in production
        self.credential = DefaultAzureCredential()
        
        # Configure agent settings with appropriate timeout and retry policies
        settings = AzureAIAgentSettings(
            endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
            model_deployment_name=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
            # timeout=120,  # Increase timeout for production workloads
        )
        
        # Create client with explicit settings
        self.client = AzureAIAgent.create_client(
            credential=self.credential,
            settings=settings
        )
        
        # Get or create the agent
        self.agent = await self._get_or_create_agent()
        
        return self.agent
    except Exception as e:
        logger.error(f"Failed to initialize ImageAnalyzerService: {e}")
        # Cleanup resources in case of initialization failure
        await self.close()
        raise


def load_prompts(self, prompt_path: str="agents/image_analyzer/agent.yml") -> str:
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

async def _get_or_create_agent(self) -> AzureAIAgent:
    """Get existing agent or create a new one"""
    try:
        # Check for existing agent
        agent_list = self.client.agents.list_agents()
        
        async for existing_agent in agent_list:
            if existing_agent.name == "security_image_analyzer_agent":
                agent_definition = await self.client.agents.get_agent(existing_agent.id)
                logger.info(f"Found existing agent: {existing_agent.id}")
                return AzureAIAgent(client=self.client, definition=agent_definition)
        
        # If no existing agent found, create a new one
        logger.info("Creating new security image analyzer agent")
        instructions = self.load_prompts()
        
        agent_definition = await self.client.agents.create_agent(
            model=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
            name="security_image_analyzer_agent",
            instructions=instructions,
            description="Agent that analyzes Remote Desktop recording and images for suspicious activities, or security threats.",
        )
        
        return AzureAIAgent(client=self.client, definition=agent_definition)
    except Exception as e:
        logger.error(f"Error getting or creating agent: {e}")
        raise

async def analyze_image(self, 
                        base64_image: str, 
                        omniparser_response: Dict[str, Any],
                        thread: Optional[AzureAIAgentThread] = None,
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
    prompt = custom_prompt or "Must describe the given image in terms of IT security point of view. If you do not see an image. Tell me I dont' have an image. Analyze the image for any suspicious activities or security threats."
    # full_prompt = f"{prompt}\n\nUse the given image data:\nAnd the omniparser response:\n{omniparser_response}"

    chat_message_contents = [
        ChatMessageContent(
            role="user",
            items=[
                ImageContent(data_format="base64", data=base64_image, mime_type="image/jpeg"),
                TextContent(text=prompt),
            ]
        ),
        # ChatMessageContent(
        #     role="user",
        #     items=[
        #         TextContent(text=full_prompt),
        #     ]
        # ),
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

async def get_image_analyzer_agent():
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

async def image_analyzer_agent_run(agent: AzureAIAgent, 
                                   thread: Optional[AzureAIAgentThread] = None, 
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
    if not isinstance(agent, AzureAIAgent):
        raise TypeError(f"agent must be an instance of AzureAIAgent, got {type(agent).__name__}")
    
    # Verify service is initialized
    if _service_instance is None:
        logger.warning("Service not initialized. Creating a new service instance.")
        _service_instance = ImageAnalyzerService()
        await _service_instance.initialize()
    
    try:
        # Use the service to analyze the image
        return await _service_instance.analyze_image(
            base64_image=base64_image,
            omniparser_response=omniparser_response,
            thread=thread,
            custom_prompt=user_input
        )
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

# async def get_or_create_thread_with_backoff(agent: AzureAIAgent, metadata: Dict[str, Any] = None) -> Optional[AzureAIAgentThread]:
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
    
#     try:
#         # Try to create the thread
#         properties = {}
#         if metadata:
#             properties["metadata"] = metadata
            
#         thread_response = await agent.client.agents.create_thread(properties=properties)
#         logger.info(f"Created new thread: {thread_response.id}")
        
#         # Reset rate limit tracking on success
#         _rate_limit_tracker.reset()
        
#         return AzureAIAgentThread(thread_id=thread_response.id, client=agent.client)
#     except Exception as e:
#         is_rate_limit, retry_after = is_rate_limit_error(e)
#         if is_rate_limit:
#             # Record the rate limit and log
#             backoff_seconds = _rate_limit_tracker.record_rate_limit(retry_after)
#             logger.warning(f"Rate limit hit when creating thread. Backing off for {backoff_seconds} seconds.")
#             # Try again with backoff if needed
#             return None
#         else:
#             logger.error(f"Error creating thread: {e}")
#             return None

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
            agent = await get_image_analyzer_agent()
            logger.info(f"Agent initialized with ID: {agent.id}")
            
            # Test the agent
            result = await image_analyzer_agent_run(
                agent=agent,
                user_input="This is a test. Analyze this image for testing purposes.",
                base64_image="TEST_IMAGE_DATA",
                # omniparser_response=None
            )
            
            logger.info(f"Test result: {result}")
        finally:
            # Always clean up resources
            await cleanup_resources()
    
    asyncio.run(main())