import os
import yaml
import asyncio
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.connectors.ai.open_ai.settings.azure_open_ai_settings import AzureOpenAISettings


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_prompts(prompt_path: str="agents/chat_completion/agent.yml") -> str:
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


async def chat(agent: ChatCompletionAgent,
               base64_image: str, 
                omniparser_response: set = ("OMNIPARSER:Unknown"),
                thread: Optional[ChatHistoryAgentThread] = None,
                custom_prompt: Optional[str] = None):
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
    if not isinstance(agent, ChatCompletionAgent):
        raise ValueError("Agent not initialized. Call initialize() first.")
    
    # Build the prompt
    prompt = custom_prompt or "Analyze the image for any suspicious activities or security threats."
    full_prompt = f"{prompt}\n\nUse the given image data:\nAnd the omniparser response:\n{omniparser_response}"

    chat_message_contents = [
        ChatMessageContent(
            role="system",
            items=[
                TextContent(text=load_prompts())
            ]
        ),
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
        response = await agent.get_response(messages=chat_message_contents, thread=thread, top_p=0.1, temperature=0.1)
        return response
    except Exception as e:
        # Handle other errors
        logger.error(f"Error analyzing image: {e}")
        if "HTTP transport has already been closed" in str(e):
            logger.error("Connection error: The Azure AI service connection was closed.")
        raise


async def get_chat_completion_agent_thread():
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            api_key=os.getenv("AZURE_OPENAI_APIKEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        ),
        name="ImageAnalyzerAgent",
        instructions=load_prompts(),
    )
    
    thread = ChatHistoryAgentThread(
        chat_history=ChatHistory(),
        thread_id="image_analyzer_thread"
    )
    return agent, thread


async def chat_completion_agent_run(agent: ChatCompletionAgent, 
                                   thread: Optional[ChatHistoryAgentThread] = None, 
                                   user_input: Optional[str] = None, 
                                   base64_image: str = "IMAGE IS MISSING", 
                                   omniparser_response: set = ("OMNIPARSER:Status_Unknown")):
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
    
    # Verify agent type with proper error handling
    if not isinstance(agent, ChatCompletionAgent):
        raise TypeError(f"agent must be an instance of AzureAIAgent, got {type(agent).__name__}")
    
    try:
        # Use the service to analyze the image
        return await chat(
            agent=agent,
            base64_image=base64_image,
            omniparser_response=omniparser_response,
            thread=thread,
            custom_prompt=user_input
        )
    except Exception as e:
        # For HTTP transport closed errors, provide a more helpful message
        if "HTTP transport has already been closed" in str(e):
            logger.error("The HTTP transport was closed. This typically happens when using an agent outside its context manager.")
            logger.error("Recommendation: Ensure all agent operations happen within the same async context.")
        
        # Re-raise for proper error propagation
        raise


if __name__ == "__main__":
    async def main():
        try:
            agent, thread = await get_chat_completion_agent_thread()
            logger.info(f"Agent initialized with ID: {agent.id} with thread ID: {thread}")
            
            # Test the agent
            result = await chat_completion_agent_run(
                agent=agent,
                user_input="This is a test. Analyze this image for testing purposes.",
                base64_image="TEST_IMAGE_DATA",
                omniparser_response=set("OMNIPARSER:Status_Unknown")
            )
            
            logger.info(f"Test result: {result}")
        except Exception as e:
            logger.error(f"An error occurred in the main function: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    asyncio.run(main())