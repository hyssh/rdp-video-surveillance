"""
# security_reviewer_agent.py
This agent can help to read image descriptions and analyze security issues
It will infer the context between the image descriptions and determine if there are any security issues
"""
import os
import yaml
import asyncio
import logging
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from typing import List, Optional, Union, Tuple, Dict, Any
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import AzureAIAgentSettings
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_prompts(prompt_path: str="agents/security_reviewer/agent.yml") -> str:
    """
    Load a prompt from a file.

    Args:
        prompt_path (str): Path to the prompt YAML file.        

    Returns:
        str: The security reviewer prompt.
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)

        security_reviewer_prompt = prompts.get('security_reviewer_prompt', '')

        if not security_reviewer_prompt:
            raise ValueError("security_reviewer_prompt must be provided in the YAML file.")
        
        return security_reviewer_prompt
    except Exception as e:
        logger.error(f"Error loading prompts from {prompt_path}: {str(e)}")
        raise


async def get_or_create_chat_completion_agent() -> ChatCompletionAgent:
    """Get existing agent or create a new one"""
    try:
        logger.info("Creating new security reviewer agent")
    
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(                
                api_key=os.getenv("AZURE_OPENAI_APIKEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            ),
            name="security_image_analyzer_agent",
            instructions=load_prompts(),  # Load the prompt from YAML
        )
        return agent
    
    except Exception as e:
        logger.error(f"Error getting or creating agent: {e}")
        raise


async def security_reviewer_agent_run(agent: ChatCompletionAgent, 
                                    image_analysis_results: Optional[List[Dict[str, Any]]] = None,
                                    thread: Optional[ChatHistoryAgentThread] = None, 
                                    custom_prompt: Optional[str] = None):
    """
    Run the security reviewer agent with the given parameters.
    
    Args:
        agent: The Azure AI Agent instance
        image_analysis_results: List of results from image analyzer with keys 'id', 'description', etc.
        thread: Optional thread ID for continuing a conversation
        custom_prompt: Optional custom prompt
        
    Returns:
        The agent's response
    """
    
    # Verify agent type with proper error handling
    if not isinstance(agent, ChatCompletionAgent):
        raise TypeError(f"agent must be an instance of AzureAIAgent, got {type(agent).__name__}")
    
    if image_analysis_results is not None:
        # Format the image analysis results for the agent
        formatted_results = ""
        for idx, result in enumerate(image_analysis_results):
            timestamp = result.get('frame_timestamp', 'unknown time')
            description = result.get('image_description', 'No description available')
            formatted_results += f"Frame {idx+1} ({timestamp}):\n{description}\n\n"
        
        # Build the prompt
        prompt = custom_prompt or "Review these image analysis results and identify any security issues or concerns."
        full_prompt = f"{prompt}\n\nImage Analysis Results:\n{formatted_results}"
    else:
        full_prompt = custom_prompt or "Review these image analysis results and identify any security issues or concerns."

    try:        
        # Get response with proper error handling
        response = await agent.get_response(messages=full_prompt, top_p=0.1, temperature=0.1)
        return response
    except Exception as e:
        # Improved error handling with more detailed diagnostics
        logger.error(f"Error running security reviewer agent: {e}\n")
        
        # Re-raise for proper error propagation
        raise ValueError(f"Failed to run security reviewer agent: {e}")
    
# async def get_or_create_ai_agent() -> AzureAIAgent:
#     """Get existing agent or create a new one"""
#     try:
#         # Check for existing agent
#         agent_list = client.agents.list_agents()
        
#         async for existing_agent in agent_list:
#             if existing_agent.name == "security_reviewer_agent":
#                 agent_definition = await self.client.agents.get_agent(existing_agent.id)
#                 logger.info(f"Found existing agent: {existing_agent.id}")
#                 return AzureAIAgent(client=self.client, definition=agent_definition)
        
#         # If no existing agent found, create a new one
#             # Configure agent settings with appropriate timeout and retry policies
#         settings = AzureAIAgentSettings(
#             endpoint=os.getenv("AZURE_AI_AGENT_ENDPOINT"),
#             model_deployment_name=os.getenv("AZURE_AI_AGENT_REASONING_MODEL_DEPLOYMENT_NAME", "o3"),
#             # timeout=300,  # Increase timeout for production workloads
#         )
        
#         # Create client with explicit settings
#         client = AzureAIAgent.create_client(
#             credential=DefaultAzureCredential(),
#             settings= settings
#         )

#         AzureAIAgent.client.agents.creat

#         logger.info("Creating new security reviewer agent")
        
#         agent = await client.agents.create_agent(
#             model=os.getenv("AZURE_AI_AGENT_REASONING_MODEL_DEPLOYMENT_NAME", "o3"),
#             name="security_reviewer_agent",
#             instructions=load_prompts(),  # Load the prompt from YAML
#             description="Agent that reviews the results from image analyzer output for security review.",            
#         )
        
#         return agent  
#     except Exception as e:
#         logger.error(f"Error getting or creating agent: {e}")
#         raise


# async def review_security(self, 
#                         image_analysis_results: str,
#                         thread_id: Optional[str] = None,
#                         custom_prompt: Optional[str] = None) -> str:
#     """
#     Review security issues based on image analysis results
    
#     Args:
#         image_analysis_results: List of results from image analyzer
#         thread_id: Optional thread ID for continuing a conversation
#         custom_prompt: Optional custom prompt to guide the analysis
        
#     Returns:
#         The security review result from the agent
#     """
#     if not self.agent:
#         raise ValueError("Agent not initialized. Call initialize() first.")
    
#     # Build the prompt
#     prompt = custom_prompt or "Review these image analysis results and identify any security issues or concerns."
#     full_prompt = f"{prompt}\n\nImage Analysis Results:\n{image_analysis_results}"

#     chat_message_contents = [
#         ChatMessageContent(
#             role="user",
#             items=[
#                 TextContent(text=full_prompt)
#             ]
#         )
#     ]
    
#     # Create thread if needed
#     # thread = AzureAIAgentThread(thread_id=thread_id, client=self.client) if thread_id else None
    
#     try:
#         # Get response with proper error handling
#         response = await agent.get_response(messages=chat_message_contents, thread=thread, top_p=0.1, temperature=0.1)
#         return response
#     except Exception as e:
#         logger.error(f"Error reviewing security: {e}")
#         if "HTTP transport has already been closed" in str(e):
#             logger.error("Connection error: The Azure AI service connection was closed.")
#         raise



if __name__ == "__main__":
    async def main():
        try:
            agent = await get_or_create_chat_completion_agent()
            logger.info(f"Agent initialized with ID: {agent.id}")
            
            # Test the agent
            test_results = [
                {"image_id": 1, "description": "A person accessing a computer terminal", "potential_issues": ["unauthorized access"]},
                {"image_id": 2, "description": "A screenshot showing a login screen", "potential_issues": ["credential exposure"]}
            ]
            
            result = await security_reviewer_agent_run(
                agent=agent,
                image_analysis_results=test_results,
                custom_prompt="Review these analysis results for security concerns"
            )
            
            logger.info(f"Test result: {result}")
        except Exception as e:
            logger.error(f"An error occurred in the main function: {e}")
    
    asyncio.run(main())