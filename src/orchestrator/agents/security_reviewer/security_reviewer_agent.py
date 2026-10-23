import yaml
import logging
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from agent_framework import Agent

from agents.llm import make_agent


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


async def get_or_create_chat_completion_agent() -> Agent:
    """Create the security reviewer agent."""
    try:
        logger.info("Creating new security reviewer agent")
        return make_agent("security_image_analyzer_agent", load_prompts())
    except Exception as e:
        logger.error(f"Error getting or creating agent: {e}")
        raise


async def security_reviewer_agent_run(agent: Agent,
                                      image_analysis_results: Optional[List[Dict[str, Any]]] = None,
                                      custom_prompt: Optional[str] = None):
    """
    Run the security reviewer agent and return its response text.

    Args:
        agent: The reviewer agent
        image_analysis_results: List of results from image analyzer with keys 'id', 'description', etc.
        custom_prompt: Optional custom prompt
    """
    default_prompt = "Review these image analysis results and identify any security issues or concerns."
    if image_analysis_results is not None:
        formatted_results = ""
        for idx, result in enumerate(image_analysis_results):
            timestamp = result.get('frame_timestamp', 'unknown time')
            description = result.get('image_description', 'No description available')
            formatted_results += f"Frame {idx+1} ({timestamp}):\n{description}\n\n"
        full_prompt = f"{custom_prompt or default_prompt}\n\nImage Analysis Results:\n{formatted_results}"
    else:
        full_prompt = custom_prompt or default_prompt

    try:
        response = await agent.run(full_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error running security reviewer agent: {e}\n")
        raise ValueError(f"Failed to run security reviewer agent: {e}")
