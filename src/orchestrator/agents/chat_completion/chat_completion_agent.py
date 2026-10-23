import yaml
import asyncio
import logging
import uuid
from dotenv import load_dotenv
from typing import Optional
from fastapi import HTTPException
from agent_framework import Agent, AgentSession

from agents.llm import make_agent, image_message, response_text


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


async def chat(agent: Agent,
               base64_image: str,
               omniparser_response: Optional[list | set],
               session: Optional[AgentSession] = None,
               custom_prompt: Optional[str] = None):
    """Analyze one frame (image + OmniParser output) with the image analyzer agent."""
    prompt = custom_prompt or "Analyze the image for any suspicious activities or security threats."
    full_prompt = f"{prompt}\n\nUse the given image data:\nAnd the omniparser response:\n{omniparser_response}"

    try:
        return await agent.run(image_message(base64_image, full_prompt), session=session)
    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        logger.error(f"Exception details: {getattr(e, '__dict__', None)}")
        raise HTTPException(status_code=500, detail=f"{getattr(e, '__dict__', None)}")


async def get_chat_completion_agent_session(session_id: Optional[str] = None):
    """
    Create the image analyzer agent and a chat session.

    Args:
        session_id: Optional session ID. Defaults to a unique ID per call so concurrent
            requests never share a session identifier.
    """
    session_id = session_id or f"image_analyzer_session_{uuid.uuid4().hex}"
    agent = make_agent("ImageAnalyzerAgent", await asyncio.to_thread(load_prompts))
    return agent, agent.create_session(session_id=session_id)


QA_AGENT_INSTRUCTIONS = (
    "You answer questions about a screen-recording (remote desktop session) using only the frame "
    "descriptions provided. Cite the supporting frames inline as [frame <id> @ <timestamp>s]. "
    "If the descriptions do not contain the answer, say so plainly. Be concise."
)


async def get_qa_agent() -> Agent:
    """Create a lightweight Q&A agent (same Azure OpenAI chat deployment as the other agents)."""
    return make_agent("VideoQAAgent", QA_AGENT_INSTRUCTIONS)


async def qa_agent_run(agent: Agent, question: str, frames: list) -> str:
    """
    Answer a question from matched frames.

    Args:
        frames: dicts with 'id', 'frame_timestamp' and 'image_description'
    """
    context = "\n\n".join(
        f"[frame {f['id']} @ {f.get('frame_timestamp')}s]\n{f.get('image_description') or ''}" for f in frames
    )
    prompt = f"Question: {question}\n\nFrame descriptions:\n{context}"
    return response_text(await agent.run(prompt))


async def chat_completion_agent_run(agent: Agent,
                                    session: Optional[AgentSession] = None,
                                    user_input: Optional[str] = None,
                                    base64_image: str = "IMAGE IS MISSING",
                                    omniparser_response: Optional[set] = None):
    """Run the image analyzer agent for one frame and return the response text."""
    response = await chat(
        agent=agent,
        base64_image=base64_image,
        omniparser_response=omniparser_response,
        session=session,
        custom_prompt=user_input,
    )
    return response_text(response)
