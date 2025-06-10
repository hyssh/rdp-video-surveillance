import os
import gc
import os
import cv2
import base64
import requests
from typing import List, Dict, Any
# from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from semantic_kernel.agents import AzureAIAgentThread
from postgresql import insert_video, insert_frames, update_frame_with_embedding, get_frames_by_video_id, get_video_by_id, update_video_with_security_review
from agents.thread_manager import ThreadManager
from agents.image_analyzer.image_analyzer_agent import get_image_analyzer_agent, image_analyzer_agent_run
from agents.security_reviewer.security_reviewer_agent import get_security_reviewer_agent, security_reviewer_agent_run
from semantic_kernel.connectors.ai.open_ai.settings.azure_open_ai_settings import AzureOpenAISettings
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
import numpy as np
import logging
import uuid

logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')

load_dotenv()

orchestrator = FastAPI()

# Initialize the thread manager as a global singleton for efficient resource usage
_thread_manager = ThreadManager()

@orchestrator.get("/")
def read_root():
    return {"message": "API is running"}


async def upload_to_blob_storage(file: UploadFile, container_name: str, blob_name: str = None):
    """

    Upload a file to Azure Blob Storage
    
    Args:
        file: The file to upload
        container_name: The container to upload to
        blob_name: Optional custom blob name, defaults to the original filename if not provided
    
    Returns:
        dict: Information about the uploaded blob
    """
    try:
        # Get connection string from environment variables
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise HTTPException(status_code=500, detail="Azure Storage connection string not configured or missing")
            
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        
        # Determine blob name - use provided name, original filename, or generate UUID
        if not blob_name:
            # Use original filename or generate a unique ID
            blob_name = file.filename or f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Read file content
        file_content = await file.read()
        
        # Determine content type
        content_type = file.content_type or "application/octet-stream"
        
        # Upload the file
        blob_client.upload_blob(
            file_content, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        
        # Return blob information
        return {
            "filename": file.filename,
            "blob_name": blob_name,
            "container_name": container_name,
            "size_bytes": len(file_content),
            "content_type": content_type,
            "url": blob_client.url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


async def upload_file_endpoint(file: UploadFile = File(...), container_name: str = "videos"):
    result = await upload_to_blob_storage(file, container_name)
    return result


async def store_video_frame_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store video and frame metadata in Azure PostgreSQL database
    """
    logging.info(f"Storing video and frame metadata: {metadata}")
    try:
        # Parse the request body
        
        video_data = metadata.get("video", {})
        if not video_data:
            raise HTTPException(status_code=400, detail="No video metadata provided")
            
        # Insert video metadata and get video ID
        video_id = insert_video(video_data)
        if not video_id:
            raise HTTPException(status_code=500, detail="Failed to insert video metadata")
            
        # Insert frame metadata
        frames_data = metadata.get("frames", {}).get("details", [])
        frames_result = insert_frames(video_id, frames_data)
        logging.info(f"Inserted frames for video ID {video_id}")
        # Return the result
        return {
            "status": "success",
            "video_id": video_id,
            "frames": frames_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store metadata: {str(e)}")


async def call_omniparser(base64_image: str) -> Dict[str, Any]:
    """
    Call the Omniparser API with a base64 encoded image
    """
    omniparser_url = f"http://{os.getenv("OMNIPARSER_HOST")}:{os.getenv("OMNIPARSER_PORT")}/parse/"
    
    # Send the base64 image to the Florence API
    response = requests.post(omniparser_url, json={"base64_image": base64_image})
    if response.status_code == 200:
        json_response = response.json()
        for item in json_response['parsed_content_list']:
            if 'bbox' in item:
                del item['bbox']
        return json_response
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Omniparser API error: {response.text}")


async def update_image_review_and_textvector(preprocess_result: Dict[str, Any]):
    """
    Process images with Azure AI and update embeddings in the database.
    """
    global _thread_manager
    
    logging.info(f"Updating image review and text vector for video ID: {preprocess_result['video_id']}")    
    review_images = get_frames_by_video_id(preprocess_result['video_id'])
    logging.info(f"Found {len(review_images)} frames for video ID {preprocess_result['video_id']}")

    AzureOpenAISettings(
        embedding_deployment_name=os.getenv("AZURE_AI_AGENT_EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-large"),
        text_deployment_name=os.getenv("AZURE_AI_AGENT_CHAT_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
        api_key= os.getenv("AZREU_TEXT_EMBEDDING_APIKEY", "MISSING"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZREU_TEXT_EMBEDDING_API_VERSION", "2024-12-01-preview"),
    )

    azuretxtembedding = AzureTextEmbedding(
        service_id="azure-text-embedding"
        )
    
    logging.info(f"Using Azure Text Embedding with deployment name: {azuretxtembedding}")
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING", "MISSING"))
    
    image_analyzer_agent = await get_image_analyzer_agent()
    if not image_analyzer_agent:
        raise HTTPException(status_code=500, detail="Image analyzer agent not found or failed to initialize")
    logging.info(f"Using image analyzer agent: {image_analyzer_agent}")

    # With proper thread management:
    # Use the image ID as thread metadata for traceability
    thread_metadata = {
        "video_id": str(preprocess_result['video_id']),
        "operation": "image_analysis"
    }
    
    # Get or create a thread for this image
    thread = await _thread_manager.get_or_create_thread(
        agent=image_analyzer_agent,
        metadata=thread_metadata
    )

    for image in review_images:
        logging.info(f"Processing image {image['id']} with URL: {image.get('url', 'N/A')} and blob name: {image.get('blob_name', 'N/A')}")
        try:
            # read the image from Azure Blob Storage
            container_client = blob_service_client.get_container_client(image["container_name"])
            blob_client = container_client.get_blob_client(image["blob_name"])
            blob_data = blob_client.download_blob().readall()

            # Convert the image to base64
            base64_image = base64.b64encode(blob_data).decode('utf-8')

            # Call the Omniparser API with the base64 image
            omniparser_response = await call_omniparser(base64_image)
            # logging.info(f"Omniparser response for image {image['id']}: {omniparser_response}")
            som_image_base64 = omniparser_response.pop('som_image_base64', None)
            # remove 'bbox' from parsed_content if it exists
            item_list = []
            for item in omniparser_response['parsed_content_list']:
                if 'bbox' in item:
                    del item['bbox']
                item_list.append(f"type:{item['type']} interactivity:{item['interactivity']} content:{item['content']}")
            item_set = set(item_list)
            # item_set_list = list(item_set)
            
            # Log the thread ID for traceability
            if thread:
                logging.info(f"Using thread {thread.id} for image {image['id']}")
            
            # Use the thread in your agent call
            image_analysis_result_by_agent = await image_analyzer_agent_run(
                agent=image_analyzer_agent, 
                base64_image=som_image_base64, 
                omniparser_response=item_set, 
                thread=thread
            )
            logging.info(f"Image analysis result: {image_analysis_result_by_agent}")

            # Update the frame with the description and embedding
            embedding_vector = await azuretxtembedding.generate_embeddings([str(image_analysis_result_by_agent.content.content)])
            logging.info(f"Generated embedding vector for image {image['id']}: {embedding_vector}")
            if update_frame_with_embedding(image["id"], str(image_analysis_result_by_agent.content.content), embedding_vector[0].tolist()):
                logging.info(f"Updated frame {image['id']} with description and embedding")
            else:
                logging.error(f"Failed to update frame {image['id']} with description and embedding")
                raise HTTPException(status_code=500, detail=f"Failed to update frame {image['id']} with description and embedding")
            
        except Exception as e:
            logging.error(f"Failed to analyze image {image.get('url')}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"{str(e)}")
    # close image_analyzer_agent agent
    # await image_analyzer_agent.client.close()

    logging.info(f"Completed updating image review and text vector for video ID: {preprocess_result['video_id']}")
    return preprocess_result

@orchestrator.post("/update-video-review-and-text/")
async def update_video_review_and_text(preprocess_result: Dict[str, Any]):
    """
    Get image_descriptions from table, frames in database
    Use the image_descriptions as the input for agent which will generate video description with security review indicator
    """
    logging.info(f"Updating video review and text for video ID: {preprocess_result}")
    
    try:
        # Get the security reviewer agent
        logging.info("Retrieving security reviewer agent...")
        security_reviewer_agent = await get_security_reviewer_agent()

        if not security_reviewer_agent:
            logging.error("Security reviewer agent not found or failed to initialize")
            raise HTTPException(status_code=500, detail="Security reviewer agent not found or failed to initialize")
        
        logging.info(f"Using security reviewer agent: {security_reviewer_agent}")
        # Use the image ID as thread metadata for traceability
        thread_metadata = {
            "video_id": str(preprocess_result['video_id']),
            "operation": "image_analysis"
        }
        
        # Get or create a thread for this image
        thread = await _thread_manager.get_or_create_thread(
            agent=security_reviewer_agent,
            metadata=thread_metadata
        )

        # Log the thread ID for traceability
        if thread:
            logging.info(f"Using thread {thread.id} for security_reviewer_agent {str(preprocess_result['video_id'])}")
        
        # Run the security reviewer agent with proper error handling
        custom_prompt = """
        Review the Remote Desktop session recordings frame descriptions below.
        Identify any security concerns such as:
        - Unauthorized access
        - Sensitive information exposure
        - Suspicious activities
        - Potential policy violations
        
        Provide a security assessment and assign a risk level (Low, Medium, High).
        """
        if thread is None:
            # Get frames for the video
            review_images = get_frames_by_video_id(preprocess_result['video_id'])
            logging.info(f"Found {len(review_images)} frames for video ID {review_images}")
            
            # Prepare the input for the security reviewer agent - format as list of dictionaries
            image_analysis_results = []
            for img in review_images:
                if img.get("image_description"):
                    image_analysis_results.append({
                        "id": img.get("id"),
                        "frame_timestamp": img.get("frame_timestamp", 0),
                        "image_description": img.get("image_description", "")
                    })
            
            if len(image_analysis_results) < 1:
                logging.warning(f"No image descriptions found for video ID {preprocess_result['video_id']}")
                return preprocess_result
                
            security_review_result = await security_reviewer_agent_run(
                agent=security_reviewer_agent, 
                image_analysis_results=image_analysis_results,
                custom_prompt=custom_prompt,
                thread=thread
            )
        else:
            security_review_result = await security_reviewer_agent_run(
                agent=security_reviewer_agent, 
                custom_prompt=custom_prompt,
                thread=thread
            )
        
        if not security_review_result:
            raise HTTPException(status_code=500, detail="Failed to run security reviewer agent")
        
        logging.info(f"Security review result: {security_review_result}")
        
        # Extract the content from the result object based on expected return type
        review_content = ""
        if hasattr(security_review_result, 'content') and hasattr(security_review_result.content, 'content'):
            review_content = security_review_result.content.content
        elif hasattr(security_review_result, 'content'):
            review_content = security_review_result.content
        else:
            review_content = str(security_review_result)
        
        # Update the video metadata with the security review result
        if update_video_with_security_review(preprocess_result['video_id'], review_content):
            logging.info(f"Successfully updated video {preprocess_result['video_id']} with security review")
            
        else:
            logging.error(f"Failed to update video {preprocess_result['video_id']} with security review result")
            raise HTTPException(status_code=500, detail=f"Failed to update video {preprocess_result['video_id']} with security review result")
    
    except Exception as e:
        logging.error(f"Error in update_video_review_and_text: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process security review: {str(e)}")
    
    await security_reviewer_agent.client.close()
    return preprocess_result


@orchestrator.post("/ingest-video/")
async def ingest_video(file: UploadFile = File(...), container_name: str = Form("videos"), frames_container: str = Form("frames")):
    """
    Ingest a video file, split it into frames, and upload to Azure Blob Storage.
    """
    temp_filename = None
    frame_paths = []
    
    try:
        # Upload the original video to blob storage
        video_blob = await upload_to_blob_storage(file, container_name)
        
        # Create a temporary file with secure naming
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.join(temp_dir, f"temp_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        
        # Reset file cursor position
        await file.seek(0)
        
        # Save uploaded file to temp location
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(await file.read())
        
        # Split the video into frames
        frame_paths = split_video(
            video_path=temp_filename, 
            blob_conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING","MISSING"),
            output_dir=frames_container,
            # interval=3
        )
        
        # Process information about uploaded frames
        uploaded_frames = []
        for frame_path in frame_paths:
            if os.path.isfile(frame_path):
                frame_basename = os.path.basename(frame_path)
                video_name = os.path.splitext(file.filename)[0] if file.filename else "video"
                
                # Record frame info
                uploaded_frames.append({
                    # "local_path": frame_path,
                    "blob_name": f"{video_name}/{frame_basename}",
                    "blob_url": f"{frame_path}",
                    "container_name": f"{frames_container}"
                })
                
                # Clean up local frame file
                try:
                    os.remove(frame_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not remove frame file {frame_path}: {cleanup_error}")

        logging.info(f"Uploaded {len(frame_paths)} details '{frame_paths}'")
        # Clean up the temporary video file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        # Return information about the uploaded video and frames
        ingested_video_frame_metadata = {
            "video": video_blob,
            "frames": {
                "count": len(frame_paths),
                "container": frames_container,
                "details": frame_paths
            }
        }

        preprocess_result = await store_video_frame_metadata(ingested_video_frame_metadata)
        preprocess_result = await update_image_review_and_textvector(preprocess_result)
        logging.info(f"Ingested video and frames metadata: {ingested_video_frame_metadata}")

        return await update_video_review_and_text(preprocess_result)

        
    except Exception as e:
        # Clean up temporary files in case of error
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
                
        # Clean up any remaining frame files
        for frame_path in frame_paths:
            if isinstance(frame_path, str) and os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except:
                    pass
                    
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


def split_video(video_path: str, blob_conn_str: str, output_dir: str=r"preprocess", interval=3):
    """
    Split a video into frames and upload directly to Azure Blob Storage.
    
    Args:
        video_path (str): Path to the input video file.
        blob_conn_str (str): Azure Blob Storage connection string.
        output_dir (str): Container name for the output frames.
        interval (int): Interval in seconds to split the video.
        
    Returns:
        list: List of paths to the saved frames.
    """
    frame_blob_paths = []
    
    try:
        # Check if the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        # Create local temp directory for frames if needed
        local_temp_dir = os.path.join(os.getcwd(), "temp_frames")
        os.makedirs(local_temp_dir, exist_ok=True)
        
        # Get video file name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Initialize Azure Blob Storage client
        blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service_client.get_container_client(output_dir)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid frame rate: {fps}")
            
        frame_interval = int(fps * interval)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame every 'frame_interval' frames
            if frame_count % frame_interval == 0:
                # Calculate the time in seconds
                time_in_seconds = frame_count / fps
                
                # Create a filename with the time in seconds
                frame_basename = f"{video_name}_{int(time_in_seconds):0>10}.jpg"
                local_frame_path = os.path.join(local_temp_dir, frame_basename)
                
                # Save frame locally first
                cv2.imwrite(local_frame_path, frame)
                
                # Create blob name with logical hierarchy
                blob_name = f"{video_name}/{frame_basename}"
                
                # Upload to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob_name)
                
                with open(local_frame_path, "rb") as frame_file:
                    # Upload with optimized settings for images
                    blob_client.upload_blob(
                        frame_file.read(),
                        overwrite=True,
                        content_settings=ContentSettings(
                            content_type="image/jpeg",
                            cache_control="max-age=86400"
                        )
                    )
                    frame_blob_paths.append(blob_client.url)

                
                # Explicitly release memory
                del frame
                gc.collect()

            frame_count += 1

        # Release the video capture object
        cap.release()
        # Clean up local frame files
        for frame_path in os.listdir(local_temp_dir):
            full_frame_path = os.path.join(local_temp_dir, frame_path)
            if os.path.isfile(full_frame_path):
                try:
                    os.remove(full_frame_path)
                except Exception as e:
                    print(f"Warning: Could not remove local frame file {full_frame_path}: {str(e)}")

        # Return the list of saved frame paths
        return frame_blob_paths

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Return any frames that were successfully processed
        return frame_blob_paths
        
    finally:
        # Ensure resources are released
        if 'cap' in locals() and cap is not None:
            cap.release()
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Only start the server if initialization was successful
    # Start the FastAPI server
    print("Starting Orchestrator API...")
    async def startup():
        """Initialize the application and run startup tests"""
        print("Starting Orchestrator API...")
        # Properly await the async function
        return True
    
    # Run the async initialization
    loop = asyncio.get_event_loop()
    init_success = loop.run_until_complete(startup())

    # Start the FastAPI server after initialization
    if init_success:
        uvicorn.run(orchestrator, host="localhost", port=8082, log_level="info")
    else:
        print("Initialization failed, server not started")