import logging
import os
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)

def insert_frames(video_id: int, frames_data: List[Any]) -> Dict[str, Any]:
    """
    Batch insert frame metadata for a video following Azure PostgreSQL best practices.
    Handles both dictionary frames and URL string frames.
    
    Args:
        video_id: ID of the video these frames belong to
        frames_data: List of frame data (either URLs as strings or dictionaries)
        
    Returns:
        Dict: Summary of the operation
    """
    from .connection import DatabaseManager
    
    if not frames_data:
        logger.warning("No frames data provided for insertion")
        return {"status": "warning", "message": "No frames data provided", "count": 0}
    
    insert_sql = """
        INSERT INTO frames 
            (video_id, blob_name, container_name, url, frame_timestamp)
        VALUES 
            (%(video_id)s, %(blob_name)s, %(container_name)s, %(url)s, %(frame_timestamp)s)
    """
    
    try:
        frame_records = []
        for frame in frames_data:
            # Handle case where frame is a URL string
            if isinstance(frame, str):
                url = frame
                
                # Parse container and blob name from URL
                # URL format: https://accountname.blob.core.windows.net/container/path/to/blob
                parsed_url = urlparse(url)
                path_parts = parsed_url.path.strip('/').split('/')
                
                if len(path_parts) >= 2:
                    container_name = path_parts[0]
                    blob_name = '/'.join(path_parts[1:])
                    
                    # Try to extract timestamp from filename
                    frame_timestamp = None
                    try:
                        # Find the last numeric part before the extension
                        filename = os.path.basename(blob_name)
                        if '_' in filename:
                            timestamp_str = filename.split('_')[-1].split('.')[0]
                            frame_timestamp = int(timestamp_str)
                    except (ValueError, IndexError):
                        pass
                    
                    frame_records.append({
                        "video_id": video_id,
                        "blob_name": blob_name,
                        "container_name": container_name,
                        "url": url,
                        "frame_timestamp": frame_timestamp
                    })
                else:
                    logger.warning(f"Could not parse container and blob name from URL: {url}")
                    
            # Handle case where frame is a dictionary
            elif isinstance(frame, dict):
                blob_name = frame.get("blob_name", "")
                frame_timestamp = None
                
                # Try to extract timestamp from filename
                try:
                    if "_" in blob_name:
                        # Assuming format like video_name/filename_0000000075.jpg
                        parts = blob_name.split("/")
                        if len(parts) >= 2:
                            filename = parts[1]
                            if "_" in filename:
                                timestamp_str = filename.split("_")[-1].split(".")[0]
                                frame_timestamp = int(timestamp_str)
                except (ValueError, IndexError):
                    pass
                
                frame_records.append({
                    "video_id": video_id,
                    "blob_name": blob_name,
                    "container_name": frame.get("container_name"),
                    "url": frame.get("url") or frame.get("blob_url"),
                    "frame_timestamp": frame_timestamp
                })
            else:
                logger.warning(f"Skipping frame with unsupported type: {type(frame)}")
        
        with DatabaseManager() as cursor:
            # Use batch insert for better performance (Azure recommended practice)
            execute_batch(cursor, insert_sql, frame_records)
            
        logger.info(f"Successfully inserted {len(frame_records)} frames for video ID {video_id}")
        return {
            "status": "success",
            "message": f"Inserted {len(frame_records)} frames",
            "count": len(frame_records)
        }
        
    except Exception as e:
        logger.error(f"Failed to insert frame metadata: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "count": 0
        }

def get_frames_by_video_id(video_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve all frames for a specific video
    
    Args:
        video_id: ID of the video to retrieve frames for
        
    Returns:
        List[Dict]: List of frame metadata dictionaries
    """
    from .connection import DatabaseManager
    
    select_sql = """
        SELECT id, video_id, blob_name, container_name, url, frame_timestamp, created_at, image_description
        FROM frames
        WHERE video_id = %s
        ORDER BY frame_timestamp
    """
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(select_sql, (video_id,))
            frames = cursor.fetchall()
            return [dict(frame) for frame in frames]
    except Exception as e:
        logger.error(f"Failed to retrieve frames for video ID {video_id}: {str(e)}")
        return []

def update_frame_with_embedding(frame_id: int, description: str, embedding_vector: list) -> bool:
    """
    Update a frame with image description and embedding vector
    
    Args:
        frame_id: ID of the frame to update
        description: Text description of the image
        embedding_vector: Vector representation of the image (1536 dimensions)
        
    Returns:
        bool: True if update was successful
    """
    from .connection import DatabaseManager
    
    update_sql = """
        UPDATE frames
        SET image_description = %s, embedding = %s
        WHERE id = %s
    """
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(update_sql, (description, embedding_vector, frame_id))
            rows_affected = cursor.rowcount
            logger.info(f"Updated frame {frame_id} with embedding vector")
            return rows_affected > 0
    except Exception as e:
        logger.error(f"Failed to update frame with embedding: {str(e)}")
        return False

def search_similar_frames(query_vector: list, threshold: float = 0.75, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for frames similar to the provided vector
    
    Args:
        query_vector: Vector to search for (1536 dimensions)
        threshold: Similarity threshold (0-1, higher means more similar)
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict]: List of similar frames with similarity scores
    """
    from .connection import DatabaseManager
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(
                "SELECT * FROM search_similar_frames(%s, %s, %s)",
                (query_vector, threshold, max_results)
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to search similar frames: {str(e)}")
        return []

def get_frames_without_embeddings(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get frames that don't have embedding vectors yet
    
    Args:
        limit: Maximum number of frames to return
        
    Returns:
        List[Dict]: List of frames without embeddings
    """
    from .connection import DatabaseManager
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(
                """
                SELECT id, video_id, blob_name, container_name, url
                FROM frames
                WHERE embedding IS NULL
                LIMIT %s
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get frames without embeddings: {str(e)}")
        return []