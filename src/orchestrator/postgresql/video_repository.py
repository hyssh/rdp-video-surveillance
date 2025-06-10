import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def insert_video(video_data: Dict[str, Any]) -> Optional[int]:
    """
    Insert video metadata into the database following Azure PostgreSQL best practices
    
    Args:
        video_data: Dictionary containing video metadata
        
    Returns:
        int: ID of the inserted video or None if insertion failed
    """
    from .connection import DatabaseManager
    
    if not video_data:
        logger.error("No video data provided for insertion")
        return None
    
    insert_sql = """
        INSERT INTO videos 
            (filename, blob_name, container_name, size_bytes, content_type, url)
        VALUES 
            (%(filename)s, %(blob_name)s, %(container_name)s, %(size_bytes)s, %(content_type)s, %(url)s)
        RETURNING id
    """
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(
                insert_sql, 
                {
                    "filename": video_data.get("filename"),
                    "blob_name": video_data.get("blob_name"),
                    "container_name": video_data.get("container_name"),
                    "size_bytes": video_data.get("size_bytes"),
                    "content_type": video_data.get("content_type"),
                    "url": video_data.get("url")
                }
            )
            video_id = cursor.fetchone()[0]
            logger.info(f"Video metadata inserted successfully with ID: {video_id}")
            return video_id
    except Exception as e:
        logger.error(f"Failed to insert video metadata: {str(e)}")
        return None

def get_video_by_id(video_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve video metadata by ID
    
    Args:
        video_id: ID of the video to retrieve
        
    Returns:
        Dict: Video metadata or None if not found
    """
    from .connection import DatabaseManager
    
    select_sql = """
        SELECT id, filename, blob_name, container_name, size_bytes, content_type, url, created_at
        FROM videos
        WHERE id = %s
    """
    
    try:
        with DatabaseManager() as cursor:
            cursor.execute(select_sql, (video_id,))
            video = cursor.fetchone()
            return dict(video) if video else None
    except Exception as e:
        logger.error(f"Failed to retrieve video with ID {video_id}: {str(e)}")
        return None
    
def update_video_with_security_review(video_id: int, description: str) -> bool:
    """
    Update the description of a video by its ID
    Args:
        video_id: ID of the video to update
        description: New description for the video
    """    
    from .connection import DatabaseManager
    update_sql = """
        UPDATE videos
        SET video_description = %s
        WHERE id = %s
    """
    try:
        with DatabaseManager() as cursor:
            cursor.execute(update_sql, (description, video_id))
            if cursor.rowcount > 0:
                logger.info(f"Video with ID {video_id} updated successfully")
                return True
            else:
                logger.warning(f"No video found with ID {video_id} to update")
                return False
    except Exception as e:
        logger.error(f"Failed to update video with ID {video_id}: {str(e)}")
        return False