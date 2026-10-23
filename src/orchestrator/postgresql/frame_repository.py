import logging
import os
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple
from psycopg2.extras import execute_batch, execute_values

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

def update_frames_with_embeddings(updates: List[Tuple[int, str, list]], page_size: int = 100) -> int:
    """
    Bulk update frames with image description and embedding in a single transaction.

    Args:
        updates: List of (frame_id, description, embedding_vector) tuples
        page_size: Rows per statement sent to the database

    Returns:
        int: Number of frames updated (0 on failure or empty input)
    """
    from .connection import DatabaseManager

    if not updates:
        return 0

    update_sql = """
        UPDATE frames AS f
        SET image_description = v.description, embedding = v.embedding::vector
        FROM (VALUES %s) AS v(id, description, embedding)
        WHERE f.id = v.id
        RETURNING f.id
    """

    try:
        rows = [(fid, desc, [float(x) for x in vec]) for fid, desc, vec in updates]
        with DatabaseManager() as cursor:
            updated = execute_values(
                cursor, update_sql, rows,
                template="(%s::int, %s::text, %s::float8[])",
                page_size=page_size, fetch=True,
            )
            logger.info(f"Bulk updated {len(updated)} frames with embedding vectors")
            return len(updated)
    except Exception as e:
        logger.error(f"Failed to bulk update frames with embeddings: {str(e)}")
        return 0

def list_frames(video_id: int, limit: int = 50, offset: int = 0):
    """
    Return (rows, total) for a page of a video's frames ordered by timestamp. Raises on database errors.
    """
    from .connection import DatabaseManager

    with DatabaseManager() as cursor:
        cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
        total = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT id, frame_timestamp, image_description, url
            FROM frames
            WHERE video_id = %s
            ORDER BY frame_timestamp, id
            LIMIT %s OFFSET %s
            """,
            (video_id, limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()], total

def search_similar_frames(query_vector: list, threshold: Optional[float] = 0.75, max_results: int = 10,
                          video_id: Optional[int] = None, raise_on_error: bool = False) -> List[Dict[str, Any]]:
    """
    Search for frames similar to the provided vector
    
    Args:
        query_vector: Vector to search for (1536 dimensions)
        threshold: Similarity threshold (0-1, higher means more similar). Only applied when
            video_id is given (the legacy SQL function ignores it); None disables it.
        max_results: Maximum number of results to return
        video_id: Optionally restrict the search to a single video
        raise_on_error: Raise database errors instead of returning []
        
    Returns:
        List[Dict]: List of similar frames with similarity scores
    """
    from .connection import DatabaseManager
    
    try:
        with DatabaseManager() as cursor:
            if video_id is None:
                cursor.execute(
                    "SELECT * FROM search_similar_frames(%s, %s, %s)",
                    (query_vector, 0.75 if threshold is None else threshold, max_results)
                )
            else:
                # pgvector text literal; similarity is cosine for unit-length embeddings (L2 ops keep the IVFFlat index usable)
                vector_literal = "[" + ",".join(repr(float(x)) for x in query_vector) + "]"
                cursor.execute(
                    """
                    SELECT * FROM (
                        SELECT id, video_id, blob_name, url, frame_timestamp, image_description,
                               1 - ((embedding <-> %(q)s::vector) ^ 2) / 2 AS similarity
                        FROM frames
                        WHERE video_id = %(video_id)s AND embedding IS NOT NULL
                        ORDER BY embedding <-> %(q)s::vector
                        LIMIT %(limit)s
                    ) ranked
                    WHERE %(threshold)s::float IS NULL OR similarity >= %(threshold)s::float
                    ORDER BY similarity DESC
                    """,
                    {"q": vector_literal, "video_id": video_id, "limit": max_results, "threshold": threshold},
                )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to search similar frames: {str(e)}")
        if raise_on_error:
            raise
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