from .connection import init_connection_pool
from .schemas import initialize_schemas, add_vector_search_functions
from .video_repository import (
    insert_video, 
    get_video_by_id, 
    update_video_with_security_review
)
from .frame_repository import (
    insert_frames, 
    get_frames_by_video_id, 
    update_frame_with_embedding,
    search_similar_frames,
    get_frames_without_embeddings
)

# Initialize the database when the module is imported
init_connection_pool()
initialize_schemas()
add_vector_search_functions()

__all__ = [
    'insert_video',
    'get_video_by_id',
    'insert_frames',
    'get_frames_by_video_id',
    'update_frame_with_embedding',
    'search_similar_frames',
    'get_frames_without_embeddings',
    'update_video_with_security_review'
]