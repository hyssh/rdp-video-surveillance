from .connection import init_connection_pool
from .schemas import initialize_schemas, add_vector_search_functions
from .video_repository import (
    insert_video, 
    get_video_by_id, 
    update_video_with_security_review,
    list_videos,
    get_video_detail,
    update_video_metrics,
    get_stats
)
from .job_repository import create_job, update_job, get_job, fail_interrupted_jobs
from .frame_repository import (
    insert_frames, 
    get_frames_by_video_id, 
    update_frame_with_embedding,
    update_frames_with_embeddings,
    search_similar_frames,
    get_frames_without_embeddings,
    list_frames,
    update_frames_analysis,
    get_frame,
    list_timeline_frames
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
    'update_frames_with_embeddings',
    'search_similar_frames',
    'get_frames_without_embeddings',
    'update_video_with_security_review',
    'list_videos',
    'get_video_detail',
    'list_frames',
    'create_job',
    'update_job',
    'get_job',
    'fail_interrupted_jobs',
    'update_video_metrics',
    'get_stats',
    'update_frames_analysis',
    'get_frame',
    'list_timeline_frames'
]