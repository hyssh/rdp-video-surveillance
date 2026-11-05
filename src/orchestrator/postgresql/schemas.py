import logging

logger = logging.getLogger(__name__)

# pgvector index limits: ivfflat/hnsw on `vector` up to 2000 dims, on `halfvec` up to 4000 dims.
MAX_VECTOR_INDEX_DIMS = 2000
MAX_HALFVEC_INDEX_DIMS = 4000


def get_embedding_dimensions() -> int:
    from azure_clients import get_embedding_dimensions as _get
    return _get()


def embedding_index_sql(dims: int):
    """Return (index_sql, distance_cast) for the embedding column.

    <=2000 dims: IVFFlat directly on vector (name kept for existing databases).
    2001-4000: HNSW on a halfvec expression (queries must use the same cast).
    >4000: no ANN index (exact scan).
    """
    if dims <= MAX_VECTOR_INDEX_DIMS:
        return (
            "CREATE INDEX IF NOT EXISTS idx_frames_embedding ON frames "
            "USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)",
            None,
        )
    if dims <= MAX_HALFVEC_INDEX_DIMS:
        return (
            "CREATE INDEX IF NOT EXISTS idx_frames_embedding_halfvec ON frames "
            f"USING hnsw ((embedding::halfvec({dims})) halfvec_l2_ops)",
            f"halfvec({dims})",
        )
    return ("", None)


def distance_cast(dims: int):
    return embedding_index_sql(dims)[1]


def initialize_schemas():
    """
    Initialize database schemas for video surveillance
    Following Azure PostgreSQL best practices for schema design.
    The embedding column size comes from EMBEDDING_DIMENSIONS (default 3072).
    """
    from .connection import DatabaseManager

    dims = get_embedding_dimensions()
    index_sql, _ = embedding_index_sql(dims)
    
    # SQL statements to create tables if they don't exist
    create_tables_sql = [
        """
        -- Ensure the vector extension is loaded
        CREATE EXTENSION IF NOT EXISTS vector;
        """,
        """
        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            blob_name VARCHAR(255) NOT NULL,
            container_name VARCHAR(100) NOT NULL,
            size_bytes BIGINT,
            content_type VARCHAR(100),
            url TEXT,
            video_description TEXT,
            security_level VARCHAR(50) DEFAULT 'low',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS frames (
            id SERIAL PRIMARY KEY,
            video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
            blob_name VARCHAR(255) NOT NULL,
            container_name VARCHAR(100) NOT NULL,
            url TEXT,
            frame_timestamp INTEGER,
            image_description TEXT,
            embedding vector({dims}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        # Migration: when EMBEDDING_DIMENSIONS differs from the existing column (e.g. legacy vector(1536)),
        # drop the vector indexes and retype the column. Existing embeddings are reset to NULL.
        f"""
        DO $$
        DECLARE cur_dims integer;
        BEGIN
            SELECT atttypmod INTO cur_dims FROM pg_attribute
            WHERE attrelid = 'frames'::regclass AND attname = 'embedding';
            IF cur_dims IS NOT NULL AND cur_dims <> {dims} THEN
                DROP INDEX IF EXISTS idx_frames_embedding;
                DROP INDEX IF EXISTS idx_frames_embedding_halfvec;
                ALTER TABLE frames ALTER COLUMN embedding TYPE vector({dims}) USING NULL;
            END IF;
        END $$
        """,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id UUID PRIMARY KEY,
            video_id INTEGER REFERENCES videos(id) ON DELETE SET NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'queued',
            stage VARCHAR(50),
            frames_total INTEGER NOT NULL DEFAULT 0,
            frames_done INTEGER NOT NULL DEFAULT 0,
            frames_failed INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT jobs_status_check CHECK (status IN ('queued','extracting','analyzing','summarizing','done','failed'))
        )
        """,
        """
        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS duration_s INTEGER,
            ADD COLUMN IF NOT EXISTS risk_score INTEGER
        """,
        """
        ALTER TABLE frames
            ADD COLUMN IF NOT EXISTS risk_score INTEGER,
            ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20),
            ADD COLUMN IF NOT EXISTS action_type VARCHAR(50),
            ADD COLUMN IF NOT EXISTS ui_elements JSONB,
            ADD COLUMN IF NOT EXISTS frame_width INTEGER,
            ADD COLUMN IF NOT EXISTS frame_height INTEGER
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames(video_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_frames_blob_name ON frames(blob_name)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_frames_video_id_timestamp ON frames(video_id, frame_timestamp)
        """
    ]
    if index_sql:
        create_tables_sql.append(index_sql)
    
    try:
        with DatabaseManager() as cursor:
            for sql in create_tables_sql:
                cursor.execute(sql)
        logger.info(f"Database schema initialized successfully with vector({dims}) support")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {str(e)}")
        return False

def build_search_function_sql(dims: int) -> str:
    cast = distance_cast(dims)
    if cast:
        distance = f"(f.embedding::{cast}) <-> (query_vector::{cast})"
    else:
        distance = "f.embedding <-> query_vector"
    return f"""
        -- Function to find similar images by vector similarity
        CREATE OR REPLACE FUNCTION search_similar_frames(
            query_vector vector,
            similarity_threshold float DEFAULT 0.75,
            max_results int DEFAULT 10
        )
        RETURNS TABLE (
            id int,
            video_id int,
            blob_name text,
            url text,
            image_description text,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                f.id,
                f.video_id,
                f.blob_name,
                f.url,
                f.image_description,
                1 - ({distance}) AS similarity
            FROM 
                frames f
            WHERE 
                f.embedding IS NOT NULL
            ORDER BY 
                {distance}
            LIMIT max_results;
        END;
        $$;
        """


def add_vector_search_functions():
    """
    Add helper functions for vector similarity search
    """
    from .connection import DatabaseManager
    
    functions_sql = [build_search_function_sql(get_embedding_dimensions())]
    
    try:
        with DatabaseManager() as cursor:
            for sql in functions_sql:
                cursor.execute(sql)
        logger.info("Vector search functions created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector search functions: {str(e)}")
        return False