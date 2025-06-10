import logging

logger = logging.getLogger(__name__)

def initialize_schemas():
    """
    Initialize database schemas for video surveillance
    Following Azure PostgreSQL best practices for schema design
    """
    from .connection import DatabaseManager
    
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
        """
        CREATE TABLE IF NOT EXISTS frames (
            id SERIAL PRIMARY KEY,
            video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
            blob_name VARCHAR(255) NOT NULL,
            container_name VARCHAR(100) NOT NULL,
            url TEXT,
            frame_timestamp INTEGER,
            image_description TEXT,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames(video_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_frames_blob_name ON frames(blob_name)
        """,
        """
        -- Create a GIN index for vector similarity search
        CREATE INDEX IF NOT EXISTS idx_frames_embedding ON frames USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100)
        """
    ]
    
    try:
        with DatabaseManager() as cursor:
            for sql in create_tables_sql:
                cursor.execute(sql)
        logger.info("Database schema initialized successfully with vector support")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {str(e)}")
        return False

def add_vector_search_functions():
    """
    Add helper functions for vector similarity search
    """
    from .connection import DatabaseManager
    
    functions_sql = [
        """
        -- Function to find similar images by vector similarity
        CREATE OR REPLACE FUNCTION search_similar_frames(
            query_vector vector(1536),
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
                1 - (f.embedding <-> query_vector) AS similarity
            FROM 
                frames f
            WHERE 
                f.embedding IS NOT NULL
            ORDER BY 
                f.embedding <-> query_vector
            LIMIT max_results;
        END;
        $$;
        """
    ]
    
    try:
        with DatabaseManager() as cursor:
            for sql in functions_sql:
                cursor.execute(sql)
        logger.info("Vector search functions created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector search functions: {str(e)}")
        return False