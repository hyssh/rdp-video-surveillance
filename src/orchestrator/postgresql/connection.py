import os
import logging
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection pool for efficient database access
connection_pool = None

def init_connection_pool(min_connections=1, max_connections=10):
    """Initialize PostgreSQL connection pool with Azure best practices"""
    global connection_pool
    
    try:
        # Get connection parameters from environment variables
        connection_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=os.getenv("PGHOST"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            database=os.getenv("PGDATABASE"),
            port=os.getenv("PGPORT"),
            sslmode='require'  # Required for Azure PostgreSQL security
        )
        logger.info("PostgreSQL connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}")
        raise


def get_connection():
    """Get a connection from the pool with proper error handling"""
    if connection_pool is None:
        init_connection_pool()
    
    try:
        connection = connection_pool.getconn()
        return connection
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {str(e)}")
        raise

def release_connection(connection):
    """Return a connection to the pool"""
    if connection_pool is not None:
        connection_pool.putconn(connection)

class DatabaseManager:
    """Context manager for handling database connections safely"""
    
    def __init__(self, dict_cursor=True):
        self.connection = None
        self.cursor = None
        self.dict_cursor = dict_cursor
        
    def __enter__(self):
        self.connection = get_connection()
        if self.dict_cursor:
            self.cursor = self.connection.cursor(cursor_factory=DictCursor)
        else:
            self.cursor = self.connection.cursor()
        return self.cursor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception occurred, rollback
            self.connection.rollback()
            logger.error(f"Transaction rolled back due to: {exc_val}")
        else:
            # No exception, commit
            self.connection.commit()
            
        # Close cursor and release connection
        if self.cursor:
            self.cursor.close()
        
        if self.connection:
            release_connection(self.connection)