import os
import psycopg2
from psycopg2.extras import RealDictCursor
from lib.logger import log_error, log_info

def get_db_connection():
    """Establish and return a connection to the PostgreSQL database using environment variables."""
    db_host = os.environ.get("POSTGRES_HOST", "db")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "worm_library_db")
    db_user = os.environ.get("POSTGRES_USER", "worm_app_user")
    db_pass = os.environ.get("POSTGRES_PASSWORD", "")

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_pass,
            connect_timeout=5
        )
        return conn
    except Exception as e:
        log_error(f"Database connection error: {str(e)}")
        raise

def execute_query(query, params=None, fetch_all=True, fetch_one=False, commit=False):
    """Safely execute parameterized SQL queries."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            result = None
            if cur.description:
                if fetch_one:
                    result = cur.fetchone()
                elif fetch_all:
                    result = cur.fetchall()

            if commit:
                conn.commit()
            return result
    except Exception as e:
        if conn and commit:
            conn.rollback()
        log_error(f"SQL execution failure: {str(e)} | Query: {query[:100]}")
        raise
    finally:
        if conn:
            conn.close()
