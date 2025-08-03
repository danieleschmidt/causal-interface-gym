"""Database connection management."""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from urllib.parse import urlparse

try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and connection pooling."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite:///causal_gym.db"
        )
        self.connection_pool: Optional[Any] = None
        self.db_type = self._detect_db_type()
        self._initialize_connection_pool()
    
    def _detect_db_type(self) -> str:
        """Detect database type from URL.
        
        Returns:
            Database type (postgresql, sqlite)
        """
        parsed = urlparse(self.database_url)
        
        if parsed.scheme in ('postgresql', 'postgres'):
            if not POSTGRES_AVAILABLE:
                logger.warning("PostgreSQL requested but psycopg2 not available, falling back to SQLite")
                return 'sqlite'
            return 'postgresql'
        elif parsed.scheme == 'sqlite':
            if not SQLITE_AVAILABLE:
                raise RuntimeError("SQLite not available")
            return 'sqlite'
        else:
            logger.warning(f"Unknown database scheme {parsed.scheme}, falling back to SQLite")
            return 'sqlite'
    
    def _initialize_connection_pool(self) -> None:
        """Initialize connection pool based on database type."""
        if self.db_type == 'postgresql':
            try:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dsn=self.database_url
                )
                logger.info("PostgreSQL connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL pool: {e}")
                # Fallback to SQLite
                self.db_type = 'sqlite'
                self.database_url = "sqlite:///causal_gym.db"
        
        if self.db_type == 'sqlite':
            # SQLite doesn't need pooling, we'll create connections as needed
            db_path = self.database_url.replace('sqlite:///', '')
            self._ensure_sqlite_db(db_path)
            logger.info(f"SQLite database initialized at {db_path}")
    
    def _ensure_sqlite_db(self, db_path: str) -> None:
        """Ensure SQLite database exists and has required tables.
        
        Args:
            db_path: Path to SQLite database file
        """
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Create database file if it doesn't exist
        conn = sqlite3.connect(db_path)
        conn.close()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection.
        
        Yields:
            Database connection
        """
        if self.db_type == 'postgresql' and self.connection_pool:
            conn = self.connection_pool.getconn()
            try:
                yield conn
            finally:
                self.connection_pool.putconn(conn)
        else:
            # SQLite
            db_path = self.database_url.replace('sqlite:///', '')
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query and return results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # For SELECT queries, fetch results
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    # For INSERT/UPDATE/DELETE, commit and return affected rows
                    conn.commit()
                    return cursor.rowcount
            except Exception as e:
                conn.rollback()
                logger.error(f"Query execution failed: {e}")
                raise
            finally:
                cursor.close()
    
    def create_tables(self) -> None:
        """Create required database tables."""
        if self.db_type == 'postgresql':
            self._create_postgresql_tables()
        else:
            self._create_sqlite_tables()
    
    def _create_postgresql_tables(self) -> None:
        """Create PostgreSQL tables."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) UNIQUE NOT NULL,
                agent_type VARCHAR(255) NOT NULL,
                causal_graph JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS belief_measurements (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                belief_statement TEXT NOT NULL,
                condition_type VARCHAR(255) NOT NULL,
                belief_value REAL NOT NULL,
                timestamp_order INTEGER NOT NULL,
                measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS intervention_records (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
                variable_name VARCHAR(255) NOT NULL,
                intervention_value JSONB NOT NULL,
                intervention_type VARCHAR(255) DEFAULT 'do',
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result JSONB DEFAULT '{}'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_id ON experiments(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_beliefs_experiment ON belief_measurements(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_interventions_experiment ON intervention_records(experiment_id);
            """
        ]
        
        for table_sql in tables:
            self.execute_query(table_sql)
    
    def _create_sqlite_tables(self) -> None:
        """Create SQLite tables."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                agent_type TEXT NOT NULL,
                causal_graph TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS belief_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                belief_statement TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                belief_value REAL NOT NULL,
                timestamp_order INTEGER NOT NULL,
                measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS intervention_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                variable_name TEXT NOT NULL,
                intervention_value TEXT NOT NULL,
                intervention_type TEXT DEFAULT 'do',
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT DEFAULT '{}',
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_id ON experiments(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_beliefs_experiment ON belief_measurements(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_interventions_experiment ON intervention_records(experiment_id);
            """
        ]
        
        for table_sql in tables:
            self.execute_query(table_sql)
    
    def close(self) -> None:
        """Close database connections."""
        if self.db_type == 'postgresql' and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()