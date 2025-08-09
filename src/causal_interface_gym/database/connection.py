"""Database connection management."""

import os
import logging
import time
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


class DatabaseError(Exception):
    """Database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Database connection errors."""
    pass


class QueryError(DatabaseError):
    """Query execution errors."""
    pass


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
    
    def execute_query(self, query: str, params: Optional[tuple] = None, retry_count: int = 3) -> Any:
        """Execute a query with retry logic and comprehensive error handling.
        
        Args:
            query: SQL query
            params: Query parameters
            retry_count: Number of retry attempts
            
        Returns:
            Query results
            
        Raises:
            DatabaseError: If query fails after retries
        """
        last_exception = None
        
        for attempt in range(retry_count):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    try:
                        # Log query for debugging (sanitized)
                        sanitized_query = self._sanitize_query_for_logging(query)
                        logger.debug(f"Executing query (attempt {attempt + 1}): {sanitized_query}")
                        
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        
                        # For SELECT queries, fetch results
                        if query.strip().upper().startswith('SELECT'):
                            results = cursor.fetchall()
                            logger.debug(f"Query returned {len(results) if results else 0} rows")
                            return results
                        else:
                            # For INSERT/UPDATE/DELETE, commit and return affected rows
                            conn.commit()
                            affected_rows = cursor.rowcount
                            logger.debug(f"Query affected {affected_rows} rows")
                            return affected_rows
                            
                    except Exception as e:
                        conn.rollback()
                        raise
                    finally:
                        cursor.close()
                        
            except Exception as e:
                last_exception = e
                if attempt < retry_count - 1:
                    import time
                    wait_time = (attempt + 1) * 0.5  # Exponential backoff
                    logger.warning(f"Query failed on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {retry_count} attempts: {e}")
        
        raise DatabaseError(f"Query failed after {retry_count} attempts: {last_exception}")
    
    def _sanitize_query_for_logging(self, query: str) -> str:
        """Sanitize query for safe logging (remove sensitive data).
        
        Args:
            query: Original query
            
        Returns:
            Sanitized query for logging
        """
        # Replace potential sensitive data patterns
        import re
        sanitized = re.sub(r"'[^']*'", "'***'", query)  # Hide string literals
        sanitized = re.sub(r'\$\d+', '$***', sanitized)  # Hide parameter placeholders
        return sanitized[:200] + '...' if len(sanitized) > 200 else sanitized
    
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
    
    def health_check(self) -> Dict[str, Any]:
        """Check database connection health.
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            result = self.execute_query("SELECT 1 as health_check")
            query_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "database_type": self.db_type,
                "query_time_ms": round(query_time * 1000, 2),
                "connection_pool_active": self.connection_pool is not None,
                "result": result is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": self.db_type,
                "error": str(e),
                "connection_pool_active": self.connection_pool is not None
            }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Connection statistics
        """
        stats = {
            "database_type": self.db_type,
            "database_url_scheme": urlparse(self.database_url).scheme
        }
        
        if self.db_type == 'postgresql' and self.connection_pool:
            try:
                stats.update({
                    "pool_size": self.connection_pool.maxconn,
                    "pool_available": len(self.connection_pool._available),
                    "pool_used": len(self.connection_pool._used)
                })
            except Exception as e:
                stats["pool_error"] = str(e)
        
        return stats
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup (SQLite only).
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if backup successful
        """
        if self.db_type != 'sqlite':
            logger.warning("Backup only supported for SQLite databases")
            return False
        
        try:
            import shutil
            db_path = self.database_url.replace('sqlite:///', '')
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space.
        
        Returns:
            True if vacuum successful
        """
        try:
            if self.db_type == 'sqlite':
                self.execute_query("VACUUM")
            elif self.db_type == 'postgresql':
                # Note: VACUUM cannot be run inside a transaction in PostgreSQL
                with self.get_connection() as conn:
                    conn.autocommit = True
                    cursor = conn.cursor()
                    cursor.execute("VACUUM ANALYZE")
                    cursor.close()
            
            logger.info("Database vacuumed successfully")
            return True
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections."""
        try:
            if self.db_type == 'postgresql' and self.connection_pool:
                self.connection_pool.closeall()
                logger.info("PostgreSQL connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()