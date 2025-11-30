"""
Database connection and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "arxiv",
        db_user: str = "postgres",
        db_password: str = ""
    ):
        """
        Initialize database manager.

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = None
        self.SessionLocal = None

    def create_engine_and_session(self, echo: bool = False):
        """
        Create database engine and session factory.

        Args:
            echo: If True, SQLAlchemy will log all SQL statements
        """
        self.engine = create_engine(
            self.db_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True  # Verify connections before using
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy Session object
        """
        if not self.SessionLocal:
            self.create_engine_and_session()
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db_manager.session_scope() as session:
                session.add(paper)
                # session automatically commits on success, rolls back on exception
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_url_for_alembic(self) -> str:
        """
        Get database URL for Alembic migrations.

        Returns:
            Database URL string
        """
        return self.db_url

    def close(self):
        """Close database engine and all connections"""
        if self.engine:
            self.engine.dispose()


def get_database_url_from_env() -> str:
    """
    Get database URL from environment variables.

    Environment variables:
        DB_HOST: PostgreSQL host (empty for Unix socket, 'localhost' for TCP/IP)
        DB_PORT: PostgreSQL port (default: 5432)
        DB_NAME: Database name (default: arxiv)
        DB_USER: Database user (default: postgres)
        DB_PASSWORD: Database password (default: empty)

    Returns:
        Database URL string
    """
    db_host = os.getenv('DB_HOST', '')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'arxiv')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')

    # Build connection URL
    # Empty host uses Unix socket (peer authentication)
    # 'localhost' uses TCP/IP (requires password)
    if db_host:
        # TCP/IP connection
        if db_password:
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            return f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
    else:
        # Unix socket connection
        if db_password:
            return f"postgresql://{db_user}:{db_password}@/{db_name}"
        else:
            return f"postgresql://{db_user}@/{db_name}"
