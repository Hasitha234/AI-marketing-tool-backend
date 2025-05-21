from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.utils.logging import get_logger

# Initialize logger
logger = get_logger("database")

logger.info("Initializing database engine")
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping = True,
    echo = True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("Database session factory created")

def get_db():
    """ Dependency to get a database session """
    logger.debug("Creating new database session")
    db = SessionLocal()
    try:
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()
