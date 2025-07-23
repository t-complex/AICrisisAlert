#!/usr/bin/env python3
"""
Database initialization script for AICrisisAlert.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alembic.config import Config
from alembic import command
from src.api.database import engine, Base
from src.api.config import get_settings

def init_database():
    """Initialize the database with tables and migrations."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting database initialization...")
    
    try:
        # Get settings
        settings = get_settings()
        logger.info(f"Database URL: {settings.database_url}")
        
        # Create tables if they don't exist
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Run Alembic migrations
        logger.info("Running database migrations...")
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
        
        # Verify database connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully")
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        sys.exit(1)

def reset_database():
    """Reset the database (drop all tables and recreate)."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.warning("This will drop all tables and recreate them. Are you sure? (y/N): ")
    response = input().strip().lower()
    
    if response != 'y':
        logger.info("Database reset cancelled.")
        return
    
    try:
        # Drop all tables
        logger.info("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        
        # Recreate tables
        logger.info("Recreating tables...")
        Base.metadata.create_all(bind=engine)
        
        # Run migrations
        logger.info("Running migrations...")
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Database reset completed successfully!")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database management for AICrisisAlert")
    parser.add_argument("--reset", action="store_true", help="Reset database (drop and recreate)")
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        init_database() 