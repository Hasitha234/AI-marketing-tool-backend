import logging
import os
import sys
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# JSON formatter for structured logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_level (int): Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Create formatters
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    json_formatter = JsonFormatter()
    
    # Console Handler (for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File Handler for all logs (with rotation)
    all_logs_file = LOGS_DIR / "all.log"
    all_logs_handler = RotatingFileHandler(
        all_logs_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    all_logs_handler.setFormatter(json_formatter)
    all_logs_handler.setLevel(logging.DEBUG)
    logger.addHandler(all_logs_handler)
    
    # Error Log Handler (with daily rotation)
    error_log_file = LOGS_DIR / "error.log"
    error_handler = TimedRotatingFileHandler(
        error_log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of error logs
        encoding='utf-8'
    )
    error_handler.setFormatter(json_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # Application-specific log file
    app_log_file = LOGS_DIR / f"{name}.log"
    app_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    app_handler.setFormatter(json_formatter)
    app_handler.setLevel(logging.INFO)
    logger.addHandler(app_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If it doesn't exist, create one with default settings.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)

# Example usage:
if __name__ == "__main__":
    # Create a logger for the main application
    logger = get_logger("main")
    
    # Example log messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
