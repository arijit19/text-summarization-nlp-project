import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_dir="logs", log_file="running_logs.log", level=logging.INFO):
    """
    Set up a logger with the specified name and parameters.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored.
        log_file (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
        logging.Logger: Configured logger.
    """
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
    log_filepath = os.path.join(log_dir, log_file)

    os.makedirs(log_dir, exist_ok=True)

    # Create handlers
    file_handler = RotatingFileHandler(log_filepath, maxBytes=10**6, backupCount=5)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create formatters and add it to handlers
    formatter = logging.Formatter(logging_str)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Usage
logger = setup_logger("textSummarizerLogger")

# Test logging
logger.info("Logger is configured and ready.")
