import os
import sys
import logging
from datetime import datetime

LOG_FILE = "/var/log/app/library.log"

def setup_logger():
    """Configure python logger to write to log file and stderr."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger("WormLibrary")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [IP: %(ip)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stderr handler for Apache CGI error log capture
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def get_remote_ip():
    return os.environ.get("REMOTE_ADDR", "127.0.0.1")

def log_info(message, username="system"):
    try:
        logger = setup_logger()
        logger.info(f"User: {username} | {message}", extra={"ip": get_remote_ip()})
    except Exception as e:
        sys.stderr.write(f"Logger failure: {e}\n")

def log_warn(message, username="system"):
    try:
        logger = setup_logger()
        logger.warning(f"User: {username} | {message}", extra={"ip": get_remote_ip()})
    except Exception as e:
        sys.stderr.write(f"Logger failure: {e}\n")

def log_error(message, username="system"):
    try:
        logger = setup_logger()
        logger.error(f"User: {username} | {message}", extra={"ip": get_remote_ip()})
    except Exception as e:
        sys.stderr.write(f"Logger failure: {e}\n")
