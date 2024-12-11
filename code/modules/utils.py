# modules/utils.py

import logging
import sys
from datetime import datetime

def setup_logging():
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    log_filename = f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )