import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Get the logs path
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the logs folder
os.makedirs(logs_path, exist_ok = True)

# Get the log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Logging cfg
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

