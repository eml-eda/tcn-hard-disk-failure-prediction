import logging
import datetime
import os
import colorlog

current_time = ''

def setup_logger():
    global current_time

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log')

    # Create the directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a file handler
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(os.path.join(log_dir, f'log_file_{current_time}.log'))
    handler.setLevel(logging.INFO)
    #handler.setLevel(logging.DEBUG)

    # Create a console handler with color
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    #console_handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

def get_log_file_path():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log')
    log_file_path = os.path.join(log_dir, f'log_file_{current_time}.log')
    return log_file_path

# The logger is set up only once when the logger_config module is first imported
logger = setup_logger()

# Import the logger object into the module's namespace
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical