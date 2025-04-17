import logging
import os

def setup_logger(name):
    """Set up a logger with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    log_file = os.path.join(os.path.dirname(__file__), 'app.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for outputting logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d - %(funcName)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
