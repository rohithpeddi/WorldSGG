import logging
import os
import sys

def setup_logging(output_dir=None, log_filename="run.log"):
    """
    Setup logging configuration.
    
    Args:
        output_dir (str, optional): Directory to save the log file. If None, only console logging is enabled.
        log_filename (str, optional): Name of the log file. Defaults to "run.log".
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, log_filename)
        handlers.append(logging.FileHandler(log_file))
        print(f"Logging to {log_file}") # Feedback to user so they know where it is

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True # Override any existing configuration
    )
