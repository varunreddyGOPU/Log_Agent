import logging
import os

def configure_logging():
    """Sets up a logger to write errors to a file."""
    # Define the log file path
    log_file_path = 'app_error.log'

    # Create a logger instance
    logger = logging.getLogger('buggy_app_logger')
    logger.setLevel(logging.ERROR) # Set the minimum level of messages to log

    # Create a file handler to write logs to a file
    # 'w' mode ensures the log file is cleared on each run for this example
    file_handler = logging.FileHandler(log_file_path, mode='w')

    # Create a formatter to define the log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def perform_division(numerator, denominator):
    """
    A function with a potential bug. It attempts to divide two numbers.
    The bug occurs if the denominator is zero.
    """
    logger = logging.getLogger('buggy_app_logger')
    try:
        result = numerator / denominator
        print(f"The result of the division is: {result}")
        return result
    except ZeroDivisionError as e:
        # Log the error with a detailed message and exception info
        error_message = f"Critical error in perform_division: Attempted to divide by zero. Numerator: {numerator}, Denominator: {denominator}."
        print(f"An error occurred. Check the log file 'app_error.log' for details.")
        logger.error(error_message, exc_info=True)
        return None

def main():
    """
    Main function to run the buggy code and generate a log.
    """
    # Configure the logger
    configure_logging()

    # Data that will cause the bug
    dividend = 10
    divisor = 0

    print("Running the application...")
    print(f"Attempting to divide {dividend} by {divisor}.")

    # Call the function that contains the bug
    perform_division(dividend, divisor)

    print("Application run finished.")

if __name__ == "__main__":
    main()