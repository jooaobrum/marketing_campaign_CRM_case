import sys
from src.logger import logging

# Function to generate an error message with details
def raise_error_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = "Error occured in the script named [{0}], line number [{1}], error message: [{2}]".format(filename, line_number, str(error))

    return error_message

# Custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = raise_error_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message
    
