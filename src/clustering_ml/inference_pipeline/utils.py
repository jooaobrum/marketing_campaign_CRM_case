import sys
from logger import logging
import boto3
import botocore
import os
import pickle
import pandas as pd
import io
import json

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


def upload_file_to_s3(client, bucket_name, local_file_path, s3_file_path):
    # Upload the file to the specified bucket and folder
    try:
        client.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f'Successfully uploaded {local_file_path} to {bucket_name}/{s3_file_path}')
        return 200

    except Exception as e:
        print(f'Error uploading {local_file_path} to {bucket_name}/{s3_file_path}: {str(e)}')
        return None

def read_file_from_s3(client, bucket_name, s3_file_path):
    try:
        response = client.get_object(Bucket=bucket_name, Key=s3_file_path)
        body = response['Body'].read()

        # Read the file contents based on the content type
        if s3_file_path.split('.')[-1] == 'pkl':
            # Binary data (e.g., pickle)
            file_contents = pickle.loads(body)

        elif s3_file_path.split('.')[-1] == 'csv':
            # CSV file
            file_contents = pd.read_csv(io.BytesIO(body) )
        
        elif s3_file_path.split('.')[-1] == 'json':
            # JSON file
            file_contents = json.loads(body.decode('utf-8'))

        else:
            # Handle unsupported content types or raise an exception
            raise ValueError('Unsupported content type.')

        print(f'Successfully read file contents from {bucket_name}/{s3_file_path}:')

        return file_contents
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f'The object {bucket_name}/{s3_file_path} does not exist.')
        else:
            print(f'Error reading {bucket_name}/{s3_file_path}: {str(e)}')
        return None
    
def download_file_from_s3(client, bucket_name, s3_file_path, local_folder_path):
    try:
        
        # Get the file name (object key)
        file_name = s3_file_path.split('/')[-1]
       

        # Define the local file path
        local_file_path = os.path.join(local_folder_path, file_name)

        # Download the file from S3 to the local folder
        with open(local_file_path, 'wb') as f:
            client.download_fileobj(bucket_name, s3_file_path, f)

        print(f'Successfully downloaded {bucket_name}/{file_name} to {local_folder_path}/{file_name}')
        
        return 200
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f'The object {bucket_name}/{s3_file_path} does not exist.')
        else:
            print(f'Error downloading {bucket_name}/{s3_file_path}: {str(e)}')
        return None
