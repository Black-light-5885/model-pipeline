# Buying Decision Support Project
# Team@Stratlytics
# @author : Bala.
# File Information: This file is to create model features for the incremental data and update the model input file with the delta data.


# -----------------------------Imports-----------------------------------
import pandas as pd
import warnings
from sklearn.model_selection import cross_validate
import logging
warnings.filterwarnings('ignore')
from sklearn.metrics import  r2_score,mean_absolute_percentage_error,mean_absolute_error
import boto3
import datetime 
from pathlib import Path
import sys
# ----------------------------- Globals --------------------------------------

# The locatioin of the accuracy file from s3 bucket.
INUPUT_FILE = "INPUT_FILE_FOR_ACCURACY_CALCULATION"
RUN_DATE = datetime.date.today().strftime("%d-%b-%y")
RUN_TIME = datetime.datetime.now().strftime("%H:%M:%S")

# S3 Configuration
LOG_TIME = datetime.date.today().strftime("%d-%b-%y_%H-%M")
S3_BUCKET_NAME = 'tcpl-buyingdecision'
S3_LOGS_PATH = "removed"

# ---------------------------------------- Logger setup -----------------------------------------------------

def upload_to_s3(local_file_path, s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    s3_client.upload_file(str(local_file_path), s3_bucket, s3_key)

class S3UploadHandler(logging.Handler):
    def __init__(self, local_file_path, s3_bucket, s3_key):
        super().__init__()
        self.local_file_path = local_file_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key

    def emit(self, record):
        # Ensure that log message is written to the local file
        self.flush()
        upload_to_s3(self.local_file_path, self.s3_bucket, self.s3_key)

def get_module_logger(module_name, log_level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    log_dir = Path(f'logs/{RUN_DATE}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the log file path
    log_file = log_dir / f"{module_name}_{RUN_TIME}.log"
    
    # Create file and stream handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create formatters and add them to handlers
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Add S3 upload handler
    s3_handler = S3UploadHandler(local_file_path=log_file, s3_bucket=S3_BUCKET_NAME, s3_key=f"{S3_LOGS_PATH}{module_name}_{RUN_TIME}.log")
    logger.addHandler(s3_handler)

    return logger

logger = get_module_logger("Model_Retaining")


# --------------------------------------- Helper Function ---------------------------------

def create_derived_cols(df)-> None:
    """_summary_

    Args:
        df (_type_): _description_
    """
    
    ...    
    
def evaluation_metric(df):
    ...
    
def assign_rank(df):
    ...

def weighted_avg_price(df,cluster,print_=True,threshold  =300):
    ...
    
def business_metric(prediction_df,date=None, print_=True):
    ...

def Summary(df, level = 'Component'):
    ...