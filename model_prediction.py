# Buying Decision Support Project
# Team@Stratlytics
# @author : Bala.
# File Information: This file is to create model features for the incremental data and update the model input file with the delta data.


# -----------------------------Imports-----------------------------------
import pandas as pd
import os
import sys
import logging
from pathlib import Path
import numpy as np 
from multiprocessing import Pool
import pickle
import boto3
import re
import subprocess
import datetime
# ----------------------------- Code -----------------------------------
# Install required packages with compatible versions
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "awswrangler"])
    import awswrangler as wr
    print("awswrangler installed and imported successfully as wr.")
except Exception as e:
    print(f"Error installing or importing awswrangler: {e}")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl installed successfully")
except Exception as e:
    print(f"Error installing or importing openpyxl: {e}")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.0"])
    print("scikit learn installed successfully")
except Exception as e:
    print(f"Error installing or importing scikit-learn: {e}")

# ----------------------------- GLOBALS -----------------------------------

S3_ROOT_PATH3 = "bgdn-pre-prod"
# Following dictionary contains the location for fetching the data
GLOBAL_PATHS= "removed"
BACKEND_ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

RUN_DATE = datetime.date.today().strftime("%d-%b-%y")
# RUN_DATE = '2024-05-24 00:00:00'

BC_FEATURES =[ "A_U_B_C_PD_MA2","T_P_A_C_PD_MA20"]

GC_FEATURES = ['A_A_G_C_AP_MA2',
 'T_U_G_C_PD_MA5',
 'O_U_G_C_IW_MA5',
 'T_A_G_C_AP_MA1',
 'A_A_G_C_PQ_MA5',
 'O_U_G_C_PD_MA2',
 'T_U_G_C_PD_MA10',
 'T_P_G_C_PD_MA5',
 'A_A_G_C_PD_MA15',
 'A_A_G_C_PQ_MA15',
 'T_P_G_C_PQ_MA1',
 'T_U_G_C_PD_MA1',
 'A_A_G_C_PD_MA5',
 'A_A_G_C_PD_MA10',
 'T_U_G_C_PQ_MA1',
 'A_A_G_C_PD_MA4',
 'A_A_G_C_PD_MA1',
 'T_P_G_C_PD_MA3',
]

COMPONENT_QUADRANTS = {
    "smooth":["3EO33",
                    "2EO32",
                    "3BO33",
                    "2BO32",
                    "2FO32",
                    "1BO22",
                    "3FO33",
                    "3DO34",
                    "2DO32",
                    "2PO32",
                    "3MO33",
                    "3PO33",
                    "2MO32",
                    "1DO22",
                    "2DO32R",
                    "3DO33",
                    "2AO32",
                    "2AO32R",
                    "2AR32R",
                    "2DR32R",
                    "2CSDO32",
                    "3AO34",
                    "NIVLRGE",
                    "2DO32A",
                    "2AO32A",
                    "2PO32R"],
    
    "intermittent": ["3EO34",
                    "NITNA",
                    "3BO34",
                    "3PO34",
                    "4EO44",
                    "3CSEO33",
                    "3FO34",
                    "3EO34M",
                    "4PO44",
                    "3MO34",
                    "4DO44",
                    "4BO44",
                    "4FO44",
                    "3CSEO34",
                    "3CSDO34",
                    "4PO45",
                    "1BO33",
                    "3BO34M",
                    "NIPRLF",
                    "4MO44",
                    "3CSDO33",
                    "4CSEO44",
                    "3CSFO33",
                    "3FO34M",
                    "NIVSMKY",
                    "3CSMO33",
                    "3CSMO34",
                    "4CSDO44",
                    "3LO34",
                    "3CSFO34",
                    "3CSDO33O",
                    "NIDULL",
                    "4CSFO44",
                    "NIBINF",
                    "4AO44",
                    "4AO45",
                    "3LO33",
                    "NISTALE",
                    "3CSAO34",
                    "2AR32A",
                    "2DR32A",
                    "2DO32H",
                    "4CSAO44",
                    "3SDO34",
                    "2PO32A",
                    "2AO32H",
                    "3AO33",
                    "2LO32",
                    "3CSAO33",
                    "2PO32H"],
    
        "lumpy" : ['4EO45',
        '4DO45',
        '4BO45',
        '4FO45',
        '3NOBOL4',
        '3EO34K',
        '4CSDO45',
        '4EO44K',
        '3EO33B',
        '3BO34K',
        '3FO34K',
        '4BO44K',
        '4FO44K',
        '3MO34K',
        '4MO44K',
        '4NOBOL4',
        '3PO34K',
        '3DO34K',
        '4PO44K',
        '5SWVY21O',
        '4EO44B',
        '4DO44K',
        '5SWVY21OGN',
        '4CSAO45',
        '4MGD19184',
        '2DR32H',
        '4PO44T',
        '3FO33B',
        '4DO44T',
        '2AR32H',
        '4MGL19174',
        '4MGD20184',
        '5LY42AO',
        '5SWY42AO',
        '4MGD22195',
        '5EO45',
        '3LO34K',
        '4AO44K',
        '4CSDY33O',
        '3AO34K',
        '4MGL19184',
        '5PO45',
        '4FO44B',
        '3MY43O',
        '3MGL17144',
        '5BO45',
        '5DO45',
        '5FVY21O',
        '5LVY21O',
        '2FO32H',
        '3SWY43O',
        '5FO45',
        '4MWVY32O',
        '5AO45',
        '5SWVY21ONP',
        '6SWVY21O',
        '4MWVY32OGN',
        '3SWVY21OGN',
        '6SWVY22O',
        '4SWVY32ORA',
        '4SPL819GRP',
        '4SPL2119BS',
        '4SPL2119MI',
        '4SPL2119LG',
        '4SPL2119NG',
        '4SPL2119TC',
        '4SPL2119AC',
        '4SPL2119GC',
        '4SPL819GRAC',
        '4SPL819GRAP',
        '4SPL819GRG',
        '4SPL819GRMC']
}

MODEL_INPUT_FEATURES = { "Auction": ['INVOICE_WT', 'A_U_B_C_PD_MA2', 'A_A_G_E_AP_MA3', 'A_A_G_C_AP_MA2',
       'T_U_G_C_PD_MA5', 'T_A_G_C_AP_MA1', 'O_U_G_C_PD_MA2', 'A_A_G_C_PD_MA5',
       'A_A_G_C_PD_MA10', 'T_U_G_E_AP_MA1', 'T_A_G_E_AP_MA3', 'O_U_G_E_AP_MA3',
       'A_A_C_R_AP_MA3', 'T_A_C_R_AP_MA1', 'T_A_C_R_AP_MA3',
       'QUALITY_VARIANCE', 'Code'],
                        "Private": ['INVOICE_WT', 'T_P_A_C_PD_MA20', 'T_U_G_C_PD_MA5', 'T_A_G_C_AP_MA1',
       'T_P_G_C_PD_MA5', 'T_U_G_E_AP_MA1', 'T_A_G_E_AP_MA3', 'T_P_C_R_AP_MA3',
       'T_P_C_R_AP_MA5', 'T_A_C_R_AP_MA3', 'QUALITY_VARIANCE', 'Code'
                                   ]
}


smooth=["3EO33",
                "2EO32",
                "3BO33",
                "2BO32",
                "2FO32",
                "1BO22",
                "3FO33",
                "3DO34",
                "2DO32",
                "2PO32",
                "3MO33",
                "3PO33",
                "2MO32",
                "1DO22",
                "2DO32R",
                "3DO33",
                "2AO32",
                "2AO32R",
                "2AR32R",
                "2DR32R",
                "2CSDO32",
                "3AO34",
                "NIVLRGE",
                "2DO32A",
                "2AO32A",
                "2PO32R"]
intermittent= ["3EO34",
                "NITNA",
                "3BO34",
                "3PO34",
                "4EO44",
                "3CSEO33",
                "3FO34",
                "3EO34M",
                "4PO44",
                "3MO34",
                "4DO44",
                "4BO44",
                "4FO44",
                "3CSEO34",
                "3CSDO34",
                "4PO45",
                "1BO33",
                "3BO34M",
                "NIPRLF",
                "4MO44",
                "3CSDO33",
                "4CSEO44",
                "3CSFO33",
                "3FO34M",
                "NIVSMKY",
                "3CSMO33",
                "3CSMO34",
                "4CSDO44",
                "3LO34",
                "3CSFO34",
                "3CSDO33O",
                "NIDULL",
                "4CSFO44",
                "NIBINF",
                "4AO44",
                "4AO45",
                "3LO33",
                "NISTALE",
                "3CSAO34",
                "2AR32A",
                "2DR32A",
                "2DO32H",
                "4CSAO44",
                "3SDO34",
                "2PO32A",
                "2AO32H",
                "3AO33",
                "2LO32",
                "3CSAO33",
                "2PO32H"]
Lumpy_bin_1=["2AR32H",
"2DR32H",
"2FO32H",
"3FO33B",
"3MGL17144",
"3NOBOL4",
"3SWVY21OGN",
"4CSDY33O",
"4NOBOL4"]
Lumpy_bin_2=["3AO34K",
"3BO34K",
"3DO34K",
"3EO33B",
"3EO34K",
"3FO34K",
"3LO34K",
"3MO34K",
"3PO34K",
"4CSAO45",
"4CSDO45",
"4MGD20184",
"4MGL19174"]
Lumpy_bin_3=["3MY43O",
"4AO44K",
"4BO44K",
"4DO44K",
"4DO44T",
"4EO44K",
"4FO44K",
"4MGD19184",
"4MGL19184",
"4MWVY32O",
"4MWVY32OGN",
"4PO44K",
"4PO44T",
"4SWVY32ORA",
"5SWVY21ONP"]
Lumpy_bin_4=["3SWY43O",
"4BO45",
"4DO45",
"4EO44B",
"4EO45",
"4FO44B",
"4FO45",
"4MGD22195",
"4MO44K",
"5FVY21O",
"5LVY21O",
"5LY42AO"]
Lumpy_bin_5=["4SPL2119AC",
"4SPL2119BS",
"4SPL2119GC",
"4SPL2119LG",
"4SPL2119MI",
"4SPL2119NG",
"4SPL2119TC",
"4SPL819GRAC",
"4SPL819GRAP",
"4SPL819GRG",
"4SPL819GRMC",
"4SPL819GRP",
"5AO45",
"5BO45",
"5DO45",
"5EO45",
"5FO45",
"5PO45",
"5SWVY21O",
"5SWVY21OGN",
"5SWY42AO",
"6SWVY21O",
"6SWVY22O"]


# ----------------------------- Functions -----------------------------------
"""
Methods available: debug, info, warning, error, critical
ex: logging.debug('This is a debug message')
"""
def get_module_logger(module_name, log_level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    log_dir = Path(f'logs/{RUN_DATE}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    log_file = log_dir / f"{module_name}.log"
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create formatters and add them to handlers
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = get_module_logger("Model_Prediction")

class NoInputFile(Exception):
    def __init__(self,path):
        logger.error(f"No input file in the provided path :{path}")
        
class BadInputFile(Exception):  # 400
    def __init__(
        self,
        status_msg="Invalid Data Received",
        category="error",
    ):
        logger.info(f"BadInputFile {category} - {status_msg}")
       

def get_file_from_s3(path:str):
    bucket = "tcpl-buyingdecision"
    # Initialize the S3 client
    s3 = boto3.client('s3')
    # List objects within the folder
    response = s3.list_objects_v2(Bucket=bucket, Prefix=path)
    files = response.get('Contents', [])
    if len(files)>0:
        output = pd.DataFrame()
        for file in files:
            path_key = f"s3://{bucket}/{file['Key']}"
            temp = pd.read_parquet(path_key)
            output = pd.concat([output,temp])
        return output
    else:
        raise NoInputFile(path)

def get_garden_std_data():
    """
    Methode to get garden standardization mapping data from the S3 bucket.

    Returns:
        garden_std_data : (pd.DataFrame)
                        - Pandas dataframe with 4 columns that contains Garden Name with their Standardized Name mapping.
                            -"GARDEN_ID",
                            -"GARDEN_CODE",
                            -"GARDEN_NAME",
                            -"Standardized Garden".
    """
    try:
        logger.info(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : started...")
        
        # garden_std_data = wr.s3.read_excel(GLOBAL_PATHS["STD_GARDEN_FILE"])
        garden_std_data = pd.read_excel(GLOBAL_PATHS["STD_GARDEN_FILE"])

        
        # Garden standardization file may contain duplicates due to the multiple GARDEN ID mappings.
        # To map garden with their unique Standard Name selecting two columns from the data and removing duplicates if any.
        garden_std_data = garden_std_data[['GARDEN_ID',"GARDEN_CODE","Standardized Garden"]].drop_duplicates().reset_index(drop = True) 
        
        logger.info(f"Number of Garden mapping data fetched from S3: %d", garden_std_data.shape[0])
        
    except Exception as e:
        logger.error(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : Failed to read Garden Standard Name mapping data from S3: %s", e)
        
    else:
        logger.info(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : : finished...")
        
        return garden_std_data 
    
def get_model_train_data():
    """
    Methode to get model train data from the S3.

    Returns:
        model_input_df : (pd.DataFrame) Pandas dataframe that contains train dataset with lot information.
    """
    try:
        logger.info(f"Pipeline_Service__data_fetch.py__get_model_train_data() : started...")
        
        # model_input_df = wr.s3.read_parquet(GLOBAL_PATHS["MODEL_INPUT_DATA"])
        model_input_df = pd.read_parquet(GLOBAL_PATHS["MODEL_INPUT_DATA"])

       
        model_input_df = model_input_df[model_input_df['DATE']<RUN_DATE].reset_index(drop = True)
        
        logger.info(f"Number of model Input data fetched from S3: %d", model_input_df.shape[0])
        
    except Exception as e:
        logger.error(f"Pipeline_Service__data_fetch.py__get_model_train_data() : Failed to read Model Input data from S3: %s", e)
        
    else:
        logger.info(f"Pipeline_Service__data_fetch.py__get_model_train_data() : : finished...")
        
        return model_input_df 
      
def get_input_data(channel = "Auction")->pd.DataFrame:
    """
    Methode to get the Price Prediction Input file from the S3 bucket.
    
    
    Parameters:
        sheet_name (str) : Sheet name in which the input file to be read.
        
    Returns:
        input_df : (pd.DataFrame)
                        - Pandas dataframe that contains the input data for the price prediction.
    """
    try:
        logger.info(f"Pipeline_Service__data_fetch.py__get_input_data() : started...")
        # input_data = wr.s3.read_excel(GLOBAL_PATHS["PREDICTION_FILE"], sheet_name= sheet_name)
        file_path = GLOBAL_PATHS["PREDICTION_FILE"][channel]
        input_data = get_file_from_s3(file_path)
        # input_data = input_data.loc[:1000,:]
        
        if input_data.shape[0]<1:
            logger.error("Pipeline_Service__data_fetch.py__get_input_data() : The input file does not contain any records")
            raise BadInputFile(
                status_msg="The input file does not contain any records"
                )
            
        logger.info(f"Number of Prediction Input data fetched from S3: %d", input_data.shape[0])
        
    except Exception as e:
        logger.error(f"Pipeline_Service__data_fetch.py__get_input_data() : Failed to read input data from S3: %s", e)
        
    else:
        logger.info(f"Pipeline_Service__data_fetch.py__get_input_data() : : finished...")
        
        return input_data 

def get_master_data()->pd.DataFrame:
    """
    Methode to get the historical master data from the S3 bucket.
    

    Returns:
        hist_data (pd.DataFrame) : Pandas dataframe that contains historical data for sold information.
    """
    try:
        logger.info(f"Pipeline_Service__data_fetch.py__get_historical_data() : started...")
        # hist_data = wr.s3.read_parquet(GLOBAL_PATHS["MASTER_FILE_LOCATION"])
        hist_data = pd.read_parquet(GLOBAL_PATHS["MASTER_FILE_LOCATION"])

        print(hist_data.shape[0])
        hist_data = hist_data[hist_data['DATE']<RUN_DATE].reset_index(drop = True)
        print(hist_data.shape[0])
        
        logger.info(f"Number of historical data fetched from S3: %d", hist_data.shape[0])
        
    except Exception as e:
        logger.error(f"Pipeline_Service__data_fetch.py__get_historical_data(): Failed to read historical data from S3: %s", e)
        
    else:
        logger.info(f"Pipeline_Service__data_fetch.py__get_historical_data() : finished...")
        
        return hist_data


class Garden():
    def __init__(self, company="A",
                 channel="A",
                 loc='s3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-component-templates',date = None):
        """
        Initialize the Garden class with company, channel, and file location.

        Parameters:
        company (str): Company identifier. Default is "A".
        channel (str): Channel identifier. Default is "A".
        loc (str): File location. Default is S3 path.
        """
        self.__comp = company
        self.__chnl = channel
        self.__loc = loc
        
        # Load template data
        self.template_price = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_AP_Template_df.csv')
        self.template_pq = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_PQ_Template_df.csv')
        self.template_Iw = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_IW_Template_df.csv')
        self.template_diff = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_PD_Template_df.csv')
        
        if date != None:
                    ap_list = self.template_price.columns.tolist()
                    pq_list = self.template_pq.columns.tolist()
                    iw_list = self.template_Iw.columns.tolist()
                    pd_list = self.template_diff.columns.tolist()
                    
                    ap_list = ['GC']+[d for d in ap_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    pq_list = ['GC']+[d for d in pq_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    iw_list = ['GC']+[d for d in iw_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    pd_list = ['GC']+[d for d in pd_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    
                    self.template_price =self.template_price[ap_list]
                    self.template_pq =self.template_pq[pq_list]
                    self.template_Iw =self.template_Iw[iw_list]
                    self.template_diff =self.template_diff[pd_list]
    
    def __updatePrice(self, df):
        """
        Update the price template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new price data.
        """
        df.rename(columns={"PURCHASED_PRICE_Detail": self.__snapDate}, inplace=True)
        unique_gc_values = self.template_price["GC"].unique().tolist()
        locl_df = df[df.GC.isin(unique_gc_values)]
        locl_df2 = df[~df.GC.isin(unique_gc_values)]
        self.template_price = self.template_price.merge(locl_df, on='GC', how='left')
        self.template_price = pd.concat([self.template_price, locl_df2])
    
    def __updatePq(self, df):
        """
        Update the purchased quantity template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new purchased quantity data.
        """
        df.rename(columns={"PURCHASED_QUANTITY_Detail": self.__snapDate}, inplace=True)
        unique_gc_values = self.template_pq["GC"].unique().tolist()
        locl_df = df[df.GC.isin(unique_gc_values)]
        locl_df2 = df[~df.GC.isin(unique_gc_values)]
        self.template_pq = self.template_pq.merge(locl_df, on='GC', how='left')
        self.template_pq = pd.concat([self.template_pq, locl_df2])
    
    def __updateIw(self, df):
        """
        Update the invoice weight template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new invoice weight data.
        """
        df.rename(columns={"INVOICE_WT": self.__snapDate}, inplace=True)
        unique_gc_values = self.template_Iw["GC"].unique().tolist()
        locl_df = df[df.GC.isin(unique_gc_values)]
        locl_df2 = df[~df.GC.isin(unique_gc_values)]
        self.template_Iw = self.template_Iw.merge(locl_df, on='GC', how='left')
        self.template_Iw = pd.concat([self.template_Iw, locl_df2])
    
    def __updateDiff(self, df):
        """
        Update the price difference template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new price difference data.
        """
        df.rename(columns={"diff": self.__snapDate}, inplace=True)
        unique_gc_values = self.template_diff["GC"].unique().tolist()
        locl_df = df[df.GC.isin(unique_gc_values)]
        locl_df2 = df[~df.GC.isin(unique_gc_values)]
        self.template_diff = self.template_diff.merge(locl_df, on='GC', how='left')
        self.template_diff = pd.concat([self.template_diff, locl_df2])
    
    def __convertDict(self, df, combinations):
        """
        Convert DataFrame to a dictionary for mapping GC values.

        Parameters:
        df (pd.DataFrame): DataFrame to convert.
        combinations (list): List of combinations to map.
        """
        self.__nan_dict = {}
        for value in combinations:
            try:
                self.__nan_dict[value] = np.array(df.loc[df["GC"] == value].iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()[0])
            except:
                self.__nan_dict[value] = []
    
    def __gc_priceMap(self, x):
        """
        Map GC value to the last known price.

        Parameters:
        x: GC value.

        Returns:
        float: Last known price or 0 if not found.
        """
        try:
            return self.__nan_dict[x][-1]
        except:
            return 0
    
    def addSnap(self, df):
        """
        Add a snapshot of the current data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the current snapshot data.
        """
        self.__snapDate = str(df.loc[0, 'DATE'])
        self.__updatePrice(df[['GC', 'PURCHASED_PRICE_Detail']])
        self.__updatePq(df[['GC', 'PURCHASED_QUANTITY_Detail']])
        self.__updateIw(df[['GC', 'INVOICE_WT']])
        df['Prv_Price'] = df['GC'].apply(lambda x: self.__gc_priceMap(x))
        df['diff'] = round(df['PURCHASED_PRICE_Detail'] - df['Prv_Price'])
        self.__updateDiff(df[['GC', 'diff']])
    
    def __singleSnap(self, out, column):
        """
        Generate single snapshot features.

        Parameters:
        out (pd.DataFrame): Output DataFrame to store features.
        column (str): Column name prefix for the features.

        Returns:
        pd.DataFrame: DataFrame with added snapshot features.
        """
        out[column+'1'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-1] if len(self.__nan_dict[x]) > 0 else np.NaN)
        
        out[column+'2'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-2:].mean() if len(self.__nan_dict[x]) > 1 else np.NaN)
        
        out[column+'3'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-3:].mean() if len(self.__nan_dict[x]) > 2 else np.NaN)
        
        out[column+'4'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-4:].mean() if len(self.__nan_dict[x]) > 3 else np.NaN)
        
        out[column+'5'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-5:].mean() if len(self.__nan_dict[x]) > 4 else np.NaN)
        
        out[column+'10'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-10:].mean() if len(self.__nan_dict[x]) > 9 else np.NaN)
        
        out[column+'15'] = out['GC'].apply(
                                            lambda x: self.__nan_dict[x][-15:].mean() if len(self.__nan_dict[x]) > 14 else np.NaN)
        
        return out
    
    def getSnap(self, comb):
        """
        Get snapshot features for a combination of values.

        Parameters:
        comb (list): List of combinations.

        Returns:
        pd.DataFrame: DataFrame containing snapshot features.
        """
        column = f"{self.__comp}_{self.__chnl}_G_C_AP_MA"
        out = pd.DataFrame(columns=['GC'])
        out['GC'] = comb
        
        self.__convertDict(self.template_price, comb)
        out = self.__singleSnap(out, column)
        
        self.__convertDict(self.template_pq, comb)
        column2 = f"{self.__comp}_{self.__chnl}_G_C_PQ_MA"
        out = self.__singleSnap(out, column2)
        
        self.__convertDict(self.template_Iw, comb)
        column3 = f"{self.__comp}_{self.__chnl}_G_C_IW_MA"
        out = self.__singleSnap(out, column3)
        
        self.__convertDict(self.template_diff, comb)
        column4 = f"{self.__comp}_{self.__chnl}_G_C_PD_MA"
        out = self.__singleSnap(out, column4)
        
        return out
    
    def savedf(self):
        """
        Save the template DataFrames to CSV files.
        """
        self.template_price.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv', index=False)
        self.template_pq.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_PQ_Template_df_UAT_5.csv', index=False)
        self.template_Iw.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_IW_Template_df_UAT_5.csv', index=False)
        self.template_diff.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_PD_Template_df_UAT_5.csv', index=False)

class GardenGrade():
    def __init__(self, company="A",
                 channel="A",
                 date = None, 
                 loc='s3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-grade-templates'):
        """
        Initialize the GardenGrade class with company, channel, and file location.

        Parameters:
        company (str): Company identifier. Default is "A".
        channel (str): Channel identifier. Default is "A".
        loc (str): File location. Default is S3 path.
        """
        self.__comp = company
        self.__chnl = channel
        self.__loc = loc
        
        # Load template data
        self.template_price = pd.read_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df.csv')
        if date != None:
                    ap_list = self.template_price.columns.tolist()
                    
                    ap_list = ['GE']+[d for d in ap_list[1:] if 'y' not in d and pd.to_datetime(re.sub('_x','',d))<pd.to_datetime(date)]
                    
                    self.template_price =self.template_price[ap_list]
    
    def __updatePrice(self, df):
        """
        Update the price template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new price data.
        """
        df.rename(columns={"PURCHASED_PRICE_Detail": self.__snapDate}, inplace=True)
        unique_gc_values = self.template_price["GE"].unique().tolist()
        locl_df = df[df.GE.isin(unique_gc_values)]
        locl_df2 = df[~df.GE.isin(unique_gc_values)]
        self.template_price = self.template_price.merge(locl_df, on='GE', how='left')
        self.template_price = pd.concat([self.template_price, locl_df2])
    
    def __convertDict(self, df, combinations):
        """
        Convert DataFrame to a dictionary for mapping GE values.

        Parameters:
        df (pd.DataFrame): DataFrame to convert.
        combinations (list): List of combinations to map.
        """
        self.__nan_dict = {}
        for value in combinations:
            try:
                self.__nan_dict[value] = np.array(df.loc[df["GE"] == value].iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()[0])
            except:
                self.__nan_dict[value] = []
    
    def addSnap(self, df):
        """
        Add a snapshot of the current data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the current snapshot data.
        """
        self.__snapDate = str(df.loc[0, 'DATE'])
        self.__updatePrice(df[['GE', 'PURCHASED_PRICE_Detail']])
    
    def __singleSnap(self, out, column):
        """
        Generate single snapshot features.

        Parameters:
        out (pd.DataFrame): Output DataFrame to store features.
        column (str): Column name prefix for the features.

        Returns:
        pd.DataFrame: DataFrame with added snapshot features.
        """
        out[column+'1'] = out['GE'].apply(lambda x: self.__nan_dict[x][-1] if len(self.__nan_dict[x]) > 0 else np.NaN)
        out[column+'2'] = out['GE'].apply(lambda x: self.__nan_dict[x][-2:].mean() if len(self.__nan_dict[x]) > 1 else np.NaN)
        out[column+'3'] = out['GE'].apply(lambda x: self.__nan_dict[x][-3:].mean() if len(self.__nan_dict[x]) > 2 else np.NaN)
        out[column+'4'] = out['GE'].apply(lambda x: self.__nan_dict[x][-4:].mean() if len(self.__nan_dict[x]) > 3 else np.NaN)
        out[column+'5'] = out['GE'].apply(lambda x: self.__nan_dict[x][-5:].mean() if len(self.__nan_dict[x]) > 4 else np.NaN)
        
        return out
    
    def getSnap(self, comb):
        """
        Get snapshot features for a combination of values.

        Parameters:
        comb (list): List of combinations.

        Returns:
        pd.DataFrame: DataFrame containing snapshot features.
        """
        column = f"{self.__comp}_{self.__chnl}_G_E_AP_MA"
        out = pd.DataFrame(columns=['GE'])
        out['GE'] = comb
        
        self.__convertDict(self.template_price, comb)
        out = self.__singleSnap(out, column)
        
        return out
    
    def savedf(self):
        """
        Save the template DataFrames to CSV files.
        """
        self.template_price.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv', index=False)

class CmpRating():
    def __init__(self, company="A", 
                 channel="A",
                 date=None, 
                 loc='s3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/component-rating-templates'):
        """
        Initialize the CmpRating class with company, channel, and file location.

        Parameters:
        company (str): Company identifier. Default is "A".
        channel (str): Channel identifier. Default is "A".
        loc (str): File location. Default is S3 path.
        """
        self.__comp = company
        self.__chnl = channel
        self.__loc = loc
        
        # Load template data
        self.template_price = pd.read_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df.csv')
        if date != None:
            ap_list = self.template_price.columns.tolist()
            
            ap_list = ['CR']+[d for d in ap_list[1:] if 'y' not in d and pd.to_datetime(re.sub('_x','',d))<pd.to_datetime(date)]
            
            self.template_price =self.template_price[ap_list]
    
    def __updatePrice(self, df):
        """
        Update the price template with new data.

        Parameters:
        df (pd.DataFrame): DataFrame containing new price data.
        """
        df.rename(columns={"PURCHASED_PRICE_Detail": self.__snapDate}, inplace=True)
        unique_cr_values = self.template_price["CR"].unique().tolist()
        locl_df = df[df.CR.isin(unique_cr_values)]
        locl_df2 = df[~df.CR.isin(unique_cr_values)]
        self.template_price = self.template_price.merge(locl_df, on='CR', how='left')
        self.template_price = pd.concat([self.template_price, locl_df2])
    
    def __convertDict(self, df, combinations):
        """
        Convert DataFrame to a dictionary for mapping CR values.

        Parameters:
        df (pd.DataFrame): DataFrame to convert.
        combinations (list): List of combinations to map.
        """
        self.__nan_dict = {}
        for value in combinations:
            try:
                self.__nan_dict[value] = np.array(
                                                    df.loc[df["CR"] == value].iloc[:, 1:].apply(
                                                        lambda row: row.dropna().tolist(), axis=1
                                                    ).tolist()[0]
                                                )
            except:
                self.__nan_dict[value] = []
    
    def addSnap(self, df):
        """
        Add a snapshot of the current data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the current snapshot data.
        """
        self.__snapDate = str(df.loc[0, 'DATE'])
        self.__updatePrice(df[['CR', 'PURCHASED_PRICE_Detail']])
    
    def __singleSnap(self, out, column):
        """
        Generate single snapshot features.

        Parameters:
        out (pd.DataFrame): Output DataFrame to store features.
        column (str): Column name prefix for the features.

        Returns:
        pd.DataFrame: DataFrame with added snapshot features.
        """
        out[column+'1'] = out['CR'].apply(
                                        lambda x: self.__nan_dict[x][-1] if len(self.__nan_dict[x]) > 0 else np.NaN)
        out[column+'2'] = out['CR'].apply(
                                        lambda x: self.__nan_dict[x][-2:].mean() if len(self.__nan_dict[x]) > 1 else np.NaN)
        out[column+'3'] = out['CR'].apply(
                                        lambda x: self.__nan_dict[x][-3:].mean() if len(self.__nan_dict[x]) > 2 else np.NaN)
        out[column+'4'] = out['CR'].apply(
                                        lambda x: self.__nan_dict[x][-4:].mean() if len(self.__nan_dict[x]) > 3 else np.NaN)
        out[column+'5'] = out['CR'].apply(
                                        lambda x: self.__nan_dict[x][-5:].mean() if len(self.__nan_dict[x]) > 4 else np.NaN)
        
        return out
    
    def getSnap(self, comb):
        """
        Get snapshot features for a combination of values.

        Parameters:
        comb (list): List of combinations.

        Returns:
        pd.DataFrame: DataFrame containing snapshot features.
        """
        column = f"{self.__comp}_{self.__chnl}_C_R_AP_MA"
        out = pd.DataFrame(columns=['CR'])
        out['CR'] = comb
        
        self.__convertDict(self.template_price, comb)
        out = self.__singleSnap(out, column)
        
        return out
    
    def savedf(self):
        """
        Save the template DataFrames to CSV files.
        """
        self.template_price.to_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv', index=False)


def feature_gen1(df, week=0, year=0, company='A', 
                 channel='A', buying_center='A', component='A',
                 param="AP", ma='MA', value=1):
    """
    Generate features based on the given filters and parameters.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing purchase data.
    week (int): Week number for filtering. Default is 0.
    year (int): Year for filtering. Default is 0.
    company (str): Company identifier for filtering. Default is 'All'.
    channel (str): Channel identifier for filtering. Default is 'All'.
    buying_center (str): Buying center code for filtering. Default is 'All'.
    component (str): Component code for filtering. Default is 'All'.
    param (str): Parameter to calculate the feature for. Default is "AP".
    ma (str): Moving average indicator ('MA' for moving average, otherwise it calculates based on previous years). Default is 'MA'.
    value (int): Window size for moving average or number of years for static window. Default is 1.

    Returns:
    pd.DataFrame: Output DataFrame containing the generated features.
    """
    # Create a copy of the input DataFrame
    filter_df = df.copy()
    
    # Filter by company if specified
    if company != 'A':
        filter_df = filter_df[filter_df.Purchase_Flag == company].reset_index(drop=True)
        
    # Filter by channel if specified
    if channel != 'A':
        filter_df = filter_df[filter_df.Channel == channel].reset_index(drop=True)
    
    # Filter by buying center or component and aggregate data accordingly
    if buying_center != 'A' and component == 'A':
        filter_df = (filter_df[filter_df.BUYING_CENTER_CODE == buying_center]
                     .groupby(['DATE', 'WeekNum', 'Year'])
                     .agg({'INVOICE_WT': 'sum', 'PURCHASED_QUANTITY_Detail': 'sum', 'PURCHASED_PRICE_Detail': 'mean',
                           'Purchases_Value': 'sum', 'CATALOG_ITEM_ID': 'count'})
                     .sort_values(by='DATE', ascending=True)).reset_index()
        _bc = 'B'
        _comp = 'A'
    elif buying_center == 'A' and component != 'A':
        filter_df = (filter_df[filter_df.FULL_COMPONENT_CODE == component]
                     .groupby(['DATE', 'WeekNum', 'Year'])
                     .agg({'INVOICE_WT': 'sum', 'PURCHASED_QUANTITY_Detail': 'sum', 'PURCHASED_PRICE_Detail': 'mean',
                           'Purchases_Value': 'sum', 'CATALOG_ITEM_ID': 'count'})
                     .sort_values(by='DATE', ascending=True)).reset_index()
        _bc = 'A'
        _comp = 'C'
    else:
        filter_df = (filter_df[(filter_df.FULL_COMPONENT_CODE == component) & (filter_df.BUYING_CENTER_CODE == buying_center)]
                     .groupby(['DATE', 'WeekNum', 'Year'])
                     .agg({'INVOICE_WT': 'sum', 'PURCHASED_QUANTITY_Detail': 'sum', 'PURCHASED_PRICE_Detail': 'mean',
                           'Purchases_Value': 'sum', 'CATALOG_ITEM_ID': 'count'})
                     .sort_values(by='DATE', ascending=True)).reset_index()
        _bc = 'B'
        _comp = 'C'
        
    # Initialize the output DataFrame
    output = pd.DataFrame()
    
    # Calculate previous price and average price difference
    filter_df['PrevPrice'] = filter_df['PURCHASED_PRICE_Detail'].shift(1)
    filter_df['AvgPriceDiff'] = filter_df['PURCHASED_PRICE_Detail'] - filter_df['PrevPrice']
    
    # Filter data for the same week in the previous years
    weekdf1 = filter_df[(filter_df.WeekNum == week) & (filter_df.Year == year - 1)]
    weekdf2 = filter_df[(filter_df.WeekNum == week) & (filter_df.Year == year - 2)]
    
    # Prefix for output columns
    pref = f"{company}_{channel}_{_bc}_{_comp}"
    output['DATE'] = filter_df['DATE']
    
    # Dictionary mapping parameters to DataFrame columns
    parameter_dict = {"AP": "PURCHASED_PRICE_Detail", "PD": "AvgPriceDiff", "PQ": "PURCHASED_QUANTITY_Detail",
                      "PV": "Purchases_Value", "IW": "INVOICE_WT", "NL": "CATALOG_ITEM_ID"}
    
    # Calculate the moving average or static window average based on the 'ma' parameter
    if ma == 'MA':
        output[f"{pref}_{param}_MA{value}"] = filter_df[parameter_dict[param]].rolling(window=value, min_periods=1).mean()
    else:
        if value == 1:
            output[f"{pref}_{param}_SW{1}Y"] = weekdf1[parameter_dict[param]].mean()
        else:
            output[f"{pref}_{param}_SW{2}Y"] = weekdf2[parameter_dict[param]].mean()
    
    # Return the last row of the output DataFrame, excluding the 'DATE' column
    return output.iloc[-1:, 1:].reset_index(drop=True)


def buying_center_feat(date, bc, comp,df,bc_feat):
    """
    Generate features for a specific buying center and component based on the given date.

    Parameters:
    date (str): The date for which features need to be generated.
    bc (str): Buying center code.
    comp (str): Component code.

    Returns:
    pd.DataFrame: DataFrame containing the generated features.
    """
    # Initialize an empty DataFrame for output
    out = pd.DataFrame()
    
    # Convert the input date to a datetime object and extract the week number and year
    week = pd.to_datetime(date).isocalendar().week
    year = pd.to_datetime(date).isocalendar().year
    
    # Loop through each combination in the bc_feat list
    for comb in bc_feat:
        # Split the combination string into a list
        comb_list = comb.split('_')
        
        # Extract integers and parameters from the combination list
        integers = re.findall(r'\d+', comb_list[5])[0]
        para = re.sub('[0-9]*', '', comb_list[5])
        
        # Check and apply different conditions based on the combination list
        if comb_list[2] != 'A':
            if comb_list[3] != 'A':
                temp = feature_gen1(df, week, year, company=comb_list[0], channel=comb_list[1], buying_center=bc, component=comp, param=comb_list[4], ma=para, value=int(integers))
            else:
                temp = feature_gen1(df, week, year, company=comb_list[0], channel=comb_list[1], buying_center=bc, component="A", param=comb_list[4], ma=para, value=int(integers))
        else:
            if comb_list[3] != 'A':
                temp = feature_gen1(df, week, year, company=comb_list[0], channel=comb_list[1], buying_center="A", component=comp, param=comb_list[4], ma=para, value=int(integers))
            else:
                temp = feature_gen1(df, week, year, company=comb_list[0], channel=comb_list[1], buying_center="A", component="A", param=comb_list[4], ma=para, value=int(integers))
        
        # Concatenate the temporary DataFrame to the output DataFrame
        out = pd.concat([out, temp], axis=1)
    
    # Return the final output DataFrame
    return out



def input_preprocessing(input_data,garden_std):
    logger.info(f"Before Pre Processing Input Data data shape : {input_data.shape[0]}")
    # logger.info(f"Before Pre Processing Input Data data shape : {input_data.columns}")
    
    input_data.rename(columns = {"Buying Center":'BUYING_CENTER_CODE','Material Code':'FULL_COMPONENT_CODE',
                            "Garden Code":"GARDEN_CODE",
                            'Quality Variance':'QUALITY_VARIANCE','Offer Quantity':'INVOICE_WT'}, inplace = True)
    
    # Removing the records where the full component code is missing.
    rejected_data_1 = input_data[(input_data['FULL_COMPONENT_CODE'].isnull())]
    rejected_data_2 = input_data[(input_data['GARDEN_CODE'].isnull())]
    
    input_data =input_data[~input_data['FULL_COMPONENT_CODE'].isnull()].reset_index(drop = True)
    input_data =input_data[~input_data['GARDEN_CODE'].isnull()].reset_index(drop = True)
    
    
    smooth = COMPONENT_QUADRANTS['smooth']
    intermittent = COMPONENT_QUADRANTS['intermittent']
    lumpy = COMPONENT_QUADRANTS['lumpy']
    
    # Based on the component assgin quadrants.
    input_data['Quadrant'] = np.where(input_data.FULL_COMPONENT_CODE.isin(smooth),'Smooth',
                                     np.where(input_data.FULL_COMPONENT_CODE.isin(intermittent),'Intermittent',
                                      'Lumpy'))
    # Quality Variance Encode. 
    input_data['QUALITY_VARIANCE'] =np.where(input_data['QUALITY_VARIANCE']=='+',1,
                                             np.where(input_data['QUALITY_VARIANCE'] =='=',0,
                                                      -1))
    # Standard Garden Name Mapping
    garden_std = garden_std[["GARDEN_CODE","Standardized Garden"]].drop_duplicates().reset_index(drop = True)
    input_data = input_data.merge(garden_std, on = 'GARDEN_CODE', how = 'left')
    input_data.drop_duplicates(inplace = True)
    input_data.rename(columns ={'GARDEN_CODE':'GARDEN_CODE1', "Standardized Garden":"GARDEN_CODE"}, inplace = True)
    
    # Taking the Garden where the standardized name was not provided.
    non_std_gardens_list = input_data[input_data.GARDEN_CODE.isnull()].GARDEN_CODE1.unique().tolist()
    logger.info(f"Number of Non-standard Gardens: {len(non_std_gardens_list)}.\n {non_std_gardens_list}")
    
    # If Standard Garden Name is not specified then using the input name as is.
    input_data.GARDEN_CODE = np.where(input_data.GARDEN_CODE.isnull(), input_data.GARDEN_CODE1, input_data.GARDEN_CODE)
    
    # Creating the BC,GC,GE,CR columns to create input features.
    input_data['BC'] = input_data['BUYING_CENTER_CODE']+'_'+input_data['FULL_COMPONENT_CODE']
    input_data['GC'] = input_data['GARDEN_CODE']+'_'+input_data['FULL_COMPONENT_CODE']
    input_data['GE'] = input_data['GARDEN_CODE']+'_'+input_data['GRADE_CODE']
    input_data['CR'] = input_data['FULL_COMPONENT_CODE']+'_'+input_data['QUALITY_VARIANCE'].astype(str)
    logger.info(f"After Pre Processing Input Data data shape : {input_data.shape[0]}")
    return input_data,rejected_data_1,rejected_data_2

# Define a function to process snapshot data for a given object
def process_snap(obj, comb):
    return obj.getSnap(comb)

def getGC_feat_2(gc_comb,obj_list,key="GE"):
    # List of objects and combination
    objects = obj_list
    combination = gc_comb

    # Create a multiprocessing Pool with the number of available CPU cores
    pool = Pool()

    # Apply parallel processing to get snapshot data for each object
    snapshots = pool.starmap(process_snap, [(obj, combination) for obj in objects])

    # Close the pool of processes and wait for all processes to complete
    pool.close()
    pool.join()

    # Merge the snapshot data obtained from different objects
    temp6 = snapshots[0]
    for temp in snapshots[1:]:
        temp6 = temp6.merge(temp, on=key, how='left')
    return temp6

def gc_feature_generation(input_data):
    obj = Garden(date=RUN_DATE)
    tu_obj =Garden('T','U',date = RUN_DATE)
    tp_obj =Garden('T','P',date = RUN_DATE)
    ta_obj =Garden('T','A',date = RUN_DATE)
    cu_obj =Garden('C','A',date = RUN_DATE)
    
    gc_objects = [obj, tu_obj, tp_obj, ta_obj, cu_obj]
    
    gc_combinations = input_data['GC'].unique().tolist()
    logger.info(f"Number of unique GC combinations: {len(gc_combinations)}")
    
    gc_features = getGC_feat_2(gc_combinations,gc_objects,'GC' )

    return gc_features

def ge_feature_generation(input_data):
    obj = GardenGrade(date=RUN_DATE)
    tu_obj =GardenGrade('T','U',date = RUN_DATE)
    tp_obj =GardenGrade('T','P',date = RUN_DATE)
    ta_obj =GardenGrade('T','A',date = RUN_DATE)
    cu_obj =GardenGrade('C','A',date = RUN_DATE)
    
    gc_objects = [obj, tu_obj, tp_obj, ta_obj, cu_obj]
    
    gc_combinations = input_data['GE'].unique().tolist()
    logger.info(f"Number of unique GE combinations: {len(gc_combinations)}")
    
    gc_features = getGC_feat_2(gc_combinations,gc_objects,'GE' )

    return gc_features

def cr_feature_generation(input_data):
    obj = CmpRating(date=RUN_DATE)
    tu_obj =CmpRating('T','U',date = RUN_DATE)
    tp_obj =CmpRating('T','P',date = RUN_DATE)
    ta_obj =CmpRating('T','A',date = RUN_DATE)
    cu_obj =CmpRating('C','A',date = RUN_DATE)
    
    gc_objects = [obj, tu_obj, tp_obj, ta_obj, cu_obj]
    
    gc_combinations = input_data['CR'].unique().tolist()
    logger.info(f"Number of unique CR combinations: {len(gc_combinations)}")
    
    gc_features = getGC_feat_2(gc_combinations,gc_objects,'CR' )

    return gc_features

def gc_missing_imputation2(gc,model_input_df,avg_price):
    """
    Imputes the missing price for a given garden_component (gc) by calculating
    the average price based on component and garden name.

    Parameters:
    gc (str): A string in the format "garden_component" where garden is the garden name and component is the component code.

    Returns:
    float: The imputed price based on the average prices of the given component and garden.
    """
   
    try:
    # Split the input string to get garden and component separately
        a = gc.split("_")
        garden = a[0]
        comp = a[1]

        # Calculate the average price for the given component
        comp_avg = model_input_df[model_input_df.FULL_COMPONENT_CODE == comp].PURCHASED_PRICE_Detail.mean()
        
        # Calculate the average price for the given garden
        gard_avg = model_input_df[model_input_df.GARDEN_NAME == garden].PURCHASED_PRICE_Detail.mean()
        
        # If garden average price is NaN, replace it with the global average price
        if str(gard_avg) == 'nan':
            gard_avg = avg_price
        
        # Normalize the component and garden average prices by the global average price
        comp_avg /= avg_price
        gard_avg /= avg_price

        # Calculate the output price by multiplying the normalized component and garden averages with the global average price
        out = avg_price * comp_avg * gard_avg
    except:
        print(gc)
        out = avg_price
    
    return out

def input_feature_generation(input_data,master,model_input_df):
    
    date = RUN_DATE
    df=(master.groupby(['DATE','WeekNum','Year','Purchase_Flag','Channel',
                          'BUYING_CENTER_CODE','FULL_COMPONENT_CODE']).agg({
                            'INVOICE_WT':'sum', 'PURCHASED_QUANTITY_Detail':'sum', 'PURCHASED_PRICE_Detail':'mean',
                            'Purchases_Value':'sum','CATALOG_ITEM_ID':'count'
                        }).sort_values(by='DATE', ascending=True)).reset_index()
    df['Channel']=df['Channel'].str.replace('Auction','U')
    df['Channel']=df['Channel'].str.replace('Private','P')
    df["Purchase_Flag"] = df["Purchase_Flag"].str.replace("C",'O')
    
    
    logger.info("BC Feature Creation Started...")
    
    bc_combination = input_data['BC'].unique().tolist()
    logger.info(f"Number of unique BC combinations: {len(bc_combination)}")
    
    bc_features = pd.DataFrame()
    for combination in bc_combination:
        split_bc = combination.split("_")
        buying_center = split_bc[0]
        component = split_bc[1]
        temp_out =buying_center_feat(RUN_DATE,buying_center,component,df, BC_FEATURES)
        temp_out['BC'] = combination
        bc_features = pd.concat([temp_out,bc_features])
    logger.info("BC Feature Creation is done...")
    logger.info("GC Feature Creation Started...")
    
    gc_features = gc_feature_generation(input_data)
    logger.info("GC Feature Creation is done...")
    logger.info("GE Feature Creation Started...")
    
    ge_features = ge_feature_generation(input_data)
    logger.info("GE Feature Creation is done...")
    logger.info("CR Feature Creation Started...")
    
    cr_features = cr_feature_generation(input_data)
    logger.info("CR Feature Creation is done...")
    
    
    avg_price = model_input_df["PURCHASED_PRICE_Detail"].mean()
    gc_features.columns = gc_features.columns.str.replace("^C_A","O_U", regex = True)
    ge_features.columns = ge_features.columns.str.replace("^C_A","O_U", regex = True)
    cr_features.columns = cr_features.columns.str.replace("^C_A","O_U", regex = True)
    
    missing_ap1 = gc_features[gc_features.A_A_G_C_AP_MA1.isnull()].GC.tolist()
    logger.info(f"GC Missing value Imputation for new GC strated")
    
    mis_val_imp = {}
    for m_gc_comb in missing_ap1:
        mis_val_imp[m_gc_comb] = gc_missing_imputation2(m_gc_comb,model_input_df,avg_price)
        # logger.info(f"GC Missing value Imputation is done for : {m_gc_comb}")
    
    gc_features["A_A_G_C_AP_MA1"] = gc_features[["A_A_G_C_AP_MA1","GC"]].apply(lambda x: mis_val_imp[str(x[1])] if str(x[0])=='nan' else x[0], axis = 1 )
    gc_features =gc_features[['GC','A_A_G_C_AP_MA1']+GC_FEATURES]
    logger.info(f"GC Missing value Imputation for new GC finished...")
    
    # Meging GC features in the input file    
    input_data = input_data.merge(gc_features,on = 'GC',how = 'left')
    logger.info(f"GC merge Input Data data shape : {input_data.shape[0]}")
    
    input_data = input_data.merge(ge_features,on = 'GE',how = 'left')
    logger.info(f"GC merge Input Data data shape : {input_data.shape[0]}")
    
    input_data = input_data.merge(cr_features,on = 'CR',how = 'left')
    logger.info(f"GC merge Input Data data shape : {input_data.shape[0]}")
    
    input_data = input_data.merge(bc_features,on = 'BC',how = 'left')
    logger.info(f"GC merge Input Data data shape : {input_data.shape[0]}")
    input_data.drop_duplicates(inplace = True)
    input_data.reset_index(drop = True, inplace = True)
    

    return input_data


def model_input_preprocessing(input_data, model_input_df,channel):
    input_data.rename(columns = {"A_A_G_C_AP_MA1":'Taget_Price'}, inplace = True)
    logger.info(f"Input columns : {input_data.columns}")
    selt_col_dic ={"Auction": ['CATALOG_ITEM_ID', 'CATALOG_HEADER_ID', 'BUYING_CENTER_CODE',
       'BUYING_TYPE_ID', 'LOT_NUMBER', 'INVOICE_NUMBER', 'GARDEN_CODE1',
       'GRADE_CODE', 'FULL_COMPONENT_CODE', 'QUALITY_VARIANCE', 'INVOICE_WT',
       'SALE_DATE',
                'Quadrant',
                'GARDEN_CODE','A_A_G_C_AP_MA2',
                                'T_U_G_C_PD_MA5',
                                'T_A_G_C_AP_MA1',
                                'O_U_G_C_PD_MA2',
                                'A_A_G_C_PD_MA5',
                                'A_A_G_C_PD_MA10',
                                'A_U_B_C_PD_MA2',
                                'A_A_G_E_AP_MA3',
                                'O_U_G_E_AP_MA3',
                                'T_A_G_E_AP_MA3',
                                'T_U_G_E_AP_MA1',
                                'A_A_C_R_AP_MA3',
                                'T_A_C_R_AP_MA3',
                                'T_A_C_R_AP_MA1',
                                'T_P_G_C_PD_MA5',
                                'T_P_A_C_PD_MA20',
                                'T_P_C_R_AP_MA3',
                                'T_P_C_R_AP_MA5',
                                'Taget_Price',],
               "Private":['CATALOG_ITEM_ID', 'CATALOG_HEADER_ID', 'BUYING_CENTER_CODE',
       'BUYING_TYPE_ID', 'LOT_NUMBER', 'INVOICE_NUMBER', 'GARDEN_CODE1',
       'GRADE_CODE', 'FULL_COMPONENT_CODE', 'QUALITY_VARIANCE', 'INVOICE_WT',
       'OFFER_DATE',
                'Quadrant',
                'GARDEN_CODE','A_A_G_C_AP_MA2',
                                'T_U_G_C_PD_MA5',
                                'T_A_G_C_AP_MA1',
                                'O_U_G_C_PD_MA2',
                                'A_A_G_C_PD_MA5',
                                'A_A_G_C_PD_MA10',
                                'A_U_B_C_PD_MA2',
                                'A_A_G_E_AP_MA3',
                                'O_U_G_E_AP_MA3',
                                'T_A_G_E_AP_MA3',
                                'T_U_G_E_AP_MA1',
                                'A_A_C_R_AP_MA3',
                                'T_A_C_R_AP_MA3',
                                'T_A_C_R_AP_MA1',
                                'T_P_G_C_PD_MA5',
                                'T_P_A_C_PD_MA20',
                                'T_P_C_R_AP_MA3',
                                'T_P_C_R_AP_MA5',
                                'Taget_Price',]}
    
    selt_col = selt_col_dic[channel]
    input_data = input_data[selt_col].reset_index(drop = True)
    
    
    last_90_days = model_input_df[model_input_df.DATE>(model_input_df.DATE.max()-pd.to_timedelta("90 days"))]
    lc_dic = {}
    for key in last_90_days.iloc[:,15:-7].columns:
        lc_dic[key]='mean'
        
    last_90_days =last_90_days.groupby(['FULL_COMPONENT_CODE','PURCHASED_PRICE_Detail']).agg(lc_dic)
    last_90_days.reset_index(inplace = True)
    
    
    def func_to_fill(comp, col, target_price):
        """
        Fills the missing value for a specific component and column based on the target price.
        The function uses a cached dictionary to store and retrieve previously calculated values.

        Parameters:
        comp (str): The component code.
        col (str): The column name for which the missing value needs to be filled.
        target_price (float): The target price around which to find the average value.

        Returns:
        float: The filled value for the given component and column based on the target price.
        """
        # Construct a unique key for the cache dictionary
        key = f"{comp}_{col}_{target_price}"

        # Check if the key is already in the cache
        if key not in func_to_dic.keys():
            # Select relevant columns from the input DataFrame
            df = last_90_days[['PURCHASED_PRICE_Detail', "FULL_COMPONENT_CODE", col]]
            
            # Define a tolerance range around the target price
            tolerance = 5
            
            # Filter the DataFrame for rows matching the component and within the tolerance range of the target price
            a = df[(df.FULL_COMPONENT_CODE == comp) & 
                (df['PURCHASED_PRICE_Detail'] >= target_price - tolerance) & 
                (df['PURCHASED_PRICE_Detail'] <= target_price + tolerance)][col].mean()
            
            # If a valid average value is found, store it in the cache and return it
            if a > 0:
                func_to_dic[key] = a
                return a
            else:
                # If no valid value is found within the tolerance range, calculate the overall average for the component
                a = df[(df.FULL_COMPONENT_CODE == comp)][col].mean()
                
                # Store the overall average in the cache and return it
                func_to_dic[key] = a
                return a
        else:
            # If the key is already in the cache, return the cached value
            return func_to_dic[key]
        
    col_fill = ['A_A_G_C_AP_MA2',
                'T_U_G_C_PD_MA5',
                'T_A_G_C_AP_MA1',
                'O_U_G_C_PD_MA2',
                'A_A_G_C_PD_MA5',
                'A_A_G_C_PD_MA10',
                'A_U_B_C_PD_MA2',
                'T_P_G_C_PD_MA5',
                'T_P_A_C_PD_MA20']
    
    
    
    func_to_dic = {}
    for col in col_fill:
        input_data[col] = input_data[col].fillna(input_data.apply(lambda x: func_to_fill(x['FULL_COMPONENT_CODE'], col, x['Taget_Price']), axis=1))

    grade_encode = pd.read_csv(GLOBAL_PATHS["GRADE_ENCODE_FILE"])
    input_data['key'] = input_data["FULL_COMPONENT_CODE"]+"_" +input_data['GRADE_CODE']
    input_data = input_data.merge(grade_encode, on = 'key',how = 'left' )
    input_data.fillna(0, inplace = True)
    input_data.columns = input_data.columns.str.replace('^C_A',"O_U",regex = True)
    
    return input_data

def get_pickle_file(channel,seg):
    # Set up your AWS S3 credentials (this assumes you have your credentials configured in the environment)
    s3 = boto3.client('s3')

    # Define your bucket name and the path to your model file
    bucket_name = 'tcpl-buyingdecision'
    path = GLOBAL_PATHS["MODEL_PICKLE_FILE"]
    model_file_path = f"{path}/{seg}_GradientBoostingRegressor_{channel}_v1_2.pkl"
    print(model_file_path)
    # Download the file from S3 to the local file system
    local_file_name = f'{seg}_GradientBoostingRegressor_{channel}_v1_2.pkl'
    s3.download_file(bucket_name, model_file_path, local_file_name)
    
    with open(f'{seg}_GradientBoostingRegressor_{channel}_v1_2.pkl','rb') as f:
        model = pickle.load(f)
    return model


def segment_prediction(input_data,channel,quadrant,seg,components=None,non= False):
    cols_to_be_droped = ['A_A_G_C_AP_MA2',
                        'T_U_G_C_PD_MA5',
                        'T_A_G_C_AP_MA1',
                        'O_U_G_C_PD_MA2',
                        'A_A_G_C_PD_MA5',
                        'A_A_G_C_PD_MA10',
                        'A_U_B_C_PD_MA2',
                        'A_A_G_E_AP_MA3',
                        'O_U_G_E_AP_MA3',
                        'T_A_G_E_AP_MA3',
                        'T_U_G_E_AP_MA1',
                        'A_A_C_R_AP_MA3',
                        'T_A_C_R_AP_MA3',
                        'T_A_C_R_AP_MA1',
                        'T_P_G_C_PD_MA5',
                        'T_P_A_C_PD_MA20',
                        'T_P_C_R_AP_MA3',
                        'T_P_C_R_AP_MA5']

    if components ==None:
        x_test = input_data[input_data['Quadrant']==quadrant][MODEL_INPUT_FEATURES[channel]].reset_index(drop = True)
        
        x_test_with_lot_info = input_data[input_data['Quadrant']==quadrant].reset_index(drop = True)
        x_test_with_lot_info  = x_test_with_lot_info.drop(columns = cols_to_be_droped)
    elif non:
        x_test = input_data[(input_data.FULL_COMPONENT_CODE.isin(components))][MODEL_INPUT_FEATURES[channel]].reset_index(drop = True)
        x_test_with_lot_info = input_data[(input_data.FULL_COMPONENT_CODE.isin(components))].reset_index(drop = True)
        x_test_with_lot_info  = x_test_with_lot_info.drop(columns = cols_to_be_droped)
        
    else:
        x_test = input_data[(input_data['Quadrant']==quadrant)&(input_data.FULL_COMPONENT_CODE.isin(components))][MODEL_INPUT_FEATURES[channel]].reset_index(drop = True)
        x_test_with_lot_info = input_data[(input_data['Quadrant']==quadrant)&(input_data.FULL_COMPONENT_CODE.isin(components))].reset_index(drop = True)
        x_test_with_lot_info  = x_test_with_lot_info.drop(columns = cols_to_be_droped)
    
    logger.info(f"input columns {x_test.columns}")
    model = get_pickle_file(channel,seg) 
    x_test_with_lot_info['MODEL_PREDICTION'] =model.predict(x_test) 
    MODEL_VERSION = f"{seg}_V-1-2"
    x_test_with_lot_info['MODEL_VERSION']=MODEL_VERSION
    
    return x_test_with_lot_info
    
    
def assign_rank(output):
    df = output.copy()
    df['CR'] = df['FULL_COMPONENT_CODE']+"_"+df['QUALITY_VARIANCE'].astype(str)
    uniq_cr = df['CR'].unique().tolist()
    out = pd.DataFrame()
    for cr in uniq_cr:
        temp = df[df['CR']==cr].reset_index(drop = True)
        temp.sort_values(by = 'MODEL_PREDICTION', inplace = True,ascending = True)
        temp = temp.reset_index(drop = True)
        temp['MODEL_RANK_CR'] = 1
        for i in range(1,temp.shape[0]):
            if temp.loc[i,'MODEL_PREDICTION']==temp.loc[i-1,'MODEL_PREDICTION']:
                temp.loc[i,'MODEL_RANK_CR'] = temp.loc[i-1,'MODEL_RANK_CR']
            else:
                temp.loc[i,'MODEL_RANK_CR'] = temp.loc[i-1,'MODEL_RANK_CR']+1
        out = pd.concat([out, temp])
    return out
    
def main(channel = 'Auction'):
    try:
        master = get_master_data()
        garden_std = get_garden_std_data()
        model_input_df = get_model_train_data()
        input_data = get_input_data(channel)
        pass
    except Exception as e:
        logger.error(f" Error while getting the Data from S3 {e}")
        
    # input data Preprocessing
    logger.info("Input Pre Processing Stated...")
    input_data,rejected_data_1,rejected_data_2 = input_preprocessing(input_data,garden_std)
    logger.info("Input Pre Processing is done...")
    
    # feature creation
    logger.info("Input Feature Creation Stated...")
    logger.info(f" Before Feature creation Input Data data shape : {input_data.shape[0]}")
    
    input_data = input_feature_generation(input_data,master,model_input_df)
    logger.info("Input Feature creation is done...")
    logger.info(f" After Feature creation Input Data data shape : {input_data.shape[0]}")
    
    # feature preparation
    logger.info("Model Input Pre Processing Stated...")
    logger.info(f" Before model pre processing Input Data data shape : {input_data.shape[0]}")
    
    input_data = model_input_preprocessing(input_data,model_input_df,channel)
    logger.info("Model Input Pre Processing is doneo...")
    logger.info(f" After model pre processing Input Data data shape : {input_data.shape[0]}")
    
    
    
    # Smooth Components Model Training. 
    output_df_list = []
    # try:
    logger.info("Model Prediction for Smooth Components Started...")
    smooth_pred= segment_prediction(input_data,channel,"Smooth","Smooth")
    logger.info(f"Smooth Prediction shape : {smooth_pred.shape[0]}")
    output_df_list.append(smooth_pred)
    # except Exception as e:
        # logger.error(" Error while training for Smooth Components {}".format(e))
        
    # Intermittent Components Model Training. 
    try:
        logger.info("Model training for Intermittent Components Started...")
        int_pred= segment_prediction(input_data,channel,"Intermittent","Intermittent",)
        logger.info(f"Intermittent Prediction shape : {int_pred.shape[0]}")
        
        output_df_list.append(int_pred)
    except Exception as e:
        logger.error(" Error while training for Intermittent Components {}".format(e))
    
    
    # Lumpy_bin_1 Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_1 Components Started...")
        
        l_b1_pred= segment_prediction(input_data,channel,"Lumpy","Lumpy_bin_1",Lumpy_bin_1)
        logger.info(f"Lumpy_bin_1 Prediction shape : {l_b1_pred.shape[0]}")
        
        output_df_list.append(l_b1_pred)
    except Exception as e:
        logger.error(" Error while training for Smooth Components {}".format(e))
    
    # Smooth Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_2 Components Started...")
        l_b2_pred= segment_prediction(input_data,channel,"Lumpy","Lumpy_bin_2",Lumpy_bin_2)
        logger.info(f"Lumpy_bin_2 Prediction shape : {l_b2_pred.shape[0]}")
        
        output_df_list.append(l_b2_pred)
    except Exception as e:
        logger.error(" Error while training for Smooth Components {}".format(e))
    
    # Lumpy_bin_3 Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_3 Components Started...")
        l_b3_pred= segment_prediction(input_data,channel,"Lumpy","Lumpy_bin_3",Lumpy_bin_3)
        logger.info(f"Lumpy_bin_3 Prediction shape : {l_b3_pred.shape[0]}")
        
        output_df_list.append(l_b3_pred)
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_3 Components {}".format(e))
    
    # Lumpy_bin_4 Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_4 Components Started...")
        l_b4_pred= segment_prediction(input_data,channel,"Lumpy","Lumpy_bin_4",Lumpy_bin_4)
        logger.info(f"Lumpy_bin_4 Prediction shape : {l_b4_pred.shape[0]}")
        
        output_df_list.append(l_b4_pred)
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_4 Components {}".format(e))
    
    
    try:
        logger.info("Model training for Lumpy_bin_5 Components Started...")
        l_b5_pred= segment_prediction(input_data,channel,"Lumpy","Lumpy_bin_5",Lumpy_bin_5)
        logger.info(f"Lumpy_bin_5 Prediction shape : {l_b5_pred.shape[0]}")
        
        output_df_list.append(l_b5_pred)
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_5 Components {}".format(e))
        
    input_component_list = input_data['FULL_COMPONENT_CODE'].unique().tolist() 
    prediction_all = pd.concat(output_df_list)
    logger.info(f"prediction_all shape : {prediction_all.shape[0]}")
    
    try:
        predicted_component_list =prediction_all['FULL_COMPONENT_CODE'].unique().tolist()
        non_tcpl_component_list = [comp for comp in input_component_list if comp not in predicted_component_list]
        
        print(non_tcpl_component_list)
        
        non_tcpl_pred= segment_prediction(input_data,channel,"Intermittent","Intermittent",non_tcpl_component_list,True)
        logger.info(f"non_tcpl_pred shape : {non_tcpl_pred.shape[0]}")
        
        rejected_data_1['Comments'] = 'Full Component Code Missing'
        rejected_data_2['Comments'] = "Garden Name Missing"
        prediction_all = pd.concat([prediction_all, non_tcpl_pred, rejected_data_1]  )
    except Exception as e:
        logger.error(" Error while training for non_tcpl_component_list Components {}".format(e))
    prediction_all.drop(columns = ['GARDEN_CODE',"Taget_Price","key","Code"], inplace = True)
    
    prediction_all["MODEL_PREDICTION"] =prediction_all["MODEL_PREDICTION"].round(2)
    db_output = assign_rank(prediction_all)
    db_output.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/model-prediction-output-backup/{RUN_DATE}/{channel}/Model_Prediction.csv', index = False)
    db_output =db_output[["CATALOG_ITEM_ID",
                            "CATALOG_HEADER_ID",
                            "BUYING_TYPE_ID",
                            "MODEL_PREDICTION",
                            "MODEL_RANK_CR",
                            "MODEL_VERSION",
                                ]]
    db_output['PREDICTION_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r_id = f"{t_now}_1"
    db_output['MODEL_RUN_ID'] = r_id
    
    db_output.to_parquet(f"s3://tcpl-buyingdecision/bgdn-pre-prod/model-prediction-output/{channel}/Model_Prediction.snappy.parquet", index = False)
if __name__=='__main__':
    logger.info(f"Starting Main Process for Auction Channel...")
    main("Auction")
    logger.info(f"Finished Main Process for Auction Channel...")
    logger.info(f"Starting Main Process for Private Channel...")
    main("Private")
    logger.info(f"Finished Main Process for Private Channel...")
    