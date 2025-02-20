# Buying Decision Support Project
# Team@Stratlytics
# @author : Bala.
# File Information: This file is to create model features for the incremental data and update the model input file with the delta data.

import subprocess
import sys
# Install required packages with compatible versions
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install","awswrangler"])
    print("Installing packages successfully done.")
except Exception as e:
    print(f"Error installing packages. {e}")

# -----------------------------Imports-----------------------------------
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import logging
from pathlib import Path
import awswrangler as wr
import numpy as np 
import re
import time
from multiprocessing import Pool
import datetime
import boto3
import psutil
# ----------------------------- Code -----------------------------------


# Get the total memory in bytes
total_memory = psutil.virtual_memory().total

# Convert the memory size to GB
total_memory_gb = total_memory / (1024 ** 3)

# Print the total memory size in GB
print(f"Total memory available: {total_memory_gb:.2f} GB")


BC_FEATURES =[ "A_U_B_C_PD_MA2","T_P_A_C_PD_MA20"]
MODEL_INPUT_FEATURES = { "Auction":['INVOICE_WT','A_A_G_C_AP_MA2', 'T_U_G_C_PD_MA5', 'T_A_G_C_AP_MA1',
                                   'O_U_G_C_PD_MA2', 'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10', 'A_U_B_C_PD_MA2',
                                   'A_A_G_E_AP_MA3', 'O_U_G_E_AP_MA3', 'T_A_G_E_AP_MA3', 'T_U_G_E_AP_MA1',
                                   'A_A_C_R_AP_MA3', 'T_A_C_R_AP_MA3', 'T_A_C_R_AP_MA1','QUALITY_VARIANCE',
                                    'Code'
                                   ],
                        "Private": ['INVOICE_WT', 'QUALITY_VARIANCE', 'T_U_G_C_PD_MA5', 'T_A_G_C_AP_MA1', 'T_P_G_C_PD_MA5',
                                   'T_P_A_C_PD_MA20', 'T_A_G_E_AP_MA3', 'T_U_G_E_AP_MA1', 'T_A_C_R_AP_MA3',
                                   'T_P_C_R_AP_MA3', 'T_P_C_R_AP_MA5', 'Code',
                                   ]}

        
# Following dictionary contains the location for fetching the data
GLOBAL_PATHS= "removed"
BACKEND_ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


# The following list contains the TCPL ID's which are in competitor competitor dataframe that needs to be removed from the competitor item table
TCPL_COMPT_ID = [
    4323,
    4035,
    2049,
    4046,
    4045,
    4001,
    4047
]
RUN_DATE = datetime.date.today().strftime("%d-%b-%y")
# RUN_DATE = '2024-05-24 00:00:00'

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

# --- Exception Handling-------
class BadInputFile(Exception):  # 400
    def __init__(
        self,
        status_msg="Invalid Data Received",
        category="error",
    ):
        logger.info(f"BadInputFile {category} - {status_msg}")
        
class NoInputFile(Exception):
    def __init__(self,path):
        logger.error(f"No input file in the provided path :{path}")

class NoDataToUpdate(Exception):
    def __init__(self):
        logger.error(f"No Incremental data to update...")
        
class S3ReadError(Exception):
    def __init__(self,e):
        logger.error(f"Error while reading Data from S3 :{e}")  


# -------------------- Functions------------------------------------------
"""
Methods available: debug, info, warning, error, critical
ex: logging.debug('This is a debug message')
"""
# S3 Configuration
LOG_TIME = datetime.date.today().strftime("%d-%b-%y %H:%M")
S3_BUCKET_NAME = 'tcpl-buyingdecision'
S3_LOGS_PATH = f'bgdn-pre-prod/dev-pipeline/logs/{LOG_TIME}/' 

def upload_to_s3(local_file_path, s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    s3_client.upload_file(str(local_file_path), s3_bucket, s3_key)

def get_module_logger(module_name, log_level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist (locally, for intermediate storage)
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

    # Upload the log file to S3 after each log message is written
    def upload_log_to_s3(_):
        s3_key = f"{S3_LOGS_PATH}{module_name}.log"
        upload_to_s3(log_file, S3_BUCKET_NAME, s3_key)

    # Attach the S3 upload function to the file handler
    file_handler.addFilter(upload_log_to_s3)

    return logger

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

class CmpRating():
    def __init__(self, company="A", channel="A",date=None, loc='s3://tcpl-buyingdecision/buying-decision-data/01_Stratlytics/10_Feature_Creation_Input_Files/03_Component_Rating_Template'):
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
        self.template_price = pd.read_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv')
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
        
        sav_col = self.template_price.columns[-min(len(self.template_price.columns)-1,730):]
        sav_col = ['CR']+list(sav_col)
        self.template_price = self.template_price[ sav_col]
        
        self.template_price.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/component-rating-templates/{self.__comp}_{self.__chnl}_AP_Template_df.csv', index=False)

class Garden():
    def __init__(self, company="A", channel="A", loc='s3://tcpl-buyingdecision/buying-decision-data/01_Stratlytics/10_Feature_Creation_Input_Files/01_Garden_Component_Template',date = None):
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
        self.template_price = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv')
        # self.template_pq = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_PQ_Template_df_UAT_5.csv')
        # self.template_Iw = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_IW_Template_df_UAT_5.csv')
        self.template_diff = pd.read_csv(f'{loc}/{self.__comp}_{self.__chnl}_PD_Template_df_UAT_5.csv')
        
        if date != None:
                    ap_list = self.template_price.columns.tolist()
                    # pq_list = self.template_pq.columns.tolist()
                    # iw_list = self.template_Iw.columns.tolist()
                    pd_list = self.template_diff.columns.tolist()
                    
                    ap_list = ['GC']+[d for d in ap_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    # pq_list = ['GC']+[d for d in pq_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    # iw_list = ['GC']+[d for d in iw_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    pd_list = ['GC']+[d for d in pd_list[1:] if 'x' not in d and 'y' not in d and pd.to_datetime(d)<pd.to_datetime(date)]
                    
                    self.template_price =self.template_price[ap_list]
                    # self.template_pq =self.template_pq[pq_list]
                    # self.template_Iw =self.template_Iw[iw_list]
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
        # self.__updatePq(df[['GC', 'PURCHASED_QUANTITY_Detail']])
        # self.__updateIw(df[['GC', 'INVOICE_WT']])
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
        
        # self.__convertDict(self.template_pq, comb)
        # column2 = f"{self.__comp}_{self.__chnl}_G_C_PQ_MA"
        # out = self.__singleSnap(out, column2)
        
        # self.__convertDict(self.template_Iw, comb)
        # column3 = f"{self.__comp}_{self.__chnl}_G_C_IW_MA"
        # out = self.__singleSnap(out, column3)
        
        self.__convertDict(self.template_diff, comb)
        column4 = f"{self.__comp}_{self.__chnl}_G_C_PD_MA"
        out = self.__singleSnap(out, column4)
        
        return out
    
    def savedf(self):
        """
        Save the template DataFrames to CSV files.
        """
        sav_col = self.template_price.columns[-min(len(self.template_price.columns)-1,730):]
        sav_col = ['GC']+list(sav_col)
        self.template_price = self.template_price[sav_col]
        
        # sav_col2 = self.template_pq.columns[-min(len(self.template_pq.columns)-1,730):]
        # sav_col2 = ['GC']+list(sav_col2)
        # self.template_pq = self.template_pq[sav_col2]
        
        # sav_col3 = self.template_Iw.columns[-min(len(self.template_Iw.columns)-1,730):]
        # sav_col3 = ['GC']+list(sav_col3)
        # self.template_Iw = self.template_Iw[ sav_col3]
        
        sav_col4 = self.template_diff.columns[-min(len(self.template_diff.columns)-1,730):]
        sav_col4 = ['GC']+list(sav_col4)
        self.template_diff = self.template_diff[ sav_col4]
        
        self.template_price.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-component-templates/{self.__comp}_{self.__chnl}_AP_Template_df.csv', index=False)
        # self.template_pq.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-component-templates/{self.__comp}_{self.__chnl}_PQ_Template_df.csv', index=False)
        # self.template_Iw.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-component-templates/{self.__comp}_{self.__chnl}_IW_Template_df.csv', index=False)
        self.template_diff.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-component-templates/{self.__comp}_{self.__chnl}_PD_Template_df.csv', index=False)

class GardenGrade():
    def __init__(self, company="A", channel="A",date = None, loc='s3://tcpl-buyingdecision/buying-decision-data/01_Stratlytics/10_Feature_Creation_Input_Files/02_Garden_Grade_Template'):
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
        self.template_price = pd.read_csv(f'{self.__loc}/{self.__comp}_{self.__chnl}_AP_Template_df_UAT_5.csv')
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
        sav_col = self.template_price.columns[-min(len(self.template_price.columns)-1,730):]
        sav_col = ['GE']+list(sav_col)
        self.template_price = self.template_price[ sav_col]
        
        self.template_price.to_csv(f's3://tcpl-buyingdecision/bgdn-pre-prod/feature-templates-files/garden-grade-templates/{self.__comp}_{self.__chnl}_AP_Template_df.csv', index=False)



logger = get_module_logger("Model_Input_Update")

# Define process_snap function to handle snapshot processing
def process_snap(obj, combination):
    return obj.getSnap(combination)

def gc_feature_update(dates, running_df):
    # Initialize an empty DataFrame to store the output
    output = pd.DataFrame()

    # Create instances of the Garden class
    logger.info("Creating Garden instances.")
    obj = Garden()
    tu_obj = Garden('T', 'U')
    tp_obj = Garden('T', 'P')
    ta_obj = Garden('T', 'A')
    cu_obj = Garden('C', 'A')

    # Iterate over each date in the list of dates
    for date in dates:
        logger.info(f"Processing date: {date}")
        stime = time.time()
        
        # Filter the running DataFrame for the current date and reset the index
        filter_df = running_df[running_df.DATE == date].reset_index(drop=True)
        gc_unique_list = filter_df.GC.unique().tolist()
        logger.debug(f"Unique GC list for date {date}: {gc_unique_list}")
        
        # Group and aggregate data for the current date
        logger.info(f"Aggregating data for date {date}.")
        aa = filter_df.groupby(["DATE", "GC"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tu = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Auction')].groupby(["DATE", "GC"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tp = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Private')].groupby(["DATE", "GC"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        ta = filter_df[(filter_df.Purchase_Flag == 'T')].groupby(["DATE", "GC"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        cu = filter_df[(filter_df.Purchase_Flag == 'C') & (filter_df.Channel == 'Auction')].groupby(["DATE", "GC"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        # List of objects and combination
        combination = gc_unique_list
        
        # Create a multiprocessing Pool with the number of available CPU cores
        logger.info(f"Starting multiprocessing pool for date {date}.")
#         with Pool() as pool:
#             # Apply parallel processing to get snapshot data for each object
#             snapshots = pool.starmap(process_snap, [(o, combination) for o in objects])
        
#         # Merge the snapshot data obtained from different objects
#         logger.info(f"Merging snapshots for date {date}.")
        snapshots = [process_snap(o, combination) for o in [obj, tu_obj, tp_obj, ta_obj, cu_obj]]
        temp6 = snapshots[0]
        for temp in snapshots[1:]:
            temp6 = temp6.merge(temp, on='GC', how='left')
        
        # Merge with the original data
        filter_df = filter_df.merge(temp6, on='GC', how='left')
        
        # Add snapshots to their respective objects if not empty
        if not aa.empty:
            obj.addSnap(aa)
        
        if not tu.empty:
            tu_obj.addSnap(tu)
        
        if not tp.empty:
            tp_obj.addSnap(tp)
        
        if not ta.empty:
            ta_obj.addSnap(ta)
        
        if not cu.empty:
            cu_obj.addSnap(cu)
        
        # Concatenate the result to the final output DataFrame
        output = pd.concat([output, filter_df])
        
        etime = time.time()
        logger.info(f"Completed processing for date {date} in {etime - stime:.2f} seconds.")
    
    # Save the dataframes from each Garden instance
    logger.info("Saving dataframes from Garden instances.")
    obj.savedf()
    tu_obj.savedf()
    tp_obj.savedf()
    ta_obj.savedf()
    cu_obj.savedf()
    
    # Replace column names starting with 'C_A' with 'O_U'
    output.columns = output.columns.str.replace('^C_A', 'O_U', regex=True)
    logger.info("Completed column name replacements.")
    required_col = ['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
       'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE',
       'INVOICE_WT', 'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail','GRADE_CODE','QUALITY_VARIANCE',
       'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag', 'Purchases_Value','A_A_G_C_AP_MA2', 'T_U_G_C_PD_MA5','T_A_G_C_AP_MA1',
        'O_U_G_C_PD_MA2',  'T_P_G_C_PD_MA5',
         'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10']
    output = output[required_col]
    return output
        
def ge_feature_update(dates, running_df):
    # Initialize GardenGrade instances
    logger.info("Creating GardenGrade instances.")
    obj_ge = GardenGrade()
    tu_obj_ge = GardenGrade('T', 'U')
    tp_obj_ge = GardenGrade('T', 'P')
    ta_obj_ge = GardenGrade('T', 'A')
    cu_obj_ge = GardenGrade('C', 'A')
    
    output = pd.DataFrame()
    
    # Iterate over each date in the list of dates
    for date in dates:
        logger.info(f"Processing date: {date}")
        stime = time.time()
        
        # Filter the running DataFrame for the current date and reset the index
        filter_df = running_df[running_df.DATE == date].reset_index(drop=True)
        combination = filter_df.GE.unique().tolist()
        logger.debug(f"Unique GE list for date {date}: {combination}")
        
        # Group and aggregate data for the current date
        logger.info(f"Aggregating data for date {date}.")
        aa = filter_df.groupby(["DATE", "GE"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tu = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Auction')].groupby(["DATE", "GE"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tp = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Private')].groupby(["DATE", "GE"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        ta = filter_df[(filter_df.Purchase_Flag == 'T')].groupby(["DATE", "GE"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        cu = filter_df[(filter_df.Purchase_Flag == 'C') & (filter_df.Channel == 'Auction')].groupby(["DATE", "GE"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        # List of objects and combination
        objects = [obj_ge, tu_obj_ge, tp_obj_ge, ta_obj_ge, cu_obj_ge]
        
        # Create a multiprocessing Pool with the number of available CPU cores
        logger.info(f"Starting multiprocessing pool for date {date}.")
        pool = Pool()
        
        # Apply parallel processing to get snapshot data for each object
        snapshots = pool.starmap(process_snap, [(obj, combination) for obj in objects])
        
        # Close the pool of processes and wait for all processes to complete
        pool.close()
        pool.join()
        
        # Merge the snapshot data obtained from different objects
        logger.info(f"Merging snapshots for date {date}.")
        temp6 = snapshots[0]
        for temp in snapshots[1:]:
            temp6 = temp6.merge(temp, on='GE', how='left')
        
        filter_df = filter_df.merge(temp6, on='GE', how='left')
        
        # Add snapshots to their respective objects if not empty
        if not aa.empty:
            obj_ge.addSnap(aa)
        
        if not tu.empty:
            tu_obj_ge.addSnap(tu)
        
        if not tp.empty:
            tp_obj_ge.addSnap(tp)
        
        if not ta.empty:
            ta_obj_ge.addSnap(ta)
        
        if not cu.empty:
            cu_obj_ge.addSnap(cu)
        
        # Concatenate the result to the final output DataFrame
        output = pd.concat([output, filter_df])
        
        etime = time.time()
        logger.info(f"Completed processing for date {date} in {etime - stime:.2f} seconds.")
    
    # Save the dataframes from each GardenGrade instance
    logger.info("Saving dataframes from GardenGrade instances.")
    obj_ge.savedf()
    tu_obj_ge.savedf()
    tp_obj_ge.savedf()
    ta_obj_ge.savedf()
    cu_obj_ge.savedf()
    
    # Replace column names starting with 'C_A' with 'O_U'
    output.columns = output.columns.str.replace('^C_A', 'O_U', regex=True)
    logger.info("Completed column name replacements.")
    
    # Filter out rows with null FULL_COMPONENT_CODE and reset index
    output = output[~output.FULL_COMPONENT_CODE.isnull()].reset_index(drop=True)
    
    # Select specific columns and drop duplicates
    output = output[[
        'CATALOG_ITEM_ID', 'A_A_G_E_AP_MA1', 'A_A_G_E_AP_MA3', 
        'T_U_G_E_AP_MA1', 
        'T_A_G_E_AP_MA3', 'O_U_G_E_AP_MA3',
    ]]
    output.drop_duplicates(inplace=True)
    output = output.reset_index(drop=True)
    
    return output

def cr_feature_update(dates, running_df):
    # Initialize CmpRating instances
    logger.info("Creating CmpRating instances.")
    obj_cr = CmpRating()
    tu_obj_cr = CmpRating('T', 'U')
    tp_obj_cr = CmpRating('T', 'P')
    ta_obj_cr = CmpRating('T', 'A')
    cu_obj_cr = CmpRating('C', 'A')
    
    output = pd.DataFrame()
    
    # Iterate over each date in the list of dates
    for date in dates:
        logger.info(f"Processing date: {date}")
        stime = time.time()
        
        # Filter the running DataFrame for the current date and reset the index
        filter_df = running_df[running_df.DATE == date].reset_index(drop=True)
        combination = filter_df.CR.unique().tolist()
        logger.debug(f"Unique CR list for date {date}: {combination}")
        
        # Group and aggregate data for the current date
        logger.info(f"Aggregating data for date {date}.")
        aa = filter_df.groupby(["DATE", "CR"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tu = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Auction')].groupby(["DATE", "CR"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        tp = filter_df[(filter_df.Purchase_Flag == 'T') & (filter_df.Channel == 'Private')].groupby(["DATE", "CR"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        ta = filter_df[(filter_df.Purchase_Flag == 'T')].groupby(["DATE", "CR"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        cu = filter_df[(filter_df.Purchase_Flag == 'C') & (filter_df.Channel == 'Auction')].groupby(["DATE", "CR"]).agg({
            "PURCHASED_PRICE_Detail": 'mean',
            "PURCHASED_QUANTITY_Detail": 'sum',
            "INVOICE_WT": 'sum'
        }).reset_index()
        
        # List of objects and combination
        objects = [obj_cr, tu_obj_cr, tp_obj_cr, ta_obj_cr, cu_obj_cr]
        
        # Create a multiprocessing Pool with the number of available CPU cores
        logger.info(f"Starting multiprocessing pool for date {date}.")
        pool = Pool()
        
        # Apply parallel processing to get snapshot data for each object
        snapshots = pool.starmap(process_snap, [(obj, combination) for obj in objects])
        
        # Close the pool of processes and wait for all processes to complete
        pool.close()
        pool.join()
        
        # Merge the snapshot data obtained from different objects
        logger.info(f"Merging snapshots for date {date}.")
        temp6 = snapshots[0]
        for temp in snapshots[1:]:
            temp6 = temp6.merge(temp, on='CR', how='left')
        
        filter_df = filter_df.merge(temp6, on='CR', how='left')
        
        # Add snapshots to their respective objects if not empty
        if not aa.empty:
            obj_cr.addSnap(aa)
        
        if not tu.empty:
            tu_obj_cr.addSnap(tu)
        
        if not tp.empty:
            tp_obj_cr.addSnap(tp)
        
        if not ta.empty:
            ta_obj_cr.addSnap(ta)
        
        if not cu.empty:
            cu_obj_cr.addSnap(cu)
        
        # Concatenate the result to the final output DataFrame
        output = pd.concat([output, filter_df])
        
        etime = time.time()
        logger.info(f"Completed processing for date {date} in {etime - stime:.2f} seconds.")
    
    # Save the dataframes from each CmpRating instance
    logger.info("Saving dataframes from CmpRating instances.")
    obj_cr.savedf()
    tu_obj_cr.savedf()
    tp_obj_cr.savedf()
    ta_obj_cr.savedf()
    cu_obj_cr.savedf()
    
    # Filter out rows with null FULL_COMPONENT_CODE and reset index
    output = output[~output.FULL_COMPONENT_CODE.isnull()].reset_index(drop=True)
    output.columns = output.columns.str.replace('^C_A', 'O_U', regex=True)
    # Select specific columns and drop duplicates
    output = output[[
        'CATALOG_ITEM_ID','A_A_C_R_AP_MA3', 
         'T_P_C_R_AP_MA3', 'T_P_C_R_AP_MA5', 'T_A_C_R_AP_MA1', 
        'T_A_C_R_AP_MA3', 
    ]]
    output.drop_duplicates(inplace=True)
    output = output.reset_index(drop=True)
    
    return output
       
def missing_value_imputation(col,model_input_df1,final_merged_df,num_bins = 30):
    # Group by 'col' and calculate mean
    temp = model_input_df1[[col, 'PURCHASED_PRICE_Detail']]
    percentiles = temp[col].quantile([0.01, 0.99]).values
    temp[col] = np.clip(temp[col], percentiles[0], percentiles[1])
    temp = temp.groupby(col, dropna=False)['PURCHASED_PRICE_Detail'].mean().reset_index()

    # Find min and max values of 'col'
    min_ = temp[col].min()
    max_ = temp[col].max()
    
    # Calculate bin width
    bin_width = (max_ - min_) / num_bins
    
    # Find the mean of 'PURCHASED_PRICE_Detail' for missing values
    mis_val_prc_avg = final_merged_df[final_merged_df[col].isnull()]['PURCHASED_PRICE_Detail'].min()
    
    # Remove rows with NaN values in 'col'
    temp = temp.dropna(subset=[col]).reset_index(drop=True)
    
    # Create bin DataFrame
    bin_df = pd.DataFrame({
        col: [min_ + i * bin_width for i in range(num_bins)],
        'Bin': range(num_bins)
    })
    
    # Merge 'temp' with 'bin_df' based on 'col'
    temp = pd.merge_asof(temp, bin_df, on=col)
    
    # Group by 'Bin' and calculate mean
    piv_temp = temp.groupby('Bin').agg({'PURCHASED_PRICE_Detail': 'mean', col: 'mean'}).reset_index()
    
    # Calculate absolute difference
    piv_temp['Diff'] = abs(piv_temp['PURCHASED_PRICE_Detail'] - mis_val_prc_avg)
    
    # Find the value with minimum difference
    out = piv_temp.loc[piv_temp['Diff'].idxmin(), col]
    
    return out     

def model_input_preprocess(current_df, model_input_df_prvs):
    logger.info("Starting model input preprocessing.")
    
    # Read the cluster mapping and grade encoding files
    logger.info("Reading cluster mapping and grade encoding files.")
    cluster_mapping = pd.read_csv(GLOBAL_PATHS["CLUSTER_FILE"])
    grade_encode = pd.read_csv(GLOBAL_PATHS["GRADE_ENCODE_FILE"])
    
    # Load component quadrants
    smooth = COMPONENT_QUADRANTS['smooth']
    intermittent = COMPONENT_QUADRANTS['intermittent']
    lumpy = COMPONENT_QUADRANTS['lumpy']

    def get_quadrant(component):
        """
        Mapping the components to its appropriate quadrant
        """
        if component in smooth:
            return 'Smooth'
        elif component in intermittent:
            return 'Intermittent'
        elif component in lumpy:
            return 'Lumpy'
        else:
            return 'Lumpy'
        
    def train_flag_check(component, price):
        """
        Creating TCPL Operating Flag
        """
        try:
            if price <= component_dict[component]:
                return True
            else:
                return False
        except KeyError:
            return False
    
    logger.info("Merging cluster mapping with current dataframe.")
    current_df = current_df.merge(cluster_mapping, on='FULL_COMPONENT_CODE', how='left')
    
    # Get unique list of components
    comp_list = current_df.FULL_COMPONENT_CODE.unique().tolist()
    
    # Define run date and filter last 52 weeks of data
    run_date = current_df.DATE.min()
    logger.info(f"Run date: {run_date}")
    last_52_weeks_data = model_input_df_prvs[
        (model_input_df_prvs.DATE >= pd.to_datetime(run_date) - pd.to_timedelta('366 days')) &
        (model_input_df_prvs.DATE <= run_date)
    ].reset_index(drop=True)
    
    # Create component dictionary with price bands
    component_dict = {}
    band = 5
    logger.info("Creating component dictionary with price bands.")
    for component in comp_list:
        max_price = last_52_weeks_data[
            (last_52_weeks_data.Purchase_Flag == 'T') &
            (last_52_weeks_data.FULL_COMPONENT_CODE == component)
        ].PURCHASED_PRICE_Detail.max()
        component_dict[component] = max_price + band

    # Apply train flag check
    logger.info("Applying train flag check.")
    current_df['Train_flag'] = current_df[['FULL_COMPONENT_CODE', "PURCHASED_PRICE_Detail"]].apply(
        lambda x: train_flag_check(x[0], x[1]), axis=1
    )
    
    # Apply quadrant mapping
    logger.info("Applying quadrant mapping.")
    current_df['Quadrant'] = current_df['FULL_COMPONENT_CODE'].apply(get_quadrant)
    
    # Create key for merging with grade encoding
    current_df['key'] = current_df["FULL_COMPONENT_CODE"] + "_" + current_df["GRADE_CODE"]
    
    # Merge with grade encoding
    logger.info("Merging with grade encoding.")
    current_df = current_df.merge(grade_encode, on='key', how='left')
    
    # Ensure the columns match the previous model input dataframe
    current_df = current_df[['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
       'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE',
       'INVOICE_WT', 'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail',
       'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag', 'Purchases_Value',
                             "A_U_B_C_PD_MA2","T_P_A_C_PD_MA20",
                             'A_A_G_E_AP_MA1', 'A_A_G_E_AP_MA3', 
                             'A_A_G_C_AP_MA2', 'T_U_G_C_PD_MA5','T_A_G_C_AP_MA1',
                             'O_U_G_C_PD_MA2',  'T_P_G_C_PD_MA5',
                             'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10','T_U_G_E_AP_MA1', 
                            'T_A_G_E_AP_MA3', 'O_U_G_E_AP_MA3','A_A_C_R_AP_MA3', 
                             'T_P_C_R_AP_MA3', 'T_P_C_R_AP_MA5', 'T_A_C_R_AP_MA1', 
                            'T_A_C_R_AP_MA3',            
       'Cluster', 'QUALITY_VARIANCE',
       'Train_flag', 'Quadrant', 'GRADE_CODE', 'key', 'Code']].drop_duplicates()
    
    
    # Handle missing cluster mappings
    logger.info("Handling missing cluster mappings.")
    current_df["Cluster"] = np.where(current_df["Cluster"].isnull(), 'No Cluster Mapping', current_df["Cluster"])
    model_input_df_prvs = model_input_df_prvs[['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
       'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE',
       'INVOICE_WT', 'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail',
       'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag', 'Purchases_Value',
                             "A_U_B_C_PD_MA2","T_P_A_C_PD_MA20",
                             'A_A_G_E_AP_MA1', 'A_A_G_E_AP_MA3', 
                             'A_A_G_C_AP_MA2', 'T_U_G_C_PD_MA5','T_A_G_C_AP_MA1',
                             'O_U_G_C_PD_MA2',  'T_P_G_C_PD_MA5',
                             'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10','T_U_G_E_AP_MA1', 
                            'T_A_G_E_AP_MA3', 'O_U_G_E_AP_MA3','A_A_C_R_AP_MA3', 
                             'T_P_C_R_AP_MA3', 'T_P_C_R_AP_MA5', 'T_A_C_R_AP_MA1', 
                            'T_A_C_R_AP_MA3',            
       'Cluster', 'QUALITY_VARIANCE',
       'Train_flag', 'Quadrant', 'GRADE_CODE', 'key', 'Code']]
    # Concatenate current and previous model input dataframes
    logger.info("Concatenating current and previous model input dataframes.")
    final_model_input = pd.concat([current_df, model_input_df_prvs])
    
    # Clean up and sort the final dataframe
    del current_df, model_input_df_prvs
    final_model_input.reset_index(drop=True, inplace=True)
    final_model_input.sort_values(by='DATE', inplace=True)
    
    logger.info("Model input preprocessing completed.")
    return final_model_input


def dataSlice(df, date):
    return df[df.DATE<date]

def main():
    logger.info("Starting main process.")
    
    # get the updated master file and previous Model Input file from S3
    try:
        logger.info("Fetching master and previous model input data from S3.")
        master = get_master_data()
        model_input_prev = get_model_train_data()
    except Exception as e:
        logger.error("Error getting file from S3: " + str(e))
        raise S3ReadError(e)

    # Filter the master dataframe for the past 365 days
    df = master[master["DATE"] >= master.DATE.max() - pd.to_timedelta('365 days')][[
        'DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
        'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE', 'INVOICE_WT',
        'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail',
        'Channel', 'Purchase_Flag', 'Purchases_Value', 'GRADE_CODE', 'QUALITY_VARIANCE'
    ]]
    
    # Group and aggregate the dataframe
    df = (df.groupby([
        'DATE', 'WeekNum', 'Year', 'Purchase_Flag', 'Channel',
        'BUYING_CENTER_CODE', 'FULL_COMPONENT_CODE'
    ]).agg({
        'INVOICE_WT': 'sum', 
        'PURCHASED_QUANTITY_Detail': 'sum', 
        'PURCHASED_PRICE_Detail': 'mean',
        'Purchases_Value': 'sum', 
        'CATALOG_ITEM_ID': 'count'
    }).sort_values(by='DATE', ascending=True)).reset_index()
    
    # Get the last update date from the previous model input dataframe
    last_update_date = model_input_prev['DATE'].max()
    logger.info(f"Last Updated Date : {last_update_date}")
    logger.info(f" Current Master Max Date : {master.DATE.max()}")
    # Filter the master dataframe for new data beyond the last update date
    running_df = master[(master["DATE"] > last_update_date)][[
        'DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
        'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE', 'INVOICE_WT',
        'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag',
        'Purchases_Value', 'GRADE_CODE', 'QUALITY_VARIANCE'
    ]]
    logger.info(f"Number of records needs to be updated : {running_df.shape}")
    
    if running_df.shape[0]<1:
        logger.info(f"No records to update...\nStopping Model Input Update Process...")
        return 
    # Convert QUALITY_VARIANCE to numerical values
    running_df['QUALITY_VARIANCE'] = np.where(running_df['QUALITY_VARIANCE'] == '+', 1,
                                              np.where(running_df['QUALITY_VARIANCE'] == '=', 0, -1))
    running_df.reset_index(drop=True, inplace=True)
    
    # Create GC, CR, GE columns for feature generation
    running_df['GC'] = running_df['GARDEN_NAME'] + "_" + running_df['FULL_COMPONENT_CODE']
    running_df['GE'] = running_df['GARDEN_NAME'] + "_" + running_df['GRADE_CODE']
    running_df['CR'] = running_df['FULL_COMPONENT_CODE'] + "_" + running_df['QUALITY_VARIANCE'].astype(str)
    
    # Prepare test dataframe
    test_df = running_df[[
        'DATE', 'CATALOG_ITEM_ID', 'INVOICE_WT', 'FULL_COMPONENT_CODE', 'BUYING_CENTER_CODE', 
        'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag', 'GRADE_CODE', 'QUALITY_VARIANCE'
    ]]
    test_df = test_df.sort_values(by='DATE', ascending=True)
    dates = test_df.DATE.astype(str).unique().tolist()
    
    del master  # Free up memory
    
    # Replace Channel and Purchase_Flag values
    df['Channel'] = df['Channel'].str.replace('Auction', 'U')
    df['Channel'] = df['Channel'].str.replace('Private', 'P')
    df['Purchase_Flag'] = df['Purchase_Flag'].str.replace("C", 'O')
    
    bc_output = pd.DataFrame()

    logger.info("Generating Buying Center features.")
    
    # Iterate over each date in the list of dates
    logger.info(f"list of dates : {dates}")
     
    for date in dates:
        stime = time.time()
        temp = pd.DataFrame()
        filter_df = test_df[test_df.DATE == date].reset_index(drop=True)
        filter_df['b'] = filter_df['BUYING_CENTER_CODE'] + '_' + filter_df['FULL_COMPONENT_CODE']
        df1 = dataSlice(df,date)
        # Iterate over each unique combination in the 'b' column
        for _combination in filter_df.b.astype(str).unique():
            if _combination == '<NA>':
                continue
            
            buying_center, component = _combination.split('_')
            feature_out = buying_center_feat(date, buying_center, component, df1, BC_FEATURES)
            feature_out['b'] = _combination
            temp = pd.concat([temp, feature_out])
            
        etime = time.time()
        logger.info(f"Completed BC featrues for date {date} in {etime - stime:.2f} seconds.")
        
        merge_df = filter_df.merge(temp, on='b', how='left')
        bc_output = pd.concat([bc_output, merge_df])
    bc_output = bc_output[[
        'CATALOG_ITEM_ID',"A_U_B_C_PD_MA2","T_P_A_C_PD_MA20"
    ]]
    bc_output.drop_duplicates(inplace=True)
    bc_output.reset_index(drop=True, inplace=True)
    
    running_df = running_df.sort_values(by='DATE', ascending=True)
    dates = running_df.DATE.astype(str).unique().tolist()
    logger.info(running_df.DATE.astype(str).unique().tolist())

    
    logger.info("Generating GC features Started.")
    gc_output = gc_feature_update(dates, running_df)
    logger.info("Generating GC features completed.")
    
    logger.info("Generating GE, features Started.")
    ge_output = ge_feature_update(dates, running_df)
    logger.info("Generating GE, features completed.")
    
    logger.info("Generating CR features Started.")
    cr_output = cr_feature_update(dates, running_df)
    logger.info("Generating CR features completed.")
    

    # Merge feature sets
    merged_df = gc_output.merge(bc_output, on='CATALOG_ITEM_ID', how='left')
    merged_df2 = merged_df.merge(ge_output, on='CATALOG_ITEM_ID', how='left')
    final_merged_df = merged_df2.merge(cr_output, on='CATALOG_ITEM_ID', how='left')
    
    del merged_df,merged_df2,gc_output,bc_output,ge_output,cr_output,running_df
    # Columns to fill with missing value imputation
    col_to_fill = [
        'A_A_G_C_AP_MA2', 'T_U_G_C_PD_MA5','T_A_G_C_AP_MA1',
        'O_U_G_C_PD_MA2',  'T_P_G_C_PD_MA5',
         'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10',
        "A_U_B_C_PD_MA2","T_P_A_C_PD_MA20"
    ]

    logger.info("Performing missing value imputation.")
    
    last_90_days_data = model_input_prev[
        model_input_prev.DATE >= model_input_prev.DATE.max() - pd.to_timedelta('90 days')
    ].reset_index()
    
    
    for col in col_to_fill:
        if final_merged_df[col].isnull().sum() > 0:
            final_merged_df[col] = final_merged_df[col].fillna(
                missing_value_imputation(col, last_90_days_data,final_merged_df)
            )

    # Preprocess the final merged dataframe
    logger.info("Preprocessing the final merged dataframe.")
    logger.info(final_merged_df.columns)
    final_merged_df = model_input_preprocess(final_merged_df, model_input_prev)
    
    logger.info(final_merged_df.columns)
    
    # Save the final dataframe to S3 as parquet
    logger.info("Saving the final dataframe to S3.")

    final_merged_df.drop_duplicates(inplace = True)
    final_merged_df.reset_index(drop = True, inplace = True)
    final_merged_df.to_parquet("s3://tcpl-buyingdecision/bgdn-pre-prod/model-taining-data/ML_Model_Input_v1_2.snappy.parquet")
    logger.info("Main process completed successfully.")

if __name__=='__main__':
    main()