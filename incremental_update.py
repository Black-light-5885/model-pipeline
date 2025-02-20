# Buying Decision Support Project
# Team@Stratlytics
# @author : Bala.
# File Information: This file is to get the delta data from S3 and update the master data accordingly. 

# -----------------------------Imports-----------------------------------
import pandas as pd
pd.options.mode.chained_assignment = None
# import awswrangler as wr
import numpy as np 
import os
import sys
import logging
from pathlib import Path
import boto3
import subprocess
import sys
import datetime
# --------------------  Config  --------------------
# Install required packages with compatible versions
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install","openpyxl"])
    print("Installing packages successfully done.")
except Exception as e:
    print(f"Error installing packages. {e}")



S3_ROOT_PATH ="s3://tcpl-buyingdecision/buying-decision-data"

S3_ROOT_PATH2 ="bgdn-pre-prod"
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
RUN_DATE = '2024-05-24 00:00:00'

TCPL_GUARDRAIL = False

REQUIRED_COLS= { 
                        "BUYING_TYPE_MASTER":[] ,
                        "FULL_COMPONENT_MASTER":[
                                                    'FULL_COMPONENT_ID', 'FULL_COMPONENT_CODE'
                                                ],
                        "GRADE_CODE_MASTER":  [
                                                    'GRADE_ID', 'GRADE_CODE', 'LEAF_TYPE'
                                                ],
                        "BUYING_CENTER_MASTER":[
                                                    'BUYING_CENTER_ID', 'BUYING_CENTER_CODE', 'BUYING_CENTER_NAME'
                                                ],
                        "GARDEN_MASTER":[
                                            'GARDEN_ID','GARDEN_NAME', 'GARDEN_BLOCKED'
                                        ],
                        "ITEM_COMPETITOR":['COMPETITOR_ID',
                                                'PURCHASED_QUANTITY', 'CATALOG_ITEM_ID'
                                                ],
                        "CATALOG_DETAIL":['CATALOG_ITEM_ID','CATALOG_HEADER_ID','GARDEN_ID','GRADE_ID','FINAL_COMPONENT_ID','BLOCKED','SOLD_INDICATOR','WITHDRAWN_INDICATOR',
                                            'LOT_NUMBER', 'E_AUCTION_LOT_NUM','INVOICE_DATE', 'OFFER_DATE','SALE_DATE','INVOICE_WT','OFFER_PRICE','PURCHASED_QUANTITY','PURCHASED_PRICE', 
                                        'TEA_SUB_TYPE',"QUALITY_VARIANCE","CONFIRMEDBIDDATE",'NUM_OF_PURCHASED_BAGS','SOLD_PRICE'
                                        ],
                        "CATALOG_HEADER": ['CATALOG_HEADER_ID','BUYING_CENTER_ID','BUYING_TYPE_ID', 'BUYING_CHANNEL_ID','FINANCIAL_YEAR'],
                       'COMPETITION_MASTER': ['COMPETITOR_ID', 'COMPETITOR_NAME','IS_PRIVATEAGENT']
                        
                        }

BUCKET_NAME = "tcpl-buyingdecision"

# ------- Functions-------------------------------
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


# Initialize logger
logging = get_module_logger("Incremental_Date_Update")



class BadInputFile(Exception):  # 400
    def __init__(
        self,
        status_msg="Invalid Data Received",
        category="error",
    ):
        logging.info(f"BadInputFile {category} - {status_msg}")
        
class NoInputFile(Exception):
    def __init__(self,path):
        logging.error(f"No input file in the provided path :{path}")


def get_incremental_data(table = 'CATALOG_DETAIL')->pd.DataFrame:
    """ Read the incremental data from the S3 bucket for the given table name.

    Args:
        table (str): ['CATALOG_DETAIL','CATALOG_HEADER','ITEM_COMPETITOR']. Defaults to 'CATALOG_DETAIL'.

    Returns:
        incremental_data (pd.DataFrame) : Pandas dataframe that contains incremental data for the given table name.
    """
    
    try:
        logging.info(f"Pipeline_Service__data_fetch.py get_incremental_data(table = {table}) : started...")
        # incremental_data = wr.s3.read_parquet(GLOBAL_PATHS['INC_DATA_ROOT_PATH'][table])
        incremental_data = get_file_from_s3(GLOBAL_PATHS['INC_DATA_ROOT_PATH'][table])
        
        # Selecting only the required columns 
        incremental_data= incremental_data[REQUIRED_COLS[table]]
        
        
    except Exception as e:
        logging.error(f"Pipeline_Service__data_fetch.py get_incremental_data(table = {table}) : Failed to read incremental data from S3: %s", e)
        raise BadInputFile('path')
        
    else:
        logging.info(f"Pipeline_Service__data_fetch.py get_incremental_data(table = {table}) : finished...")
        return incremental_data
    

def get_dimension_data(table = 'BUYING_TYPE_MASTER', type_='parquet')->pd.DataFrame:
    """ Read the dimension master data from the S3 bucket for the given table name.

    Args:
        table (str): Dimension Table Name.  Defaults to 'BUYING_TYPE_MASTER'.
                        Options -> ['BUYING_TYPE_MASTER', 'FULL_COMPONENT_MASTER', 
                                    'GRADE_CODE_MASTER', 'BUYING_CENTER_MASTER', 
                                        'GARDEN_MASTER', 'COMPETITION_MASTER']

    Returns:
        dimension_data (pd.DataFrame) : Pandas dataframe that contains dimension data for the given table name.
    """
    
    try:
        logging.info(f"Pipeline_Service__data_fetch.py get_dimension_data(table = {table}) : started...")
        if type_ =='parquet':
            # dimension_data = wr.s3.read_parquet(GLOBAL_PATHS['DIMENSION_TABLE_PATH'][table])
            dimension_data = pd.read_parquet(GLOBAL_PATHS['DIMENSION_TABLE_PATH'][table])

        elif type_ =='csv':
            # dimension_data = wr.s3.read_csv(GLOBAL_PATHS['DIMENSION_TABLE_PATH'][table])
            dimension_data = pd.read_csv(GLOBAL_PATHS['DIMENSION_TABLE_PATH'][table])

        else:
            raise BadInputFile(
                status_msg="The input file does not contain any records"
                )
            
        
        # Selecting only the required columns 
        dimension_data= dimension_data[REQUIRED_COLS[table]]
        pass
    except Exception as e:
        logging.error(f"Pipeline_Service__data_fetch.py get_dimension_data(table = {table}) : Failed to read dimension data from S3: %s", e)
        pass
    else:
        logging.info(f"Pipeline_Service__data_fetch.py get_dimension_data(table = {table}) : finished...")
        return dimension_data
    pass


def get_master_data()->pd.DataFrame:
    """
    Methode to get the historical master data from the S3 bucket.
    

    Returns:
        hist_data (pd.DataFrame) : Pandas dataframe that contains historical data for sold information.
    """
    try:
        logging.info(f"Pipeline_Service__data_fetch.py__get_historical_data() : started...")
        # hist_data = wr.s3.read_parquet(GLOBAL_PATHS["MASTER_FILE_LOCATION"])
        hist_data = pd.read_parquet(GLOBAL_PATHS["MASTER_FILE_LOCATION"])

        print(hist_data.shape[0])
        hist_data = hist_data[hist_data['DATE']<RUN_DATE].reset_index(drop = True)
        print(hist_data.shape[0])
        
        logging.info(f"Number of historical data fetched from S3: %d", hist_data.shape[0])
        
    except Exception as e:
        logging.error(f"Pipeline_Service__data_fetch.py__get_historical_data(): Failed to read historical data from S3: %s", e)
        
    else:
        logging.info(f"Pipeline_Service__data_fetch.py__get_historical_data() : finished...")
        
        return hist_data 
    
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
        logging.info(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : started...")
        
        # garden_std_data = wr.s3.read_excel(GLOBAL_PATHS["STD_GARDEN_FILE"])
        garden_std_data = pd.read_excel(GLOBAL_PATHS["STD_GARDEN_FILE"])

        
        # Garden standardization file may contain duplicates due to the multiple GARDEN ID mappings.
        # To map garden with their unique Standard Name selecting two columns from the data and removing duplicates if any.
        garden_std_data = garden_std_data[['GARDEN_ID',"GARDEN_CODE","Standardized Garden"]].drop_duplicates().reset_index(drop = True) 
        
        logging.info(f"Number of Garden mapping data fetched from S3: %d", garden_std_data.shape[0])
        
    except Exception as e:
        logging.error(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : Failed to read Garden Standard Name mapping data from S3: %s", e)
        
    else:
        logging.info(f"Pipeline_Service__data_fetch.py__get_garden_std_data() : : finished...")
        
        return garden_std_data 

def private_agent_mapping(cat_details: pd.DataFrame, header: pd.DataFrame) -> pd.DataFrame:
    """Maps private agents' purchases from the private channel to the auction channel.

    Args:
        cat_details (pd.DataFrame): Catalog details dataframe.
        header (pd.DataFrame): Header information dataframe.

    Returns:
        pd.DataFrame: Mapped private agents' purchases dataframe.
    """
    
    logging.info("Starting private agent mapping process")
    
    # Merge the details table with the Header Table to get the Buying Type information.
    logging.info("Merging catalog details with header to get buying type information")
    merge_df = cat_details.merge(header, on='CATALOG_HEADER_ID', how='left', indicator=True)
    del cat_details, header
    
    # Filter the Auction Records.
    logging.info("Filtering auction records")
    auction_df = merge_df[merge_df.BUYING_TYPE_ID == 1]
    
    # Creating Week Number and Year columns to create a composite key.
    logging.info("Creating week number and year columns for auction records")
    auction_df['week_num'] = auction_df.SALE_DATE.dt.isocalendar()['week']
    auction_df['Year'] = auction_df.SALE_DATE.dt.isocalendar()['year']
    
    # Create composite key to map private channel records to Auction Channel records.
    logging.info("Creating composite key for auction records")
    auction_df['key'] = auction_df.E_AUCTION_LOT_NUM + '_' + auction_df.BUYING_CENTER_ID.astype(str) + '_' + \
                        auction_df.week_num.astype(str) + '_' + auction_df.Year.astype(str) + '_' + auction_df.GARDEN_ID.astype(str) + \
                        '_' + auction_df.FINAL_COMPONENT_ID.astype(str) + '_' + auction_df.GRADE_ID.astype(str)
                            
    # Since both tables have the same column names, rename the Auction channel columns to differentiate them.
    logging.info("Renaming auction channel columns to differentiate them")
    auction_df.rename(columns={
        'CATALOG_ITEM_ID': 'CATALOG_ITEM_ID_Auction',
        'CATALOG_HEADER_ID': 'CATALOG_HEADER_ID_Auction',
        "INVOICE_WT": 'INVOICE_WT_Auction',
        'SALE_DATE': 'SALE_DATE_Auction',
        'PURCHASED_QUANTITY': 'PURCHASED_QUANTITY_Auction',
        'PURCHASED_PRICE': 'PURCHASED_PRICE_Auction'
    }, inplace=True)
    
    # Filtering Private Channel Data
    logging.info("Filtering private channel data")
    pvt_agent_df = merge_df[merge_df.BUYING_CHANNEL_ID == 5]
    
    # Creating Week Number and Year columns to create a composite key.
    logging.info("Creating week number and year columns for private channel records")
    pvt_agent_df['week_num'] = pvt_agent_df.OFFER_DATE.dt.isocalendar()['week']
    pvt_agent_df['Year'] = pvt_agent_df.OFFER_DATE.dt.isocalendar()['year']
    
    # Create composite key for private channel records.
    logging.info("Creating composite key for private channel records")
    pvt_agent_df['key'] = pvt_agent_df.LOT_NUMBER + '_' + pvt_agent_df.BUYING_CENTER_ID.astype(str) + '_' + \
                          pvt_agent_df.week_num.astype(str) + '_' + pvt_agent_df.Year.astype(str) + '_' + \
                          pvt_agent_df.GARDEN_ID.astype(str) + '_' + pvt_agent_df.FINAL_COMPONENT_ID.astype(str) + '_' + pvt_agent_df.GRADE_ID.astype(str)
                            
    # Taking unique keys from Private Channel.
    logging.info("Extracting unique keys from private channel")
    pvt_key_list = pvt_agent_df['key'].unique().tolist()
    
    # For the private composite keys, filter the auction records.
    logging.info("Filtering auction records using private channel keys")
    filter_df = auction_df[((auction_df.key.isin(pvt_key_list))) & (auction_df.PURCHASED_PRICE_Auction > 0)]
    
    # Selecting relevant columns from the filtered auction records.
    filter_df = filter_df[['key', 'CATALOG_ITEM_ID_Auction', 'CATALOG_HEADER_ID_Auction',
                           'SALE_DATE_Auction', 'INVOICE_WT_Auction',
                           'PURCHASED_QUANTITY_Auction', 'PURCHASED_PRICE_Auction']]
    
    # Dropping the merge indicator column from private agent dataframe.
    logging.info("Dropping merge indicator column from private agent dataframe")
    pvt_agent_df.drop('_merge', axis=1, inplace=True)
    
    # Merging private agent data with filtered auction data.
    logging.info("Merging private agent data with filtered auction data")
    merge = pvt_agent_df.merge(filter_df, on='key', how='left', indicator=True)
    
    # Identifying duplicated keys.
    logging.info("Identifying duplicated keys in merged dataframe")
    duplicated_keys_list = merge[merge.duplicated('key')].key.unique().tolist()
    
    # Filtering out records with both matched and non-duplicated keys.
    logging.info("Filtering out records with both matched and non-duplicated keys")
    pvt_agnt_and_auct_mapping = merge[(merge._merge == 'both') & ~(merge.key.isin(duplicated_keys_list))].reset_index(drop=True)
    
    logging.info("Private agent mapping process completed")
    return pvt_agnt_and_auct_mapping

def master_segmentation(master_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting master segmentation process")
    
    # Only TCPL Purchase
    logging.info("Segmenting TCPL only purchases")
    tcpl_purchase = master_df[(master_df.PURCHASED_QUANTITY_Detail > 0) & (master_df.PURCHASED_QUANTITY.isnull())]
    tcpl_purchase['Purchase_Flag'] = 'T'  # Marking as TCPL purchase
    tcpl_purchase.drop(['COMPETITOR_ID', 'PURCHASED_QUANTITY', 'COMPETITOR_NAME'], axis=1, inplace=True)
    
    # TCPL And Competition Split Purchase
    logging.info("Segmenting TCPL and Competition split purchases")
    tcpl_and_comp_split = master_df[(master_df.PURCHASED_QUANTITY_Detail > 0) & (master_df.PURCHASED_QUANTITY > 0)]
    
    # Creating Separate records for TCPL  
    tcpl_split_purchase = tcpl_and_comp_split.copy()
    tcpl_split_purchase.drop(['COMPETITOR_ID', 'PURCHASED_QUANTITY', 'COMPETITOR_NAME'], axis=1, inplace=True)
    tcpl_split_purchase.drop_duplicates(inplace=True)
    tcpl_split_purchase['Purchase_Flag'] = 'T'  # Marking as TCPL purchase
    tcpl_split_purchase['Split_Flag'] = 'Y'  # Indicating split purchase
    
    # Creating Separate records for Competitor
    logging.info("Creating separate records for competitor in split purchases")
    competitor_split_purchase = tcpl_and_comp_split.copy()
    competitor_split_purchase.drop('PURCHASED_QUANTITY_Detail', axis=1, inplace=True)
    competitor_split_purchase.rename(columns={'PURCHASED_QUANTITY': 'PURCHASED_QUANTITY_Detail'}, inplace=True)
    competitor_split_purchase['Purchase_Flag'] = 'C'  # Marking as competitor purchase
    competitor_split_purchase['Split_Flag'] = 'Y'  # Indicating split purchase
    
    # Competitor Only Purchase 
    logging.info("Segmenting competitor only purchases")
    competitor_purchase = master_df[(master_df.PURCHASED_QUANTITY_Detail.isnull()) & (master_df.PURCHASED_QUANTITY > 0)]
    competitor_purchase.drop('PURCHASED_QUANTITY_Detail', axis=1, inplace=True)
    competitor_purchase.rename(columns={'PURCHASED_QUANTITY': 'PURCHASED_QUANTITY_Detail'}, inplace=True)
    competitor_purchase['Purchase_Flag'] = 'C'  # Marking as competitor purchase
    
    # Not Sold to anyone
    logging.info("Segmenting unsold items")
    not_sold = master_df[(master_df.PURCHASED_QUANTITY_Detail.isnull()) & (master_df.PURCHASED_QUANTITY.isnull())]
    not_sold.drop(['COMPETITOR_ID', 'PURCHASED_QUANTITY', 'COMPETITOR_NAME'], axis=1, inplace=True)
    not_sold['Purchase_Flag'] = 'Not Sold'  # Marking as not sold
    qty_not_avail = not_sold[not_sold.PURCHASED_PRICE_Detail > 0]  # Not sold but has a price detail
    
    logging.info(f"TCPL Purchases: {tcpl_purchase.shape[0]}, "
                 f"TCPL Split Purchases: {tcpl_split_purchase.shape[0]}, "
                 f"Competitor Purchases: {competitor_purchase.shape[0]}, "
                 f"Competitor Split Purchases: {competitor_split_purchase.shape[0]}, "
                 f"Not Sold: {not_sold.shape[0]}")
    
    # Concatenating all segmented dataframes
    logging.info("Concatenating all segmented dataframes")
    list_of_df = [tcpl_purchase, tcpl_split_purchase, competitor_purchase, competitor_split_purchase, not_sold]
    master_df = pd.concat(list_of_df)
    
    del tcpl_purchase, tcpl_split_purchase, competitor_purchase, competitor_split_purchase, not_sold
    # Ensuring datetime format for CONFIRMEDBIDDATE
    logging.info("Ensuring datetime format for CONFIRMEDBIDDATE")
    master_df.CONFIRMEDBIDDATE = master_df.CONFIRMEDBIDDATE.astype("datetime64[ns]")
    
    # Rearranging columns for consistency
    logging.info("Rearranging columns for consistency")
    col = [
        'CATALOG_ITEM_ID', 'CATALOG_HEADER_ID', 'GARDEN_ID', 'GRADE_ID',
        'BUYING_CENTER_ID', 'BUYING_TYPE_ID', 'FULL_COMPONENT_ID', 'BLOCKED',
        'SOLD_INDICATOR', 'LOT_NUMBER', 'E_AUCTION_LOT_NUM', 'WITHDRAWN_INDICATOR',
        'COMPETITOR_ID', 'COMPETITOR_NAME', 'INVOICE_DATE', 'OFFER_DATE', 'SALE_DATE',
        'CONFIRMEDBIDDATE', 'FINANCIAL_YEAR', 'INVOICE_WT', 'OFFER_PRICE',
        'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail', 'NUM_OF_PURCHASED_BAGS',
        'FULL_COMPONENT_CODE', 'QUALITY_VARIANCE', 'GRADE_CODE', 'LEAF_TYPE', 'TEA_SUB_TYPE',
        'BUYING_CENTER_CODE', 'BUYING_CENTER_NAME', 'GARDEN_NAME', 'Channel', 'Purchase_Flag', 'Split_Flag'
    ]
    master_df = master_df[col]
    
    logging.info("Master segmentation process completed")
    return master_df
# ---------------------
def main():
    # Read Dimensions Table from S3.
    try: 
        logging.info("Fetching dimension data from S3")
        buying_type_df = get_dimension_data(table='BUYING_TYPE_MASTER')
        full_comp_df = get_dimension_data(table='FULL_COMPONENT_MASTER')
        grade_df = get_dimension_data(table='GRADE_CODE_MASTER')
        master_buying_df = get_dimension_data(table='BUYING_CENTER_MASTER')
        master_garden_df = get_dimension_data(table='GARDEN_MASTER')
        garden_std = get_garden_std_data()
        comp_master = get_dimension_data(table='COMPETITION_MASTER', type_='csv')
        master_his = get_master_data()
        logging.info("Dimension data fetched successfully")
    except Exception as e:
        logging.error(f"Error while getting the dimension data from the S3: {e}")
        return

    # Read Incremental data from S3.
    try:
        logging.info("Fetching incremental data from S3")
        cat_details = get_incremental_data(table='CATALOG_DETAIL')
        header = get_incremental_data(table='CATALOG_HEADER')
        # pvt_data_details = get_incremental_data(table='CATALOG_DETAIL')
        # pvt_data_header = get_incremental_data(table='CATALOG_HEADER')
        competitor = get_incremental_data(table='ITEM_COMPETITOR')
        logging.info("Incremental data fetched successfully")
    except Exception as e:
        logging.error(f"Error while getting the incremental data from the S3: {e}")
        return

    # # Condition for TCPL Purchase lots.  
    # logging.info("Updating PURCHASED_QUANTITY based on NUM_OF_PURCHASED_BAGS")
    # cat_details["PURCHASED_QUANTITY"] = np.where(
    #     ((cat_details.NUM_OF_PURCHASED_BAGS > 0) and cat_details.),
    #     cat_details["PURCHASED_QUANTITY"],
    #     np.NaN
    # )
    logging.info(f"Number of TCPL Purchases between {cat_details.SALE_DATE.min()} to {cat_details.SALE_DATE.max()}: {cat_details[cat_details['PURCHASED_QUANTITY'] > 0].shape[0]}")

    # Update Purchased Price where it is missing.
    logging.info("Updating PURCHASED_PRICE where it is missing")
    cat_details['PURCHASED_PRICE'] = np.where(
        cat_details['PURCHASED_PRICE'].isnull(),
        cat_details['SOLD_PRICE'],
        cat_details['PURCHASED_PRICE']
    )

    # # Concatenate Private and Auction detail/header tables into a single DataFrame.  
    # logging.info("Concatenating Private and Auction detail/header tables into a single DataFrame")
    # cat_details = pd.concat([auc_cat_details, pvt_data_details])
    # header = pd.concat([auc_data_header, pvt_data_header])

    # For better memory consumption, deleting the redundant DataFrames from memory.
    # logging.info("Deleting redundant DataFrames from memory")
    # del auc_cat_details, pvt_data_details, auc_data_header, pvt_data_header

    # Remove TCPL records from the competition table using the TCPL_COMPT_IDs.  
    logging.info("Removing TCPL records from the competition table using TCPL_COMPT_IDs")
    competitor = competitor[~competitor.COMPETITOR_ID.isin(TCPL_COMPT_ID)].reset_index(drop=True)

    # Removing duplicates from the competitor table.
    logging.info(f"Number of duplicates removed from Competitor Table: {competitor[competitor.duplicated('CATALOG_ITEM_ID')].shape[0]}")
    competitor.drop_duplicates(inplace=True)

    # Standard Garden Name Mapping. 
    logging.info("Standard Garden Name Mapping")
    garden = master_garden_df.merge(garden_std, how='left', indicator=True)
    garden['GARDEN_NAME'] = np.where(
        garden['Standardized Garden'].astype(str) != 'nan',
        garden['Standardized Garden'],
        garden['GARDEN_NAME']
    )
    logging.warning(f"Number of Gardens missing standardized Garden Name: {garden[garden['Standardized Garden'].astype(str) == 'nan'].shape[0]}")

    # Getting Private Agent Mapping 
    logging.info("Getting Private Agent Mapping")
    pvt_agnt_and_auct_mapping = private_agent_mapping(cat_details, header)

    # Renaming columns for consistency
    logging.info("Renaming columns for consistency")
    cat_details.rename(columns={
        'PURCHASED_QUANTITY': 'PURCHASED_QUANTITY_Detail',
        'PURCHASED_PRICE': 'PURCHASED_PRICE_Detail'
    }, inplace=True)

    # Extracting private agent IDs and competition details
    logging.info("Extracting private agent IDs and competition details")
    pvt_agnt_ids = comp_master[comp_master.IS_PRIVATEAGENT == 1].COMPETITOR_ID.unique().tolist()
    comp_detail = comp_master[comp_master.IS_PRIVATEAGENT == 0][['COMPETITOR_ID', 'COMPETITOR_NAME']].drop_duplicates()

    # Mapping auction data to private agent data
    logging.info("Mapping auction data to private agent data")
    auction_map = pvt_agnt_and_auct_mapping[['CATALOG_ITEM_ID_Auction', 'PURCHASED_QUANTITY']]
    auction_map.rename(columns={'PURCHASED_QUANTITY': 'PURCHASED_QUANTITY_PVT'}, inplace=True)

    # Filtering competitor data for private agents
    logging.info("Filtering competitor data for private agents")
    pvt_comp = competitor[competitor.COMPETITOR_ID.isin(pvt_agnt_ids)]
    competitor_mapping = pvt_comp.merge(
        auction_map,
        left_on='CATALOG_ITEM_ID', right_on='CATALOG_ITEM_ID_Auction',
        how='right', indicator=True
    )
    competitor_mapping['Flag'] = np.where(
        competitor_mapping.PURCHASED_QUANTITY == competitor_mapping.PURCHASED_QUANTITY_PVT,
        True,
        False
    )
    competitor_mapping = competitor_mapping[competitor_mapping.Flag == True]
    competitor_mapping['key'] = competitor_mapping.COMPETITOR_ID.astype(str) + '_' + competitor_mapping.CATALOG_ITEM_ID.astype(str)
    remove_comp_list = competitor_mapping['key'].unique().tolist()

    # Removing mapped competitors from the competition table
    logging.info("Removing mapped competitors from the competition table")
    competitor['key'] = competitor.COMPETITOR_ID.astype(str) + '_' + competitor.CATALOG_ITEM_ID.astype(str)
    competitor = competitor[~competitor.key.isin(remove_comp_list)].reset_index(drop=True)
    competitor = competitor.merge(comp_detail, on='COMPETITOR_ID', how='left')
    competitor.drop('key', axis=1, inplace=True)

    # Merging all dataframes into a single Master dataframe
    logging.info("Merging all dataframes into a single Master dataframe")
    merge1 = cat_details.merge(header, on='CATALOG_HEADER_ID', how='left')
    merge1 = merge1.merge(auction_map, left_on='CATALOG_ITEM_ID', right_on='CATALOG_ITEM_ID_Auction', how='left')
    merge1['PURCHASED_QUANTITY_Detail'] = np.where(
        (merge1.PURCHASED_QUANTITY_Detail.isnull()) & (merge1.PURCHASED_QUANTITY_PVT > 0),
        merge1.PURCHASED_QUANTITY_PVT,
        merge1.PURCHASED_QUANTITY_Detail
    )
    merge1.drop(['CATALOG_ITEM_ID_Auction', 'PURCHASED_QUANTITY_PVT'], axis=1, inplace=True)
    merge2 = merge1.merge(competitor, on='CATALOG_ITEM_ID', how='left')
    merge3 = merge2.merge(full_comp_df, left_on='FINAL_COMPONENT_ID', right_on='FULL_COMPONENT_ID', how='left')
    merge4 = merge3.merge(grade_df, on='GRADE_ID', how='left')
    merge5 = merge4.merge(master_buying_df, on='BUYING_CENTER_ID', how='left')
    master_df = merge5.merge(master_garden_df, on='GARDEN_ID', how='left')
    del merge5, merge4, merge3, merge2, merge1
    # Rearranging columns for consistency and removing unwanted data
    logging.info("Rearranging columns for consistency and removing unwanted data")
    arranged_colu = [
        'CATALOG_ITEM_ID', 'CATALOG_HEADER_ID', 'GARDEN_ID', 'GRADE_ID',
        'BUYING_CENTER_ID', 'BUYING_TYPE_ID', 'BUYING_CHANNEL_ID', 'COMPETITOR_ID', 'FULL_COMPONENT_ID',
        'BLOCKED', 'SOLD_INDICATOR', 'LOT_NUMBER', 'E_AUCTION_LOT_NUM',
        'WITHDRAWN_INDICATOR', 'INVOICE_DATE', 'OFFER_DATE', 'SALE_DATE', 'CONFIRMEDBIDDATE', 'FINANCIAL_YEAR',
        'INVOICE_WT', 'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail', 'PURCHASED_QUANTITY',
        'NUM_OF_PURCHASED_BAGS',
        'FULL_COMPONENT_CODE', 'QUALITY_VARIANCE', 'GRADE_CODE', 'LEAF_TYPE', 'TEA_SUB_TYPE',
        'BUYING_CENTER_CODE', 'BUYING_CENTER_NAME', 'GARDEN_NAME', 'COMPETITOR_NAME'
    ]

    master_df = master_df[arranged_colu]
    master_df = master_df[(master_df.BUYING_CHANNEL_ID != 5)]
    master_df['Channel'] = np.where((master_df.BUYING_TYPE_ID == 1), 'Auction', 'Private')

    # Segmenting master data
    logging.info("Segmenting master data")
    master_df = master_segmentation(master_df)
    master_df['Purchases_Value'] = master_df.PURCHASED_QUANTITY_Detail * master_df.PURCHASED_PRICE_Detail

    # Creating DATE and WeekNum columns
    logging.info(master_df.CONFIRMEDBIDDATE.unique())
    logging.info(master_df[(master_df.SALE_DATE.isnull())&(master_df.CONFIRMEDBIDDATE.isnull())].shape[0])
    logging.info(master_df[(master_df.SALE_DATE.isnull())].CONFIRMEDBIDDATE.unique())
    logging.info(master_df[(master_df.SALE_DATE.isnull())&(master_df.CONFIRMEDBIDDATE.isnull())].shape[0])
    
    logging.info("Creating DATE and WeekNum columns")
    master_df['DATE'] = np.where(
        master_df.CONFIRMEDBIDDATE.isnull(),
        np.where(master_df.SALE_DATE.isnull(), master_df.OFFER_DATE, master_df.SALE_DATE),
        master_df.CONFIRMEDBIDDATE
    )
    master_df['WeekNum'] = master_df['DATE'].dt.isocalendar().week
    master_df['Year'] = master_df['DATE'].dt.year

    master_df_sold = master_df[(master_df.Purchase_Flag!='Not Sold')]
    del master_df
    master_df_sold['DATE'] = np.where(master_df_sold['Channel']=='Private',master_df_sold['DATE'].dt.normalize(),master_df_sold['DATE'])
    
    updated_master_df = pd.concat([master_his,master_df_sold])
    del master_df_sold,master_his
    updated_master_df = updated_master_df[~updated_master_df.FULL_COMPONENT_CODE.isnull()].reset_index(drop = True)
    
    
    
    
if __name__ == "__main__":
    main()
