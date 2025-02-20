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
import decimal
import pickle 
import time
import sys
import datetime
import boto3
from pathlib import Path
from io import BytesIO
s3_client = boto3.client('s3')
from sklearn.ensemble import GradientBoostingRegressor
# ----------------------------- Code -----------------------------------


RUN_DATE = datetime.date.today().strftime("%d-%b-%y")

S3_LOGS_PATH = f'bgdn-pre-prod/dev-pipeline/logs/{RUN_DATE}' 

# Following dictionary contains the location for fetching the data
GLOBAL_PATHS= {
                "MODEL_INPUT_DATA":f"{S3_ROOT_PATH}/model-taining-data/ML_Model_Input_v1_2.snappy.parquet",
                "MODEL_PICKLE_FILE":"bgdn-pre-prod/model-pickle-files/model-v-1-2/" ,
}

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

lot_info_variables2 = ['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 
                     'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE','QUALITY_VARIANCE','INVOICE_WT',
                     'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail',
                     'PURCHASED_PRICE_Detail', 'Channel', 'Purchase_Flag', 'Purchases_Value','Cluster','Quadrant','GRADE_CODE','key'
                     ]


LOT_INFO_VARIABLES ={
    "Auction": ['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
                                   'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE',
                                   'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail',
                                   'Channel', 'Purchase_Flag', 'Purchases_Value',  'Cluster',
                                   'Train_flag', 'Quadrant', 'GRADE_CODE', 'key','T_P_C_R_AP_MA5',
                                   "T_P_C_R_AP_MA3","T_P_A_C_PD_MA20","T_P_G_C_PD_MA5","A_A_G_E_AP_MA1"
                        ],
                     
                     
                     "Private":['DATE', 'WeekNum', 'Year', 'CATALOG_ITEM_ID', 'BUYING_TYPE_ID',
                                   'BUYING_CENTER_CODE', 'GARDEN_NAME', 'FULL_COMPONENT_CODE',
                                   'OFFER_PRICE', 'PURCHASED_QUANTITY_Detail', 'PURCHASED_PRICE_Detail',
                                   'Channel', 'Purchase_Flag', 'Purchases_Value', 'Cluster', 'Train_flag', 'Quadrant', 'GRADE_CODE',
                            'key',"A_U_B_C_PD_MA2", 'O_U_G_E_AP_MA3', 
                                "A_A_G_E_AP_MA1",'A_A_C_R_AP_MA3',
                                'A_A_G_E_AP_MA3', 'A_A_G_C_AP_MA2',
                                'O_U_G_C_PD_MA2', 'T_A_C_R_AP_MA1',
                                'A_A_G_C_PD_MA5', 'A_A_G_C_PD_MA10',]
              }

gbr = GradientBoostingRegressor(learning_rate=0.07,
    n_estimators=100,
    subsample=1.0,max_depth=4)

model_list = {  #"LinearRegression":lr,
                #"XGBRegressor":xgb,
                "GradientBoostingRegressor":gbr
                #"RandomForestRegressor":rf
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
    
     

logger = get_module_logger("Model_Retaining")

def evaluation_metric(df):
    """
    Computes evaluation metrics for a given DataFrame 'df' containing actual and predicted values.

    Args:
    - df (pandas DataFrame): DataFrame containing at least two columns:
        - 'PURCHASED_PRICE_Detail': Actual purchase prices.
        - 'Model_Prediction': Predicted purchase prices.

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'r2': R-squared score between 'PURCHASED_PRICE_Detail' and 'Model_Prediction'.
        - 'mape': Mean Absolute Percentage Error (MAPE) between 'PURCHASED_PRICE_Detail' and 'Model_Prediction'.
        - 'mae': Mean Absolute Error (MAE) between 'PURCHASED_PRICE_Detail' and 'Model_Prediction'.
        - 'NumOfLots': Number of rows (instances) in 'df'.

    Note:
    - Requires sklearn.metrics.r2_score for R-squared calculation.
    - Assumes 'mean_absolute_percentage_error' is defined elsewhere for MAPE calculation.
    """
    out = {}

    # Calculate R-squared score
    out["r2"] = r2_score(df.PURCHASED_PRICE_Detail, df.Model_Prediction)

    # Calculate Mean Absolute Percentage Error (MAPE)
    out["mape"] = mean_absolute_percentage_error(df.PURCHASED_PRICE_Detail, df.Model_Prediction)

    # Calculate Mean Absolute Error (MAE)
    out["mae"] = mean_absolute_error(df.PURCHASED_PRICE_Detail, df.Model_Prediction)

    # Count the number of rows (instances) in 'df'
    out['NumOfLots'] = df.shape[0]

    return out


def get_accuracy(model, X_train, y_train, cv):
    """
    Evaluates a given model using cross-validation and returns accuracy metrics.

    Args:
    - model (estimator object): The model to be evaluated.
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values (ground truth) for X_train.
    - cv (int or cross-validation generator): Determines the cross-validation splitting strategy.

    Returns:
    - list: A list containing:
        - Mean R-squared (r2) score across cross-validation folds.
        - The first estimator trained during cross-validation.

    Note:
    - Assumes the model supports the 'fit' method and returns estimator objects in 'cv_results'.
    - Requires 'sklearn.model_selection.cross_validate' for cross-validation.
    """
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=-1,  # Use all available CPU cores for parallel computation
        scoring=[
            #"neg_mean_absolute_error", 
            #"neg_root_mean_squared_error",
            "neg_mean_absolute_percentage_error",
            "r2"
        ],
        return_estimator=True
    )

    # Extract R-squared scores from cross-validation results
    r2 = cv_results["test_r2"]

    # Prepare the result list with mean R-squared score and the first estimator
    result = [
        float(format((r2.mean()), ".3f")),  # Mean R-squared score formatted to 3 decimal places
        cv_results["estimator"][0]  # The first estimator object trained during cross-validation
    ]

    return result

def train_test_split(df, chnl="Auction", test_all=False, rolling_52=False, period = "365 days"):
    """
    Splits the given DataFrame into training and testing datasets based on specific conditions.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    chnl (str): The channel to filter the data on (default is "Auction").
    test_all (bool): Whether to test on all data after the split date (default is False).
    rolling_52 (bool): Whether to filter data for the last 52 weeks (default is False).
    
    Returns:
    tuple: A tuple containing:
        - X_train (pd.DataFrame): The training features.
        - y_train (pd.DataFrame): The training target values.
        - X_test (pd.DataFrame): The testing features.
        - y_test (pd.DataFrame): The testing target values.
        - test_df_with_lot_info (pd.DataFrame): The test DataFrame with lot information.
    """
    lot_info_variables = LOT_INFO_VARIABLES[chnl]
    # Get the list of unique FULL_COMPONENT_CODEs where Purchase_Flag is 'T'
    tcpl_components = df[df.Purchase_Flag == 'T'].FULL_COMPONENT_CODE.unique().tolist()
    
    # Filter the DataFrame to include only rows with FULL_COMPONENT_CODEs in tcpl_components
    df = df[df.FULL_COMPONENT_CODE.isin(tcpl_components)].reset_index(drop=True)
    
    # Define the split date 
    date = RUN_DATE
    
    # Sort the DataFrame by date in ascending order and reset the index
    df.sort_values(by='DATE', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Fill missing values in the 'Code' column with 0
    df.Code = df.Code.fillna(0)
    
    # If rolling_52 is True, filter the DataFrame to include only the last 52 weeks of data
    if rolling_52:
        df = df[df.DATE >= pd.to_datetime(date) - pd.to_timedelta(period)]
    
    # Define the training set
    if TCPL_GUARDRAIL:
        X_train = df[(df.DATE <= date) & (df.Train_flag == True) & (df.Channel == chnl)].reset_index(drop=True).drop(columns=lot_info_variables)
        y_train = df[(df.DATE <= date) & (df.Train_flag == True) & (df.Channel == chnl)].reset_index(drop=True)[["PURCHASED_PRICE_Detail"]]
    
    # Define the training set
    else:
        X_train = df[(df.DATE <= date) & (df.Channel == chnl)].reset_index(drop=True).drop(columns=lot_info_variables)
        y_train = df[(df.DATE <= date) & (df.Channel == chnl)].reset_index(drop=True)[["PURCHASED_PRICE_Detail"]]
    # Define the testing set
    if not test_all:
        # Filter the test set based on the date and Train_flag
        X_test = df[(df.DATE > date) & (df.Train_flag == True) & (df.Channel == chnl)].reset_index(drop=True).drop(columns=lot_info_variables)
        y_test = df[(df.DATE > date) & (df.Train_flag == True) & (df.Channel == chnl)].reset_index(drop=True)[["PURCHASED_PRICE_Detail"]]
        test_df_with_lot_info = df[(df.DATE > date) & (df.Train_flag == True) & (df.Channel == chnl)].reset_index(drop=True)[lot_info_variables2]
    else:
        # Filter the test set based only on the date
        X_test = df[(df.DATE > date) & (df.Channel == chnl)].reset_index(drop=True).drop(columns=lot_info_variables)
        y_test = df[(df.DATE > date) & (df.Channel == chnl)].reset_index(drop=True)[["PURCHASED_PRICE_Detail"]]
        test_df_with_lot_info = df[(df.DATE > date) & (df.Channel == chnl)].reset_index(drop=True)[lot_info_variables2]
    X_train.fillna(0,inplace = True)
    
    return X_train, y_train, X_test, y_test, test_df_with_lot_info

def convert_decimals_to_float(df):
    for column in df.columns:
        if df[column].dtype == 'object' and df[column].apply(lambda x: isinstance(x, decimal.Decimal)).any():
            df[column] = df[column].apply(float)
    return df

def model_test(model, x_test, test_df_with_lot_info):
    test_prediction = model.predict(x_test)
    test_df_with_lot_info['Model_Prediction'] = test_prediction
    # return plot_graph(test_df_with_lot_info, date, comp, split, cutoff)
    return test_df_with_lot_info


def mult_model(X_train, y_train, X_test, y_test, test_df_with_lot_info, models, seg=None,chnl ='Auction',  save=False):
    """
    Trains multiple models on the given training data, evaluates them on test data, and saves the models if specified.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame): Training target values.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.DataFrame): Testing target values.
    test_df_with_lot_info (pd.DataFrame): Testing target values with additional information.
    models (dict): Dictionary of models to be trained.
    seg (str): Segment identifier for saving models.
    save (bool): Flag to save the models (default is False).
    
    Returns:
    tuple: A tuple containing:
        - output_df (pd.DataFrame): DataFrame containing evaluation metrics for each model.
        - test_df_with_lot_info (pd.DataFrame): Testing target values with predictions added.
    """
    
    # Get a list of unique component codes from the test data
    component_list = test_df_with_lot_info.FULL_COMPONENT_CODE.unique().tolist()
    
    # Initialize the output DataFrame to store evaluation metrics
    output_df = pd.DataFrame(columns=['DATE', 'Component', 'NumOfLots'], index=range(len(component_list)))
    output_df['Component'] = component_list
    
    # Dictionary to store trained models
    model_dic = {}
    
    # Loop through each model in the provided models dictionary
    for model_name, model in models.items():
        start_time = time.time()
        
        # Train the model and get the accuracy score
        trained_model = get_accuracy(model, X_train, y_train.values.ravel(), 10)[1]
        model_dic[model_name] = trained_model
        end_time = time.time()
        
        logger.info(f"Training completed for {model_name} in {end_time - start_time} seconds.")
        training_error =  mean_absolute_percentage_error(trained_model.predict(X_train), y_train)
        logger.info(f" Training Error for {model_name} -- {training_error}")
        
        if save:
            with BytesIO() as bytes_stream:
                pickle.dump(trained_model, bytes_stream)
                bytes_stream.seek(0)
                path = GLOBAL_PATHS["MODEL_PICKLE_FILE"]
                object_key = f"{path}{seg}_{model_name}_{chnl}_v1_2.pkl"
                s3_client.upload_fileobj(bytes_stream,S3_BUCKET_NAME, object_key)
                logger.info(f'Model saved in {object_key}.')
        

    return 

def split_pred(model_input_df,quad_comp, seg, chnl='Auction',period='366 days'):
    """
    Splits the data for given components and segments, and performs prediction using multiple models.
    
    Parameters:
    model_input_df(pd.DataFrame): Model training and validation data.
    quad_comp (list): List of component codes to filter the data on.
    seg (str): The segment to be used in the model.
    chnl (str): The channel to filter the data on (default is 'Auction').
    
    Returns:
    tuple: A tuple containing:
        - test_prediction: Component level datascience metric summary for test_prediction.
        - out2: Actual test_prediction with lot information
        - c: The test features from the train_test_split function.
        - e: The test target values from the train_test_split function.
    """
    
    # Filter the model_input_df DataFrame to include only rows with FULL_COMPONENT_CODEs in quad_comp
    temp = model_input_df[model_input_df.FULL_COMPONENT_CODE.isin(quad_comp)].reset_index(drop=True)
    
    # Split the filtered data into training and testing sets with specific conditions
    X_train, y_train, X_test, y_test, test_df_with_lot_info = train_test_split(temp, chnl, test_all=False, rolling_52=False,period=period)
    logger.info(f"Model Input Summary\n-----{chnl}----{seg}-----\nTraingin data:  {X_train.shape[0]} records with  {X_train.shape[1]} features ")
    logger.info(f"Model Input Feature Names are : {X_train.columns}")
    # Perform prediction using multiple models
    mult_model(X_train, y_train, X_test, y_test, test_df_with_lot_info, model_list, seg,chnl, True)
    
    # Return the results
    return 


def main(Channel = 'Auction'):
    logger.info(f"--------------- Model retraing Module Started for {RUN_DATE} with TCPL Guardrail = {TCPL_GUARDRAIL}------------------")
    try:
        model_input_df = get_model_train_data() # train data from s3 
    except Exception as e:
        logger.error(" Error getting model input data: %s" % e)
        
    logger.info(f"Max date using for model training.. {model_input_df.DATE.max()}")
    # Smooth Components Model Training. 
    try:
        logger.info("Model training for Smooth Components Started...")
        split_pred(model_input_df,smooth,'Smooth',Channel)
    except Exception as e:
        logger.error(" Error while training for Smooth Components {}".format(e))
        
    # Intermittent Components Model Training. 
    try:
        logger.info("Model training for Intermittent Components Started...")
        split_pred(model_input_df,intermittent,'Intermittent', Channel)
    except Exception as e:
        logger.error(" Error while training for Intermittent Components {}".format(e))
    
    
    # Lumpy_bin_1 Components Model Training. 
    try:
        logger.info("Model training for Smooth Components Started...")
        split_pred(model_input_df,Lumpy_bin_1,'Lumpy_bin_1', Channel)
        
    except Exception as e:
        logger.error(" Error while training for Smooth Components {}".format(e))
    
    # Smooth Components Model Training. 
    try:
        logger.info("Model training for Smooth Components Started...")
        split_pred(model_input_df,Lumpy_bin_2,'Lumpy_bin_2', Channel)
    except Exception as e:
        logger.error(" Error while training for Smooth Components {}".format(e))
    
    # Lumpy_bin_3 Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_3 Components Started...")
        split_pred(model_input_df,Lumpy_bin_3,'Lumpy_bin_3', Channel)
        
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_3 Components {}".format(e))
    
    # Lumpy_bin_4 Components Model Training. 
    try:
        logger.info("Model training for Lumpy_bin_4 Components Started...")
        split_pred(model_input_df,Lumpy_bin_4,'Lumpy_bin_4', Channel)
        
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_4 Components {}".format(e))
    
    
    try:
        logger.info("Model training for Lumpy_bin_5 Components Started...")
        split_pred(model_input_df,Lumpy_bin_5,'Lumpy_bin_5', Channel)
        
    except Exception as e:
        logger.error(" Error while training for Lumpy_bin_5 Components {}".format(e))
    
    
if __name__ == '__main__':
    logger.info(f"Starting Main Process for Auction Channel...")
    main("Auction")
    logger.info(f"Finished Main Process for Auction Channel...")
    logger.info(f"Starting Main Process for Private Channel...")
    main("Private")
    logger.info(f"Finished Main Process for Private Channel...")
