"""
Title:       Build a preprocessing pipeline for cyber attack data that helps user preprocess training
             and test data from the corresponding CSV input files.

Author:      ARYA KONER

"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler


#----------------------------------------------------------------------------#
INPUT_DIR = "../data/train test sets/Batch 0/"
OUTPUT_DIR = "../processed_data/supervised_learning/"
LOG_DIR = "../logs/featureEngineering/"

TRAIN_FILE = "train.csv"
TRAIN_FEATURES_FILE = 'train_features.txt'

TEST_FILE = "test.csv"
TEST_FEATURES_FILE = 'test_features.txt'

# logging
logger= logging.getLogger('cyber_featureEngineering')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)

# Ensure LOG_DIR exists and use absolute path
import os
log_dir = os.path.abspath(LOG_DIR)
os.makedirs(log_dir, exist_ok=True)
fh = logging.FileHandler(os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
logger.addHandler(fh)

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#----------------------------------------------------------------------------#
def read_data(file_name, drop_na=False):
    """
    Read cyber attack data in the .csv file.

    Inputs:
        - data_file (string): name of the data file.
        - drop_na (bool): whether to drop rows with any missing values

    Returns:
        (DataFrame) clean data set with correct data types

    """
    data = pd.read_csv(INPUT_DIR + file_name)

    if drop_na:
        data.dropna(axis=0, inplace=True)

    return data


def ask(names, message):
    """
    Ask user for their choice of index for model or metrics.

    Inputs:
        - name (list of strings): name of choices
        - message (str): type of index to request from user

    Returns:
        (int) index for either model or metrics
    """
    indices = []

    print("\nUp till now we support:")
    for i, name in enumerate(names):
        print("%s. %s" % (i + 1, name))
        indices.append(str(i + 1))

    index = input("Please input a %s index:\n" % message)

    if index in indices:
        return int(index) - 1
    else:
        print("Input wrong. Type one in {} and hit Enter.".format(indices))
        return ask(names, message)


def create_dirs(dir_path):
    """
    Create directory if it does not exist.

    Inputs:
        - dir_path (string): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def fill_missing_values(df, method='median'):
    """
    Fill missing values in the data.

    Inputs:
        - df (DataFrame): input data
        - method (string): method to fill missing values

    Returns:
        (DataFrame) data with filled missing values

    """
    logger.info("Filling missing values...")

    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            if method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'zero':
                df[col].fillna(0, inplace=True)

    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def create_cyber_features(df):
    """
    Create cyber security specific features.

    Inputs:
        - df (DataFrame): input data

    Returns:
        (DataFrame) data with new features

    """
    logger.info("Creating cyber security features...")

    # Traffic intensity features
    if 'traffic_volume' in df.columns and 'duration' in df.columns:
        df['traffic_intensity'] = df['traffic_volume'] / (df['duration'] + 1)
    
    if 'packet_count' in df.columns and 'duration' in df.columns:
        df['packet_rate'] = df['packet_count'] / (df['duration'] + 1)
    
    if 'byte_count' in df.columns and 'duration' in df.columns:
        df['byte_rate'] = df['byte_count'] / (df['duration'] + 1)

    # Port-based features
    if 'dest_port' in df.columns:
        df['is_common_port'] = df['dest_port'].isin([80, 443, 22, 21, 25, 53, 110, 143]).astype(int)
        df['is_privileged_port'] = (df['dest_port'] < 1024).astype(int)

    # Protocol-based features
    if 'protocol_type' in df.columns:
        df['is_tcp'] = (df['protocol_type'] == 'tcp').astype(int)
        df['is_udp'] = (df['protocol_type'] == 'udp').astype(int)
        df['is_icmp'] = (df['protocol_type'] == 'icmp').astype(int)

    # Service-based features
    if 'service_type' in df.columns:
        df['is_web_service'] = df['service_type'].isin(['http', 'https']).astype(int)
        df['is_mail_service'] = df['service_type'].isin(['smtp', 'pop3', 'imap']).astype(int)
        df['is_file_service'] = df['service_type'].isin(['ftp', 'sftp']).astype(int)

    # Time-based features
    if 'hour' in df.columns:
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night_hours'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

    # Attack pattern features
    if 'failed_logins' in df.columns:
        df['high_failed_logins'] = (df['failed_logins'] > df['failed_logins'].quantile(0.95)).astype(int)
    
    if 'root_access' in df.columns:
        df['suspicious_root_access'] = (df['root_access'] > 0).astype(int)
    
    if 'num_compromised' in df.columns:
        df['high_compromise'] = (df['num_compromised'] > df['num_compromised'].quantile(0.95)).astype(int)

    # Network behavior features
    if 'source_network' in df.columns and 'dest_network' in df.columns:
        df['same_network'] = (df['source_network'] == df['dest_network']).astype(int)

    return df


def discretize_continuous_variables(df, method='quantile', n_bins=10):
    """
    Discretize continuous variables.

    Inputs:
        - df (DataFrame): input data
        - method (string): discretization method
        - n_bins (int): number of bins

    Returns:
        (DataFrame) data with discretized variables

    """
    logger.info("Discretizing continuous variables...")

    continuous_columns = df.select_dtypes(include=[np.number]).columns
    
    # Remove target variable and already categorical variables
    if 'is_attack' in continuous_columns:
        continuous_columns = continuous_columns.drop('is_attack')
    
    # Remove binary variables
    binary_columns = [col for col in continuous_columns if df[col].nunique() == 2]
    continuous_columns = continuous_columns.drop(binary_columns)

    for col in continuous_columns:
        if df[col].nunique() > n_bins:
            if method == 'quantile':
                df[f'{col}_binned'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            elif method == 'equal_width':
                df[f'{col}_binned'] = pd.cut(df[col], bins=n_bins, labels=False, duplicates='drop')

    return df


def encode_categorical_variables(df):
    """
    Encode categorical variables.

    Inputs:
        - df (DataFrame): input data

    Returns:
        (DataFrame) data with encoded categorical variables

    """
    logger.info("Encoding categorical variables...")

    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Use ordinal encoding for categorical variables
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    if len(categorical_columns) > 0:
        df_encoded = encoder.fit_transform(df[categorical_columns])
        df_encoded = pd.DataFrame(df_encoded, columns=categorical_columns, index=df.index)
        
        # Replace original columns with encoded ones
        for col in categorical_columns:
            df[col] = df_encoded[col]
        
        # Save encoder for later use
        joblib.dump(encoder, OUTPUT_DIR + 'categorical_encoder.pkl')

    return df


def scale_features(df, method='standard'):
    """
    Scale features.

    Inputs:
        - df (DataFrame): input data
        - method (string): scaling method

    Returns:
        (DataFrame) data with scaled features

    """
    logger.info("Scaling features...")

    # Separate features and target
    if 'is_attack' in df.columns:
        target = df['is_attack']
        features = df.drop('is_attack', axis=1)
    else:
        features = df
        target = None

    # Scale features
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

    # Save scaler for later use
    joblib.dump(scaler, OUTPUT_DIR + 'feature_scaler.pkl')

    # Recombine features and target
    if target is not None:
        df_scaled = pd.concat([features_scaled, target], axis=1)
    else:
        df_scaled = features_scaled

    return df_scaled


def save_features(df, file_name):
    """
    Save features to file.

    Inputs:
        - df (DataFrame): data to save
        - file_name (string): name of the file

    """
    logger.info(f"Saving features to {file_name}...")

    # Save feature names
    feature_names = df.columns.tolist()
    if 'is_attack' in feature_names:
        feature_names.remove('is_attack')
    
    with open(OUTPUT_DIR + file_name, 'w') as f:
        for feature in feature_names:
            f.write(feature + '\n')

    # Save data as numpy array
    if 'is_attack' in df.columns:
        X = df.drop('is_attack', axis=1).values
        y = df['is_attack'].values
        np.save(OUTPUT_DIR + 'X_' + file_name.replace('.txt', '.npy'), X)
        np.save(OUTPUT_DIR + 'y_' + file_name.replace('.txt', '.npy'), y)
    else:
        X = df.values
        np.save(OUTPUT_DIR + 'X_' + file_name.replace('.txt', '.npy'), X)


def read_feature_names(file_name):
    """
    Read feature names from file.

    Inputs:
        - file_name (string): name of the file

    Returns:
        (list) list of feature names

    """
    feature_names = []
    with open(OUTPUT_DIR + file_name, 'r') as f:
        for line in f:
            feature_names.append(line.strip())
    return feature_names


def main():
    """
    Main function for cyber attack feature engineering.
    """
    parser = argparse.ArgumentParser(description='Cyber Attack Feature Engineering')
    parser.add_argument('--ask', type=int, default=1, help='Ask user for choices')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.info("Starting cyber attack feature engineering...")

    # Create directories
    create_dirs(OUTPUT_DIR)
    create_dirs(LOG_DIR)

    # Read data
    train_data = read_data(TRAIN_FILE)
    test_data = read_data(TEST_FILE)
    
    if args.verbose:
        logger.info(f"Train data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")

    # Fill missing values
    train_data = fill_missing_values(train_data)
    test_data = fill_missing_values(test_data)

    # Create cyber security features
    train_data = create_cyber_features(train_data)
    test_data = create_cyber_features(test_data)

    # Discretize continuous variables - FIT ONLY ON TRAINING DATA
    logger.info("Discretizing continuous variables...")
    continuous_columns = train_data.select_dtypes(include=[np.number]).columns
    
    # Remove target variable and already categorical variables
    if 'is_attack' in continuous_columns:
        continuous_columns = continuous_columns.drop('is_attack')
    
    # Remove binary variables
    binary_columns = [col for col in continuous_columns if train_data[col].nunique() == 2]
    continuous_columns = continuous_columns.drop(binary_columns)

    # Store bin edges for each column
    bin_edges = {}
    
    for col in continuous_columns:
        if train_data[col].nunique() > 10:  # Only bin if enough unique values
            # Create bins based on training data
            bins = pd.qcut(train_data[col], q=10, retbins=True, duplicates='drop')[1]
            bin_edges[col] = bins
            
            # Apply to both train and test data
            train_data[f'{col}_binned'] = pd.cut(train_data[col], bins=bins, labels=False, include_lowest=True)
            test_data[f'{col}_binned'] = pd.cut(test_data[col], bins=bins, labels=False, include_lowest=True)
            
            # Fill NaN values (for values outside the bin range)
            train_data[f'{col}_binned'] = train_data[f'{col}_binned'].fillna(-1)
            test_data[f'{col}_binned'] = test_data[f'{col}_binned'].fillna(-1)

    # Encode categorical variables - FIT ONLY ON TRAINING DATA
    logger.info("Encoding categorical variables...")
    categorical_columns = train_data.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        # Fit encoder on training data only
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(train_data[categorical_columns])
        
        # Transform both train and test data
        train_encoded = encoder.transform(train_data[categorical_columns])
        test_encoded = encoder.transform(test_data[categorical_columns])
        
        # Convert to DataFrame and replace original columns
        train_encoded = pd.DataFrame(train_encoded, columns=categorical_columns, index=train_data.index)
        test_encoded = pd.DataFrame(test_encoded, columns=categorical_columns, index=test_data.index)
        
        for col in categorical_columns:
            train_data[col] = train_encoded[col]
            test_data[col] = test_encoded[col]
        
        # Save encoder for later use
        joblib.dump(encoder, OUTPUT_DIR + 'categorical_encoder.pkl')

    # Scale features - FIT ONLY ON TRAINING DATA
    logger.info("Scaling features...")
    
    # Separate features and target for training data
    if 'is_attack' in train_data.columns:
        train_target = train_data['is_attack']
        train_features = train_data.drop('is_attack', axis=1)
        test_target = test_data['is_attack']
        test_features = test_data.drop('is_attack', axis=1)
    else:
        train_features = train_data
        test_features = test_data
        train_target = None
        test_target = None

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Transform both train and test features
    train_features_scaled = scaler.transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Convert back to DataFrame
    train_features_scaled = pd.DataFrame(train_features_scaled, columns=train_features.columns, index=train_features.index)
    test_features_scaled = pd.DataFrame(test_features_scaled, columns=test_features.columns, index=test_features.index)
    
    # Save scaler for later use
    joblib.dump(scaler, OUTPUT_DIR + 'feature_scaler.pkl')

    # Recombine features and target
    if train_target is not None:
        train_data = pd.concat([train_features_scaled, train_target], axis=1)
        test_data = pd.concat([test_features_scaled, test_target], axis=1)
    else:
        train_data = train_features_scaled
        test_data = test_features_scaled

    # Save processed data
    save_features(train_data, TRAIN_FEATURES_FILE)
    save_features(test_data, TEST_FEATURES_FILE)

    if args.verbose:
        logger.info("Cyber attack feature engineering completed successfully!")


if __name__ == "__main__":
    main()
