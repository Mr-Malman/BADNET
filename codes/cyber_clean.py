"""
Title:       Clean the aggregated cyber attack data. Create new features that takes
             cyber attacks in the past into considerations.

Author:      ARYA KONER
"""

import pandas as pd
import numpy as np
import os
from os import path

#----------------------------------------------------------------------------#
INPUT_DIR = "../data/"
OUTPUT_DIR = "../data/aggregated_data/"


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

    data['timestamp'] = pd.to_datetime(data['timestamp'])

    return data


def slice_data(data, var, years):
    
    (start_year, end_year) = years
    early_start_date = str(start_year - 5) + "-01-01"
    end_date = str(end_year + 1) + "-01-01"

    full_data = data[(data[var] >= early_start_date) & (data[var] < end_date)]

    return full_data


def clean(df):
    df.timestamp = pd.to_datetime(df.timestamp)
    
    # Defensive check before dropping columns
    drop_cols = ['eventid', 'source_ip', 'dest_ip']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df['year'] = df.timestamp.dt.year
    df['month'] = df.timestamp.dt.month
    df['hour'] = df.timestamp.dt.hour
    df = df.assign(network_id=(
                df['source_network'].astype(str) + '_' + df['dest_network'].astype(
            str)).astype('category').cat.codes)
    df['unique_id'] = df[['network_id', 'year']].apply(tuple, axis=1)

    # Convert numeric columns
    numeric_cols = ['traffic_volume', 'packet_count', 'byte_count', 'duration',
                   'source_port', 'dest_port']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    return df


def aggregate(df):
    df_list = list()
    col_names = list()
    
    # Handle attack_type specially - use mode (most common) instead of nunique
    if 'attack_type' in df.columns:
        temp_series = df.groupby('unique_id')['attack_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        df_list.append(temp_series)
        col_names.append('attack_type')
    
    # Count unique values for other categorical features
    for col in ['protocol_type', 'service_type', 'flag']:
        if col in df.columns:
            temp_series = df.groupby('unique_id')[col].nunique()
            df_list.append(temp_series)
            col_names.append(col)

    # Sum numeric features
    for col in ['traffic_volume', 'packet_count', 'byte_count', 'duration',
                'failed_logins', 'root_access', 'num_compromised']:
        if col in df.columns:
            temp_series = df.groupby('unique_id')[col].sum()
            df_list.append(temp_series)
            col_names.append(col)

    # Take unique values for categorical features
    for col in ['source_network', 'dest_network', 'year', 'month', 'hour', 'network_id']:
        if col in df.columns:
            temp_series = df.groupby('unique_id')[col].unique().apply(
                lambda x: x[0])
            df_list.append(temp_series)
            col_names.append(col)

    # Calculate averages for continuous features
    for col in ['traffic_volume', 'packet_count', 'byte_count', 'duration']:
        if col in df.columns:
            temp_series = df.groupby('unique_id')[col].mean()
            df_list.append(temp_series)
            col_names.append(f'{col}_avg')

    # Calculate standard deviations
    for col in ['traffic_volume', 'packet_count', 'byte_count', 'duration']:
        if col in df.columns:
            temp_series = df.groupby('unique_id')[col].std()
            # Replace NaN std with 0 (occurs if single record in group)
            temp_series = temp_series.fillna(0)
            df_list.append(temp_series)
            col_names.append(f'{col}_std')

    aggregated_df = pd.concat(df_list, axis=1)
    aggregated_df.columns = col_names

    # Reset index to flatten DataFrame
    aggregated_df = aggregated_df.reset_index(drop=True)

    return aggregated_df


def create_target(df):
    """
    Create target variable for cyber attack prediction.
    """
    # Create binary target: 1 if any attack occurred, 0 otherwise
    if 'attack_type' in df.columns:
        # Check if attack_type is numeric (encoded) or string
        if pd.api.types.is_numeric_dtype(df['attack_type']):
            # Assuming 1 is 'normal' and others are attacks
            df['is_attack'] = (df['attack_type'] != 1).astype(int)
        else:
            df['is_attack'] = (df['attack_type'] != 'normal').astype(int)
    else:
        # If no attack_type column, use other indicators
        attack_indicators = ['failed_logins', 'root_access', 'num_compromised']
        # Ensure these columns exist
        existing_inds = [col for col in attack_indicators if col in df.columns]
        df['is_attack'] = df[existing_inds].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    
    return df


def create_sample_cyber_data():
    """
    Create sample cyber attack data for testing.
    """
    np.random.seed(42)
    n_samples = 10000
    
    # Generate timestamps
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    
    # Generate sample data with more balanced attack distribution
    data = pd.DataFrame({
        'eventid': range(n_samples),
        'timestamp': dates,
        'source_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'dest_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'source_network': [f"192.168.{np.random.randint(1, 10)}" for _ in range(n_samples)],
        'dest_network': [f"10.0.{np.random.randint(1, 10)}" for _ in range(n_samples)],
        'source_port': np.random.randint(1024, 65535, n_samples),
        'dest_port': np.random.choice([80, 443, 22, 21, 25, 53, 110, 143], n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service_type': np.random.choice(['http', 'https', 'ssh', 'ftp', 'smtp', 'dns'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S1', 'S2', 'S3', 'OTH'], n_samples),
        'traffic_volume': np.random.exponential(1000, n_samples),
        'packet_count': np.random.poisson(50, n_samples),
        'byte_count': np.random.exponential(5000, n_samples),
        'duration': np.random.exponential(10, n_samples),
        'failed_logins': np.random.poisson(0.1, n_samples),
        'root_access': np.random.binomial(1, 0.01, n_samples),
        'num_compromised': np.random.poisson(0.05, n_samples),
        'attack_type': np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r'], n_samples, p=[0.7, 0.1, 0.1, 0.07, 0.03])
    })
    
    # Ensure we have a good mix of normal and attack traffic
    # Make some adjustments to create more realistic patterns
    normal_mask = data['attack_type'] == 'normal'
    attack_mask = data['attack_type'] != 'normal'
    
    # Normal traffic should have lower failed_logins, root_access, num_compromised
    data.loc[normal_mask, 'failed_logins'] = np.random.poisson(0.01, sum(normal_mask))
    data.loc[normal_mask, 'root_access'] = np.random.binomial(1, 0.001, sum(normal_mask))
    data.loc[normal_mask, 'num_compromised'] = np.random.poisson(0.001, sum(normal_mask))
    
    # Attack traffic should have higher failed_logins, root_access, num_compromised
    data.loc[attack_mask, 'failed_logins'] = np.random.poisson(2.0, sum(attack_mask))
    data.loc[attack_mask, 'root_access'] = np.random.binomial(1, 0.3, sum(attack_mask))
    data.loc[attack_mask, 'num_compromised'] = np.random.poisson(1.0, sum(attack_mask))
    
    # Save sample data
    if not path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    data.to_csv(INPUT_DIR + "cyber_attacks.csv", index=False)
    
    return data


def create_train_test_splits(aggregated_data):
    """
    Create train/test splits for different time periods.
    """
    # Create train test sets directory
    train_test_dir = "../data/train test sets/"
    if not path.exists(train_test_dir):
        os.makedirs(train_test_dir)
    
    # Create batch directory
    batch_dir = train_test_dir + "Batch 0/"
    if not path.exists(batch_dir):
        os.makedirs(batch_dir)
    
    # Target variable should already exist from aggregation
    if 'is_attack' not in aggregated_data.columns:
        print("Warning: is_attack column not found. Creating target variable...")
        aggregated_data = create_target(aggregated_data)
    
    # Split data by year
    years = np.sort(aggregated_data['year'].unique())
    
    # Use first 80% of years for training, last 20% for testing
    split_idx = max(int(len(years) * 0.8), 1)  # ensure at least one year in train
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_data = aggregated_data[aggregated_data['year'].isin(train_years)]
    test_data = aggregated_data[aggregated_data['year'].isin(test_years)]
    
    # Check if both classes exist in train and test. If not, perform stratified random split.
    train_classes = train_data['is_attack'].unique()
    test_classes = test_data['is_attack'].unique()

    if len(train_classes) < 2 or len(test_classes) < 2:
        print(f"Warning: Train or Test set lacks both classes. Performing stratified random split instead.")
        from sklearn.model_selection import train_test_split
        # Reset index to avoid warning
        aggregated_data = aggregated_data.reset_index(drop=True)
        train_data, test_data = train_test_split(
            aggregated_data, test_size=0.2, stratify=aggregated_data['is_attack'], random_state=42)
        
        print(f"Train class distribution after stratified split: {train_data['is_attack'].value_counts().to_dict()}")
        print(f"Test class distribution after stratified split: {test_data['is_attack'].value_counts().to_dict()}")

    # Save splits
    train_data.to_csv(batch_dir + "train.csv", index=False)
    test_data.to_csv(batch_dir + "test.csv", index=False)
    
    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    print(f"Train attack distribution: {train_data['is_attack'].value_counts().to_dict()}")
    print(f"Test attack distribution: {test_data['is_attack'].value_counts().to_dict()}")


def main():
    """
    Main function to clean and aggregate cyber attack data.
    """
    print("Starting cyber attack data cleaning...")
    
    # Read the cyber attack data
    try:
        data = read_data("cyber_attacks.csv")
        print(f"Loaded {len(data)} cyber attack records")
    except FileNotFoundError:
        print("cyber_attacks.csv not found. Creating sample data...")
        data = create_sample_cyber_data()
    
    # Clean the data
    data = clean(data)
    print("Data cleaned successfully")
    
    # Aggregate data
    aggregated_data = aggregate(data)
    print(f"Aggregated data shape: {aggregated_data.shape}")
    
    # Create target variable AFTER aggregation
    aggregated_data = create_target(aggregated_data)
    print("Target variable created")
    
    # Save aggregated data
    if not path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    aggregated_data.to_csv(OUTPUT_DIR + "aggregated_cyber_data.csv", index=False)
    print(f"Aggregated data saved to {OUTPUT_DIR}aggregated_cyber_data.csv")
    
    # Create train/test splits
    create_train_test_splits(aggregated_data)
    print("Train/test splits created successfully")


if __name__ == "__main__":
    main()