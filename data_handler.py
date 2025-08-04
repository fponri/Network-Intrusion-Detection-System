
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import os

scaler = StandardScaler()
feature_columns = None

def load_data(filepath):
    try:
        
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            print("\033[92m" + "Data loaded successfully.\n" + "\033[0m")
            return data
        else:
            # Try to load from current directory
            filename = os.path.basename(filepath)
            if os.path.exists(filename):
                data = pd.read_csv(filename)
                print("\033[92m" + f"Data loaded from {filename} successfully.\n" + "\033[0m")
                return data
            
            
            unsw_files = [f for f in os.listdir('.') if 'UNSW' in f and f.endswith('.csv')]
            if unsw_files:
                data = pd.read_csv(unsw_files[0])
                print("\033[92m" + f"Loaded authentic UNSW-NB15 data from {unsw_files[0]}.\n" + "\033[0m")
                return data
            
            print(f"File {filepath} not found. Please ensure UNSW-NB15 CSV files are in the current directory.")
            return None
            
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, fit_scaler=False, is_train=False):
    
    global feature_columns, scaler
    
    if data is not None:
        data = data.copy()
        
        data.ffill(inplace=True)
        
        # Standardize label column name if needed
        if 'Label' in data.columns:
            data = data.rename(columns={'Label': 'label'})
        if 'Attack_cat' in data.columns:
            data = data.drop('Attack_cat', axis=1)
        
        # Convert labels to binary if they're categorical
        if 'label' in data.columns:
            if data['label'].dtype == 'object':
                data['label'] = data['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
        
        # Handle categorical columns with label encoding to prevent string conversion errors
        categorical_columns = data.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            if col != 'label':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le
        
        data_encoded = pd.get_dummies(data)
        
        if is_train:
            feature_columns = data_encoded.columns
        else:
            data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
        
        if fit_scaler:
            scaler.fit(data_encoded.drop('label', axis=1))
            print("\nScaler fitted on initial data.\n\n")
        
        features = scaler.transform(data_encoded.drop('label', axis=1))
        labels = data_encoded['label'].values
        print("\033[92m" + "Data preprocessed successfully." + "\033[0m")
        return features, labels
    else:
        print("No data to preprocess.")
        return None, None

def simulate_data_stream(data, batch_size=100):
    
    data_copy = data.copy()
    for i in range(0, len(data_copy), batch_size):
        yield data_copy.iloc[i:i + batch_size]
