import os
import pandas as pd

def download_ms_marco(data_dir):
    pass

def load_data(file_path):
    return pd.read_csv(file_path)

def save_processed_data(data, file_path):
    data.to_csv(file_path, index=False)
