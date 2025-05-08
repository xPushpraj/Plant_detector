import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

def download_dataset():
    # Set up Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset = 'pushpraj072/plants'
    download_path = 'data/raw'
    os.makedirs(download_path, exist_ok=True)
    
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    
    # Move files to proper structure
    for item in os.listdir(download_path):
        src = os.path.join(download_path, item)
        dst = os.path.join('data', item)
        if os.path.isdir(src):
            shutil.move(src, dst)
    
    # Clean up
    shutil.rmtree(download_path)