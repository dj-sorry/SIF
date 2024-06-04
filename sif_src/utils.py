import numpy as np
import os
import wget

def load_wv(output_dir):
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_file = 'glove.6B.zip'
    
    # Download the zip file
    if not os.path.exists(os.path.join(output_dir, zip_file)):
        print(f"Downloading {zip_file}...")
        wget.download(url, out=output_dir)
        print("Download complete.")
    else:
        print(f"{zip_file} already exists.")
    
    # Extract the zip file
    if not os.path.exists(os.path.join(output_dir, 'glove.6B')):
        print(f"Extracting {zip_file}...")
        import zipfile
        with zipfile.ZipFile(os.path.join(output_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete.")
    else:
        print("Extracted files already exist.")
    
    # Return the path to the smallest set of GloVe vectors
    return os.path.join(output_dir, 'glove.6B', 'glove.6B.300d.txt')
