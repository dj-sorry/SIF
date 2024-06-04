import os
import requests
import zipfile
import numpy as np

def load_glove(output_dir):
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_file = 'glove.6B.zip'

    if not os.path.exists(os.path.join(output_dir, zip_file)):
        print(f"Downloading {zip_file}...")
        response = requests.get(url)
        with open(os.path.join(output_dir, zip_file), 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"{zip_file} already exists.")

    if not os.path.exists(os.path.join(output_dir, 'glove.6B')):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(os.path.join(output_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete.")
    else:
        print("Extracted files already exist.")

    return os.path.join(output_dir, 'glove.6B', 'glove.6B.300d.txt')


def load_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors