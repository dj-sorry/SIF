import os
import requests
import zipfile
import numpy as np
import jax.numpy as jnp
import tqdm

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


def load_glove_vectors(filepath):
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        num_lines = sum(1 for line in f)
        f.seek(0)  # Reset file pointer to beginning
        for line in tqdm.tqdm(f, total=num_lines, desc="Loading GloVe Vectors"):
            values = line.split()
            word = values[0]
            coefs = jnp.array(values[1:], dtype=jnp.float32)
            embeddings_index[word] = coefs
    return embeddings_index
