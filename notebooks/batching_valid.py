import cProfile
import pstats
import io
import sys
sys.path.append("..")
from sif_src.loader import load_data, save_processed_data
from sif_src.preprocess_data import preprocess_text
from sif_src.utils import load_glove, load_glove_vectors
from sif_src.sif import compute_word_frequencies, compute_sif_weights, compute_sif_embeddings, remove_pc
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import gc

def main():
    print('beginning')

    # Load data
    train_df = pd.read_pickle("pickle_backups/marco_train_df2024-06-04T17.38.1717490321.pickle")
    valid_df = pd.read_pickle("pickle_backups/marco_valid_df2024-06-04T17.38.1717490321.pickle")
    glove_vectors = load_glove_vectors('wv/glove.6B.50d.txt')

    def extract_passage_texts(row):
        passage_dicts = row['passages']
        passage_texts = [passage_dicts['passage_text'][i] for i in range(len(passage_dicts['passage_text']))]
        return " ".join(passage_texts)

    print("finish pickling")

    # Apply the function to the DataFrame
    train_df['passage_text'] = train_df.apply(extract_passage_texts, axis=1)
    valid_df['passage_text'] = valid_df.apply(extract_passage_texts, axis=1)
    print("finish passage texts")

    train_corpus = train_df['query'].tolist() + train_df['passage_text'].tolist()
    word_freq = compute_word_frequencies(train_corpus)
    sif_weights = compute_sif_weights(word_freq)

    print("finish weighting")

    # Parameters for batch processing
    batch_size = 50000  # Adjust based on your memory constraints
    num_batches = len(valid_df) // batch_size + (1 if len(valid_df) % batch_size != 0 else 0)

    file_path3 = "pickle_backups/valid_passages_sif_embeddings.pickle"
    log_file_path = "pickle_backups/valid_processed_batches.log"

    # Create log file if it doesn't exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write("")

    # Read the log file to get the list of processed batches
    with open(log_file_path, 'r') as log_file:
        processed_batches = set(int(line.strip()) for line in log_file)

    def process_batch(batch_index):
        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, len(valid_df))
        
        batch_passages_sif = compute_sif_embeddings(valid_df['passage_text'].iloc[start_idx:end_idx].tolist(), glove_vectors, sif_weights)
        
        # Append to pickle file
        with open(file_path3, 'ab') as f:
            pickle.dump(batch_passages_sif, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log the completed batch
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{batch_index}\n")

        # Explicitly delete batch data and run garbage collection
        del batch_passages_sif
        gc.collect()

    # Profile the main function
    pr = cProfile.Profile()
    pr.enable()

    # Process and save embeddings in batches with tqdm for progress
    for i in tqdm(range(num_batches), desc="Processing batches"):
        if i in processed_batches:
            print(f"Batch {i+1} already processed, skipping.")
            continue

        process_batch(i)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    print("SIF embeddings are loaded successfully.")
    print("finished")

if __name__ == "__main__":
    main()
