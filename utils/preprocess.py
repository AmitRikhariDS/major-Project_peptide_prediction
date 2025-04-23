# import numpy as np
# import pickle
# from keras.preprocessing.sequence import pad_sequences

# with open("data/Feature_50.pkl", "rb") as f:
#     Feature_50 = pickle.load(f)

# # Create a dictionary to map amino acids to indices
# AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(Feature_50.keys())}
# AA_TO_INDEX['X'] = 0  # Fallback for unknown AAs

# def preprocess_input(seq, max_len=50):
#     seq = seq.upper()
#     # Convert to index-based encoding using Feature_50 keys
#     encoded_seq = [AA_TO_INDEX.get(aa, 0) for aa in seq]
#     padded_seq = pad_sequences([encoded_seq], maxlen=max_len, padding='post')
#     return np.array(padded_seq)

# utils/preprocess.py
import numpy as np
import pickle
import pandas as pd

# Load selected top 50 features
with open("data/Feature_50.pkl", "rb") as f:
    Feature_50 = pickle.load(f)

# Define constants
max_length = 50
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # Standard amino acids
AA_TO_INDEX = {aa: i for i, aa in enumerate(amino_acids)}

# Pad peptide sequence to max length
def pad_sequence(seq, max_length):
    return seq.ljust(max_length, 'X')[:max_length]

# One-hot encode a sequence
def one_hot_encode(seq, aa_to_index):
    num_classes = len(aa_to_index)
    encoded = np.zeros((max_length, num_classes), dtype=int)
    for i, aa in enumerate(seq):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1
    return encoded.flatten()

# Main preprocessing function for model input
def preprocess_input(seq):
    padded = pad_sequence(seq.upper(), max_length)
    encoded = one_hot_encode(padded, AA_TO_INDEX)

    # Convert to DataFrame and filter top features
    column_names = [f'{aa}_Pos_{pos+1}' for pos in range(max_length) for aa in amino_acids]
    df = pd.DataFrame([encoded], columns=column_names)
    return df[list(Feature_50.keys())]

# Access Feature_50 in app
__all__ = ["preprocess_input", "Feature_50"]
