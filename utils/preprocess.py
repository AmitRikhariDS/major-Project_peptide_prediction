import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

with open("data/Feature_50.pkl", "rb") as f:
    Feature_50 = pickle.load(f)

# Create a dictionary to map amino acids to indices
AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(Feature_50.keys())}
AA_TO_INDEX['X'] = 0  # Fallback for unknown AAs

def preprocess_input(seq, max_len=50):
    seq = seq.upper()
    # Convert to index-based encoding using Feature_50 keys
    encoded_seq = [AA_TO_INDEX.get(aa, 0) for aa in seq]
    padded_seq = pad_sequences([encoded_seq], maxlen=max_len, padding='post')
    return np.array(padded_seq)