import os
import pickle
import numpy as np

# ✅ Load Tokenized Data
input_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/tokenized_data.pkl"
output_dir = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/split_data"
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "rb") as f:
    data = pickle.load(f)

tokenized_smiles = data["tokenized_smiles"]
char_to_idx = data["char_to_idx"]
idx_to_char = data["idx_to_char"]
max_smiles_length = data["max_smiles_length"]

# ✅ Shuffle Data
np.random.seed(42)  # For reproducibility
np.random.shuffle(tokenized_smiles)

# ✅ Split Data into 10 Parts
num_splits = 10
split_size = len(tokenized_smiles) // num_splits
splits = [tokenized_smiles[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

# ✅ Save Each Split as a Separate .pkl File
for i, split in enumerate(splits):
    split_data = {
        "tokenized_smiles": split,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "max_smiles_length": max_smiles_length
    }
    split_file = os.path.join(output_dir, f"split_{i + 1}.pkl")
    with open(split_file, "wb") as f:
        pickle.dump(split_data, f)
    print(f"✅ Saved split {i + 1} to {split_file}")

print("✅ Data successfully split into 10 parts.")
