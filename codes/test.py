import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Hypothetical raw SMILES list
smiles_list = ["CCO", "c1ccccc1CC", "CC(=O)O"]

# Tokenize
PATTERN = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|[a-z]|@{1,2}|#|=|\\|\/|\+|-|\(|\)|\d+)"
tokenized_smiles = [re.findall(PATTERN, s) for s in smiles_list]

# Compute max length
max_length = max(len(tokens) for tokens in tokenized_smiles)
print(f"Max length: {max_length}")

# Save to .pkl (simplified)
data = {"tokenized_smiles": tokenized_smiles, "max_smiles_length": max_length}
with open("/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/tokenized_data.pkl", "wb") as f:
    pickle.dump(data, f)