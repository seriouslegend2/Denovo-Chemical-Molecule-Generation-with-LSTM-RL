import os
import re
import logging
import pickle
from rdkit import Chem

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Regular expression for SMILES tokenization (Now includes lowercase letters)
PATTERN = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|@{1,2}|#|=|\\|\/|\+|-|\(|\)|\d+|[a-z])"

def load_valid_smiles(file_path):
    """Load SMILES from a file and validate using RDKit."""
    valid_smiles = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Skip header if first line contains "smiles"
        if lines[0].strip().lower() == "smiles":
            lines = lines[1:]

        for line in lines:
            smiles = line.strip().split()[0]  # Extract first word (handles extra spaces)
            
            if smiles:  # Ignore empty lines
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    valid_smiles.append(smiles)
                else:
                    logger.warning("❌ Invalid SMILES skipped: %s", smiles)

    return valid_smiles

def tokenize_smiles(smiles_list):
    """Tokenize SMILES using a consistent regex approach."""
    return [re.findall(PATTERN, smiles) for smiles in smiles_list]

def create_token_dicts(tokenized_smiles):
    """Creates character-to-index and index-to-character mappings."""
    unique_tokens = sorted(set(token for smiles in tokenized_smiles for token in smiles))

    # ✅ Add padding and unknown tokens for safety
    char_to_idx = {"<PAD>": 0, "<UNK>": 1}  # Start indexing from 0
    char_to_idx.update({token: i + 2 for i, token in enumerate(unique_tokens)})  

    idx_to_char = {i: token for token, i in char_to_idx.items()}

    return char_to_idx, idx_to_char

def save_tokenized_data(tokenized_smiles, char_to_idx, idx_to_char, file_path):
    """Save tokenized SMILES and token dictionaries."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists

    max_smiles_length = max(len(smiles) for smiles in tokenized_smiles)

    data = {
        "tokenized_smiles": tokenized_smiles,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "max_smiles_length": max_smiles_length
    }

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    logger.info("💾 Tokenized SMILES saved to: %s", file_path)

def display_samples(tokenized_smiles, num_samples=5):
    """Display sample tokenized SMILES."""
    logger.info("🔍 Sample tokenized SMILES:")
    for i, tokens in enumerate(tokenized_smiles[:num_samples]):
        print(f"SMILES {i+1}: {tokens}")

if __name__ == "__main__":
    input_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/preprocessing_step_1.smi"  # Update this path
    output_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/tokenized_data.pkl"

    if not os.path.exists(input_file):
        logger.error("🚨 File not found: %s", input_file)
    else:
        logger.info("📂 Loading and tokenizing SMILES from %s", input_file)
        smiles_list = load_valid_smiles(input_file)
        
        if not smiles_list:
            logger.error("🚨 No valid SMILES found. Exiting.")
            exit(1)

        logger.info("✅ Loaded %d valid SMILES", len(smiles_list))
        
        tokenized_smiles = tokenize_smiles(smiles_list)
        char_to_idx, idx_to_char = create_token_dicts(tokenized_smiles)

        save_tokenized_data(tokenized_smiles, char_to_idx, idx_to_char, output_file)

        # Display some samples
        display_samples(tokenized_smiles)
