import os
import re
import logging
import pickle
from rdkit import Chem
import matplotlib.pyplot as plt  # ✅ Import matplotlib for plotting

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

def count_smiles_tokens(tokenized_smiles):
    """Count individual SMILES tokens."""
    token_counts = {}
    for smiles in tokenized_smiles:
        for token in smiles:
            token_counts[token] = token_counts.get(token, 0) + 1
    return token_counts

def normalize_counts(counts):
    """Normalize a list of counts to proportions between 0 and 1."""
    total_count = sum(counts)
    return [count / total_count if total_count > 0 else 0 for count in counts]

def add_counts_to_bars(bars, absolute_counts, proportions):
    """Add absolute counts and proportions on top of bars."""
    for bar, abs_count, proportion in zip(bars, absolute_counts, proportions):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f"{abs_count} ({proportion:.2f})", ha="center", va="bottom", fontsize=10)

def plot_token_counts(token_counts, save_path=None):
    """Plot a normalized bar graph of SMILES token counts with different colors, a legend, and counts on bars."""
    tokens = list(token_counts.keys())
    counts = list(token_counts.values())

    # Normalize counts
    normalized_counts = normalize_counts(counts)

    # Assign unique colors to each token
    colors = plt.cm.tab20(range(len(tokens)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(tokens, normalized_counts, color=colors)

    # Add counts on top of bars
    add_counts_to_bars(bars, counts, normalized_counts)

    plt.xlabel("Tokens")
    plt.ylabel("Proportion")
    plt.title("Normalized SMILES Token Counts")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add legend
    plt.legend(bars, tokens, title="Tokens", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("📊 Normalized token count plot saved to: %s", save_path)
    else:
        plt.show()

def plot_validity_graph(valid_smiles, invalid_smiles_count, save_path=None):
    """Plot a normalized bar graph showing the count of valid and invalid molecules with counts and proportions on bars."""
    valid_count = len(valid_smiles)
    absolute_counts = [valid_count, invalid_smiles_count]
    normalized_counts = normalize_counts(absolute_counts)

    labels = ["Valid", "Invalid"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, normalized_counts, color=["green", "red"])

    # Add counts and proportions on top of bars
    add_counts_to_bars(bars, absolute_counts, normalized_counts)

    plt.xlabel("Molecule Validity")
    plt.ylabel("Proportion")
    plt.title("Normalized Validity of SMILES Molecules")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path)
        logger.info("📊 Normalized validity graph saved to: %s", save_path)
    else:
        plt.show()

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

        # Count and plot token occurrences
        token_counts = count_smiles_tokens(tokenized_smiles)
        plot_save_path_tokens = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/Preprocessing/smiles_token_counts_normalized.png"
        plot_token_counts(token_counts, save_path=plot_save_path_tokens)

        # Avoid re-checking validity; use already validated SMILES
        valid_smiles = smiles_list
        invalid_smiles_count = len(smiles_list) - len(valid_smiles)
        plot_save_path_validity = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/Preprocessing/smiles_validity_counts_normalized.png"
        plot_validity_graph(valid_smiles, invalid_smiles_count, save_path=plot_save_path_validity)
