import os
import numpy as np
import tensorflow as tf
import pickle
import re
from rdkit import Chem
from rdkit.Chem import Draw, QED, AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load model and tokenizer
MODEL_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_finetuned_rl.h5"
DATA_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/tokenized_data.pkl"

num_samples=100

model = tf.keras.models.load_model(MODEL_PATH)
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

token_to_idx = data["char_to_idx"]
idx_to_token = data["idx_to_char"]
max_length = data["max_smiles_length"]
training_smiles_set = set(data.get("smiles", []))

# Tokenizer
PATTERN = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|[a-z]|@{1,2}|#|=|\\|\/|\+|-|\(|\)|\d+)"
def tokenize_smiles(s): return re.findall(PATTERN, s)
def detokenize(indices): return "".join([idx_to_token.get(i, "") for i in indices if idx_to_token.get(i, "") not in ["<PAD>", "<UNK>"]])

# Fragment-controlled generation with nucleus (top-p) sampling
def generate_smiles_with_goal(goal_fragment="CH", num_samples=100, temperature=1.0, top_p=0.9):
    generated = []
    start_token = goal_fragment[0] if goal_fragment[0] in token_to_idx else "C"
    start_idx = token_to_idx.get(start_token, 1)

    for _ in tqdm(range(num_samples)):
        tokens = [start_idx]
        for _ in range(max_length - 1):
            padded = pad_sequences([tokens], maxlen=max_length - 1, padding="post", value=0)
            preds = model.predict(padded, verbose=0)[0, len(tokens) - 1]
            preds = np.log(preds + 1e-8) / temperature  # Apply temperature scaling
            preds = np.exp(preds) / np.sum(np.exp(preds))  # Re-normalize probabilities

            # Nucleus (top-p) sampling
            sorted_indices = np.argsort(preds)[::-1]
            cumulative_probs = np.cumsum(preds[sorted_indices])
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            selected_indices = sorted_indices[:cutoff_index + 1]
            selected_probs = preds[selected_indices]
            selected_probs /= np.sum(selected_probs)  # Normalize selected probabilities
            next_token = np.random.choice(selected_indices, p=selected_probs)

            tokens.append(next_token)
            if idx_to_token.get(next_token) == "<PAD>":
                break
        smiles = detokenize(tokens)
        generated.append(smiles)
    return generated

# Evaluation metrics with uniqueness calculation
def compute_metrics(smiles_list, goal_fragment="CCl"):
    valid_smiles = []
    novel = 0
    match_goal = 0
    mols = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid_smiles.append(s)
            mols.append(mol)
            if s not in training_smiles_set:
                novel += 1
            if goal_fragment in s:
                match_goal += 1

    # Diversity (avg pairwise Tanimoto)
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    diversity = 0
    count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            diversity += 1 - TanimotoSimilarity(fps[i], fps[j])
            count += 1
    diversity_score = diversity / count if count else 0

    # QED
    qed_scores = [QED.qed(m) for m in mols]
    avg_qed = np.mean(qed_scores) if qed_scores else 0

    # Uniqueness
    unique_valid_smiles = set(valid_smiles)
    uniqueness = 100 * len(unique_valid_smiles) / len(valid_smiles) if valid_smiles else 0

    metrics = {
        "Total": len(smiles_list),
        "Valid": len(valid_smiles),
        "Validity %": 100 * len(valid_smiles) / len(smiles_list),
        "Novelty %": 100 * novel / len(valid_smiles) if valid_smiles else 0,
        "Diversity": diversity_score,
        "Avg QED": avg_qed,
        "Uniqueness %": uniqueness,
    }
    return metrics, valid_smiles

# Save generated SMILES to file with validity
def save_generated_smiles(smiles_list, file_path="generated_smiles_CN.smi"):
    with open(file_path, "w") as f:
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            validity = "✔️" if mol else "❌"
            f.write(f"{s}\t{validity}\n")
    print(f"💾 Saved {len(smiles_list)} generated SMILES to: {file_path}")

# Save all valid molecule images
def save_molecule_images(valid_smiles_list, save_path="generated_molecules_CN.png", mols_per_row=5):
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles_list if Chem.MolFromSmiles(s)]
    if not mols:
        print("⚠️ No valid molecules to draw.")
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(250, 250))
    img.save(save_path)
    print(f"🖼️ Saved {len(mols)} valid molecule images to: {save_path}")

# Show a few molecules inline (optional)
def show_molecules(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    img = Draw.MolsToGridImage(mols[:20], molsPerRow=5, subImgSize=(250, 250))
    img.show()

# ==== Run Generation & Evaluation ====
generated_smiles = generate_smiles_with_goal(goal_fragment="CN", num_samples=num_samples, temperature=1.0, top_p=0.9)  # Use top-p sampling
metrics, valid_smiles = compute_metrics(generated_smiles, goal_fragment="CH")

# Save SMILES and image
save_generated_smiles(generated_smiles, "generated_smiles_CN.smi")
save_molecule_images(valid_smiles, "generated_molecules_CN.png")

print("\n🔬 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
# Optional: show molecules inline (only top 20)
show_molecules(valid_smiles)
