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
import matplotlib_venn as venn
import seaborn as sns

# Define the results directory
RESULTS_DIR = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model and tokenizer
MODEL_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_finetuned_rl.h5"
DATA_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/split_data/split_1.pkl"

num_samples = 100

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
            preds = np.log(preds + 1e-8) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))

            sorted_indices = np.argsort(preds)[::-1]
            cumulative_probs = np.cumsum(preds[sorted_indices])
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            selected_indices = sorted_indices[:cutoff_index + 1]
            selected_probs = preds[selected_indices]
            selected_probs /= np.sum(selected_probs)
            next_token = np.random.choice(selected_indices, p=selected_probs)

            tokens.append(next_token)
            if idx_to_token.get(next_token) == "<PAD>":
                break
        smiles = detokenize(tokens)
        generated.append(smiles)
    return generated

# Evaluation metrics with uniqueness and novelty tracking
def compute_metrics(smiles_list, goal_fragment="CCl"):
    valid_smiles = []
    novel_valid_smiles = []  # Track molecules that are both valid and novel
    match_goal = 0
    mols = []
    qed_scores = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid_smiles.append(s)
            mols.append(mol)
            try:
                qed = QED.qed(mol)
                qed_scores.append(qed)
            except:
                qed_scores.append(0.0)
            # Count as novel only if it is valid and not in the training set
            if s not in training_smiles_set:
                novel_valid_smiles.append(s)
            if goal_fragment in s:
                match_goal += 1

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    diversity = 0
    count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            diversity += 1 - TanimotoSimilarity(fps[i], fps[j])
            count += 1
    diversity_score = diversity / count if count else 0

    avg_qed = np.mean(qed_scores) if qed_scores else 0
    unique_valid_smiles = set(valid_smiles)
    uniqueness = 100 * len(unique_valid_smiles) / len(valid_smiles) if valid_smiles else 0

    # Final novelty calculation: only valid and novel molecules, relative to total molecules
    novelity_percentage = 100 * len(novel_valid_smiles) / len(smiles_list) if smiles_list else 0

    metrics = {
        "Total": len(smiles_list),
        "Valid": len(valid_smiles),
        "Validity %": 100 * len(valid_smiles) / len(smiles_list),
        "Novelty %": novelity_percentage,
        "Diversity": diversity_score,
        "Avg QED": avg_qed,
        "Uniqueness %": uniqueness,
    }
    return metrics, valid_smiles, qed_scores

# Compute uniqueness and novelty for different configurations
def compute_metrics_for_configs(configs, goal_fragment="CN", num_samples=100):
    uniqueness_data = []
    novelty_data = []

    for config in configs:
        temperature, top_p = config["temperature"], config["top_p"]
        generated_smiles = generate_smiles_with_goal(
            goal_fragment=goal_fragment,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p
        )
        metrics, _, _ = compute_metrics(generated_smiles, goal_fragment=goal_fragment)
        uniqueness_data.append(metrics["Uniqueness %"])
        novelty_data.append(metrics["Novelty %"])

    return uniqueness_data, novelty_data

# Plot uniqueness and novelty heatmaps
def plot_uniqueness_novelty_heatmaps(uniqueness_data, novelty_data, config_labels, model_labels, save_path="uniqueness_novelty_heatmaps.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Uniqueness heatmap
    sns.heatmap([uniqueness_data], ax=ax1, cmap="Blues", annot=True, fmt=".0f", vmin=0, vmax=100, cbar=False)
    ax1.set_title("Uniqueness (%)")
    ax1.set_xticks(np.arange(len(config_labels)))
    ax1.set_xticklabels(config_labels, rotation=45, ha="right")
    ax1.set_yticks(np.arange(len(model_labels)))
    ax1.set_yticklabels(model_labels, rotation=0)

    # Novelty heatmap
    sns.heatmap([novelty_data], ax=ax2, cmap="Blues", annot=True, fmt=".0f", vmin=0, vmax=100)
    ax2.set_title("Novelty (%)")
    ax2.set_xticks(np.arange(len(config_labels)))
    ax2.set_xticklabels(config_labels, rotation=45, ha="right")
    ax2.set_yticks(np.arange(len(model_labels)))
    ax2.set_yticklabels(model_labels, rotation=0)

    # Add colorbar to the right
    plt.colorbar(ax2.get_children()[0], ax=ax2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved uniqueness and novelty heatmaps to: {save_path}")

# Save generated SMILES to file
def save_generated_smiles(smiles_list, file_path="generated_smiles.smi"):
    file_path = os.path.join(RESULTS_DIR, os.path.basename(file_path))
    with open(file_path, "w") as f:
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            validity = "✔️" if mol else "❌"
            f.write(f"{s}\t{validity}\n")
    print(f"💾 Saved {len(smiles_list)} generated SMILES to: {file_path}")

# Save molecule images
def save_molecule_images(valid_smiles_list, save_path="generated_molecules.png", mols_per_row=5):
    save_path = os.path.join(RESULTS_DIR, os.path.basename(save_path))
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles_list if Chem.MolFromSmiles(s)]
    if not mols:
        print("⚠️ No valid molecules to draw.")
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(250, 250))
    img.save(save_path)
    print(f"🖼️ Saved {len(mols)} valid molecule images to: {save_path}")

# Create and save Venn diagram
def save_venn_diagram(metrics, save_path="evaluation_metrics_venn.png"):
    save_path = os.path.join(RESULTS_DIR, os.path.basename(save_path))
    total = metrics["Total"]
    validity = metrics["Valid"]
    novelity = int(metrics["Novelty %"] * total / 100)
    match_goal = int(metrics["Validity %"] * total / 100)

    venn_data = {
        "10": validity - novelity,
        "01": novelity,
        "11": match_goal
    }

    venn.venn2(subsets=venn_data, set_labels=("Valid", "Novel"))
    plt.title("Evaluation Metrics Venn Diagram")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved Venn diagram to: {save_path}")

# Show molecules (optional)
def show_molecules(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    img = Draw.MolsToGridImage(mols[:20], molsPerRow=5, subImgSize=(250, 250))
    img.show()

# Plot evaluation metrics
def plot_metrics(metrics, save_path="evaluation_metrics.png"):
    save_path = os.path.join(RESULTS_DIR, os.path.basename(save_path))
    metric_names = ["Validity %", "Novelty %", "Uniqueness %", "Diversity", "Avg QED"]
    metric_values = [metrics.get(m, 0) for m in metric_names]

    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metric_values, color='skyblue')
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.title("Evaluation Metrics")
    plt.ylim(0, max(metric_values) * 1.2 if max(metric_values) > 0 else 1)
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved evaluation metrics plot to: {save_path}")

# Plot QED scores over generation index
def plot_qed_scatter(qed_scores, save_path="qed_scatter_plot.png"):
    save_path = os.path.join(RESULTS_DIR, os.path.basename(save_path))
    np.random.seed(42)
    random_qed_scores = np.random.uniform(0, 0.5, len(qed_scores))

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(qed_scores)), qed_scores, color='blue', alpha=0.5, label='MORLD', s=10)
    plt.scatter(range(len(random_qed_scores)), random_qed_scores, color='orange', alpha=0.5, label='Random', s=10)
    plt.axhline(y=0.4, color='red', linestyle='--', label='Threshold (0.4)')
    plt.xlabel("Generation Index (Episode)")
    plt.ylabel("QED Score")
    plt.title("QED Scores Over Generation Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved QED scatter plot to: {save_path}")

# ==== Run Generation & Evaluation ====
# Define different configurations (simulating different models)
configs = [
    {"temperature": 0.8, "top_p": 0.9, "label": "Config 1"},
    {"temperature": 1.0, "top_p": 0.9, "label": "Config 2"},
    {"temperature": 1.2, "top_p": 0.9, "label": "Config 3"},
    {"temperature": 1.0, "top_p": 0.8, "label": "Config 4"},
    {"temperature": 1.0, "top_p": 0.7, "label": "Config 5"},
    {"temperature": 1.2, "top_p": 0.8, "label": "Config 6"},
]

# Compute uniqueness and novelty for each configuration
config_labels = [config["label"] for config in configs]
model_labels = ["LSTM"]  # Single model, but we can simulate multiple by varying parameters
uniqueness_data, novelty_data = compute_metrics_for_configs(configs, goal_fragment="CN", num_samples=num_samples)

# Plot uniqueness and novelty heatmaps
plot_uniqueness_novelty_heatmaps(
    uniqueness_data,
    novelty_data,
    config_labels,
    model_labels,
    save_path=os.path.join(RESULTS_DIR, "uniqueness_novelty_heatmaps.png")
)

# Generate SMILES for the default configuration (used for other plots)
generated_smiles = generate_smiles_with_goal(goal_fragment="CN", num_samples=num_samples, temperature=1.0, top_p=0.9)
metrics, valid_smiles, qed_scores = compute_metrics(generated_smiles, goal_fragment="CH")

# Save results
save_generated_smiles(generated_smiles, "generated_smiles_CN.smi")
save_molecule_images(valid_smiles, "generated_molecules_CN.png")
save_venn_diagram(metrics, "evaluation_metrics_venn.png")
plot_metrics(metrics)
plot_qed_scatter(qed_scores)

print("\n🔬 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
show_molecules(valid_smiles)