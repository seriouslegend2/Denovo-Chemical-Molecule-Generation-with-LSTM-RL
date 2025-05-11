import os
import numpy as np
import tensorflow as tf
import pickle
import re
from rdkit import Chem
from rdkit.Chem import QED
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import matplotlib.pyplot as plt

# ✅ Load Model and Tokenizer
MODEL_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_generator.h5"
DATA_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/split_data/split_1.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

token_to_idx = data["char_to_idx"]
idx_to_token = data["idx_to_char"]
max_length = data["max_smiles_length"]
training_smiles_set = set(data.get("smiles", []))

vocab_size = len(token_to_idx)

# ✅ Tokenizer
PATTERN = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|[a-z]|@{1,2}|#|=|\\|\/|\+|-|\(|\)|\d+)"
def tokenize_smiles(s): return re.findall(PATTERN, s)
def detokenize(indices):
    tokens = [idx_to_token.get(i, "") for i in indices if idx_to_token.get(i, "") not in ["<PAD>", "<UNK>"]]
    return "".join(tokens)

# ✅ Reward Function Helpers
def is_valid(s): return Chem.MolFromSmiles(s) is not None
def is_novel(s): return s not in training_smiles_set

# ✅ Desired Fragments (Multiple)
FRAGMENTS = ["CN", "C(=O)O", "CC"]

# ✅ Initialize CSV file for logging reward metrics
metrics_log_path = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Training/reward_metrics_log.csv"
os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
with open(metrics_log_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Validity", "Fragments", "Novelty", "QED", "Total Reward"])

# ✅ Compute Reward
def compute_reward(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    validity_reward = 5.0
    fragment_reward = 0.0
    novelty_reward = 0.0
    qed_score = 0.0

    for frag in FRAGMENTS:
        if frag in smiles:
            fragment_reward += 1.0

    if is_novel(smiles):
        novelty_reward = 10.0

    try:
        qed_score = QED.qed(mol)
    except:
        qed_score = 0.0

    total_reward = validity_reward + fragment_reward + novelty_reward + qed_score
    return total_reward, validity_reward, fragment_reward, novelty_reward, qed_score

# ✅ SMILES Sampler
def sample_smiles(start_token="C"):
    tokens = [token_to_idx.get(start_token, 1)]
    for _ in range(max_length - 1):
        padded = pad_sequences([tokens], maxlen=max_length - 1, padding="post", value=0)
        preds = model.predict(padded, verbose=0)[0, len(tokens) - 1]
        preds = preds / np.sum(preds)
        next_token = np.random.choice(len(preds), p=preds)
        tokens.append(next_token)
        if idx_to_token.get(next_token) == "<PAD>":
            break
    return detokenize(tokens), tokens

# ✅ REINFORCE Training Step
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

def train_step():
    with tf.GradientTape() as tape:
        smiles, token_ids = sample_smiles()
        reward, _, _, _, _ = compute_reward(smiles)

        input_seq = token_ids[:-1]
        target_seq = token_ids[1:]

        X = pad_sequences([input_seq], maxlen=max_length - 1, padding="post", value=0)
        y = pad_sequences([target_seq], maxlen=max_length - 1, padding="post", value=0)

        logits = model(X, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        loss = tf.reduce_mean(loss)

        total_loss = -reward * loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return smiles, reward, total_loss.numpy()

# ✅ Training Loop
EPOCHS = 100
total_reward_history = []
cumulative_validity = []
cumulative_fragments = []
cumulative_novelty = []
cumulative_qed = []

for episode in range(EPOCHS):
    smiles, reward, loss = train_step()
    total_reward, validity, fragments, novelty, qed = compute_reward(smiles)
    total_reward_history.append(total_reward)

    cumulative_validity.append(validity if not cumulative_validity else cumulative_validity[-1] + validity)
    cumulative_fragments.append(fragments if not cumulative_fragments else cumulative_fragments[-1] + fragments)
    cumulative_novelty.append(novelty if not cumulative_novelty else cumulative_novelty[-1] + novelty)
    cumulative_qed.append(qed if not cumulative_qed else cumulative_qed[-1] + qed)

    with open(metrics_log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, validity, fragments, novelty, qed, total_reward])

    print(f"[Ep {episode}]🎯Reward:{total_reward:.2f}|Loss:{loss:.4f}|💊SMILES: {smiles}")
    print(f"🎯Cumulative Validity: {cumulative_validity[-1]:.2f} | Cumulative Fragments: {cumulative_fragments[-1]:.2f} | "
          f"Cumulative Novelty: {cumulative_novelty[-1]:.2f} | Cumulative QED: {cumulative_qed[-1]:.2f}")

# ✅ Save Fine-Tuned Model
model.save("/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_finetuned_rl.h5")
print("Fine-tuned model saved.")

# ✅ Plot Cumulative Metrics
metrics = {
    "Validity": cumulative_validity,
    "Fragments": cumulative_fragments,
    "Novelty": cumulative_novelty,
    "QED": cumulative_qed,
    "Total Reward": np.cumsum(total_reward_history)
}

for metric_name, metric_values in metrics.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCHS), metric_values, label=f"Cumulative {metric_name}", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Cumulative {metric_name} Progression Over Epochs")
    plt.legend()
    plt.grid(True)
    plot_path = f"/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/{metric_name.lower()}_cumulative_plot.png"
    plt.savefig(plot_path)
    plt.close()  # Close figure to avoid memory issues
    print(f"{metric_name} cumulative plot saved to {plot_path}.")