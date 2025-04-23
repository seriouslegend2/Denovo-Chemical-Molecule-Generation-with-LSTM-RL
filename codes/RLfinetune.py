import os
import numpy as np
import tensorflow as tf
import pickle
import re
from rdkit import Chem
from rdkit.Chem import QED
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv  # Add import for CSV
import matplotlib.pyplot as plt  # Add import for plotting

# ✅ Load Model and Tokenizer
MODEL_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_generator.h5"
DATA_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/tokenized_data.pkl"

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

# ✅ Compute Reward (validity + fragment(s) + novelty + QED)
def compute_reward(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    reward = 5.0  # Validity

    for frag in FRAGMENTS:
        if frag in smiles:
            reward += 1.0

    if is_novel(smiles):
        reward += 10.0

    try:
        qed_score = QED.qed(mol)
        reward += qed_score
    except:
        pass

    return reward

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
        reward = compute_reward(smiles)

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

# ✅ Training Loop (Clean Logging)
EPOCHS = 100
total_reward_history = []

# Initialize CSV file for logging rewards
rewards_log_path = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/rewards_log.csv"
os.makedirs(os.path.dirname(rewards_log_path), exist_ok=True)
with open(rewards_log_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Reward", "Total Reward"])  # Add "Total Reward" to header

for episode in range(EPOCHS):
    smiles, reward, loss = train_step()
    total_reward_history.append(reward)
    total_reward = sum(total_reward_history)  # Compute total reward

    # Log reward and total reward to CSV
    with open(rewards_log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward, total_reward])  # Log total reward

    print(f"[Ep {episode}]🎯Reward:{reward:.2f}|Loss:{loss:.4f}|💊SMILES: {smiles}")
    print(f"🎯Total Reward Accumulated: {total_reward:.4f}")

# ✅ Save Fine-Tuned Model
model.save("/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_finetuned_rl.h5")
print("Fine-tuned model saved.")

# ✅ Plot Rewards
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), total_reward_history, label="Reward per Epoch", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.title("Reward Progression Over Epochs")
plt.legend()
plt.grid(True)
plot_path = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/reward_plot.png"
plt.savefig(plot_path)  # Save the plot as an image
plt.show()
print(f"Reward plot saved to {plot_path}.")
