"""Stage 1: Train base LSTM SMILES language model on processed_data/CCAB.pkl.
Saves: saved_model/lstm_generator.keras
"""

import os, pickle, time, numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "processed_data", "CCAB.pkl")
MODEL_DIR = os.path.join(ROOT, "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_generator.keras")

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

print(f"Loading {DATA_PATH}")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
tok = data["tokenized_smiles"]
c2i = data["char_to_idx"]
max_len = data["max_smiles_length"]
vocab = len(c2i)
print(f"  Samples: {len(tok)}  Vocab: {vocab}  MaxLen: {max_len}")

unk = c2i.get("<UNK>", 1)
X = pad_sequences(
    [[c2i.get(t, unk) for t in s] for s in tok], maxlen=max_len, padding="post"
)
Y = X[:, 1:]
X = X[:, :-1]
split = int(0.9 * len(X))
Xtr, Xte, Ytr, Yte = X[:split], X[split:], Y[:split], Y[split:]
print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

model = Sequential(
    [
        Input(shape=(max_len - 1,)),
        Embedding(input_dim=vocab, output_dim=128),
        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        Dense(vocab, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

EPOCHS = 5
BATCH = 128
t0 = time.time()
model.fit(
    Xtr, Ytr, validation_data=(Xte, Yte), batch_size=BATCH, epochs=EPOCHS, verbose=2
)
print(f"Training done in {time.time() - t0:.1f}s")

model.save(MODEL_PATH)
print(f"Saved model -> {MODEL_PATH}")
