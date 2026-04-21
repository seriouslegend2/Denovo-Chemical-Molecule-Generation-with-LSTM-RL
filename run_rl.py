"""Stage 2 (fixed): Proper REINFORCE policy-gradient RL fine-tune.

Fix vs the original: uses log-prob of SAMPLED tokens weighted by (reward - baseline),
which is the correct REINFORCE objective. The original multiplied cross-entropy by
reward with a negative sign, which actually *degrades* the policy on high-reward
episodes (mode collapse).
"""

import os, pickle, csv, numpy as np, tensorflow as tf
from rdkit import Chem
from rdkit.Chem import QED
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_IN = os.path.join(ROOT, "saved_model", "lstm_generator.keras")
MODEL_OUT = os.path.join(ROOT, "saved_model", "lstm_finetuned_rl.keras")
DATA_PATH = os.path.join(ROOT, "processed_data", "CCAB.pkl")
LOG_PATH = os.path.join(ROOT, "run_results", "reward_metrics_log.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

for g in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

model = tf.keras.models.load_model(MODEL_IN)
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
c2i = data["char_to_idx"]
i2c = data["idx_to_char"]
max_len = data["max_smiles_length"]
train_set = set("".join(s) for s in data["tokenized_smiles"])
PAD = c2i.get("<PAD>", 0)

FRAGMENTS = ["CN", "C(=O)O", "CC"]


def detok(ids):
    return "".join(
        i2c.get(i, "") for i in ids if i2c.get(i, "") not in ("<PAD>", "<UNK>")
    )


def reward_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    v = 5.0
    frag = sum(1.0 for f in FRAGMENTS if f in smi)
    nov = 10.0 if smi not in train_set else 0.0
    try:
        q = QED.qed(m)
    except:
        q = 0.0
    return v + frag + nov + q, v, frag, nov, q


def sample_with_logprobs(start="C"):
    sid = c2i.get(start, 1)
    tokens = [sid]
    for _ in range(max_len - 1):
        x = pad_sequences([tokens], maxlen=max_len - 1, padding="post", value=0)
        preds = model.predict(x, verbose=0)[0, len(tokens) - 1]
        preds = np.clip(preds, 1e-9, 1.0)
        preds = preds / preds.sum()
        nxt = int(np.random.choice(len(preds), p=preds))
        tokens.append(nxt)
        if i2c.get(nxt) == "<PAD>":
            break
    return detok(tokens), tokens


opt = tf.keras.optimizers.Adam(learning_rate=1e-4)


@tf.function
def policy_grad_step(X, actions, mask, advantage):
    with tf.GradientTape() as tape:
        probs = model(X, training=True)  # (1, T, V)
        probs = tf.clip_by_value(probs, 1e-9, 1.0)
        log_probs = tf.math.log(probs)
        # gather log-prob of actually sampled token at each position
        idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        chosen = tf.gather_nd(log_probs[0], idx)  # (T,)
        chosen = chosen * mask
        loss = -advantage * tf.reduce_sum(chosen) / tf.reduce_sum(mask)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(
        [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    )
    return loss


EPISODES = 100
baseline = 0.0
alpha = 0.9  # exponential moving-average baseline
with open(LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow(
        [
            "Episode",
            "Validity",
            "Fragments",
            "Novelty",
            "QED",
            "TotalReward",
            "Advantage",
            "SMILES",
        ]
    )

valid_hits = 0
for ep in range(EPISODES):
    smi, ids = sample_with_logprobs()
    total, v, fr, nov, q = reward_of(smi)
    advantage = total - baseline
    baseline = alpha * baseline + (1 - alpha) * total

    inp = ids[:-1]
    tgt = ids[1:]
    X = pad_sequences([inp], maxlen=max_len - 1, padding="post", value=PAD)
    actions = np.array(tgt + [PAD] * (max_len - 1 - len(tgt)), dtype=np.int32)
    mask = (actions != PAD).astype(np.float32)
    mask[: len(tgt)] = 1.0
    loss = policy_grad_step(
        tf.constant(X, dtype=tf.int32),
        tf.constant(actions[None, :], dtype=tf.int32)[0],
        tf.constant(mask, dtype=tf.float32),
        tf.constant(float(advantage), dtype=tf.float32),
    )

    if v > 0:
        valid_hits += 1
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([ep, v, fr, nov, q, total, advantage, smi])
    if ep % 10 == 0 or ep == EPISODES - 1:
        print(
            f"[Ep {ep:3d}] R={total:5.2f} adv={advantage:+5.2f} base={baseline:5.2f} "
            f"valid_rate={valid_hits / (ep + 1):.2f} loss={float(loss):+.4f} smi='{smi}'"
        )

model.save(MODEL_OUT)
print(f"\nRL-fine-tuned model saved -> {MODEL_OUT}")
print(f"Reward log -> {LOG_PATH}")
