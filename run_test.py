"""Stage 3: Generate molecules with the RL-fine-tuned model and evaluate.
Loads: saved_model/lstm_finetuned_rl.keras + processed_data/CCAB.pkl
Saves: run_results/generated_smiles_CN.smi, run_results/generated_molecules_CN.png
"""

import os, pickle, numpy as np, tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Draw, QED, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "saved_model", "lstm_finetuned_rl.keras")
DATA_PATH = os.path.join(ROOT, "processed_data", "CCAB.pkl")
OUT_DIR = os.path.join(ROOT, "run_results")
os.makedirs(OUT_DIR, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

model = tf.keras.models.load_model(MODEL_PATH)
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
c2i = data["char_to_idx"]
i2c = data["idx_to_char"]
max_len = data["max_smiles_length"]
train_set = set("".join(s) for s in data["tokenized_smiles"])


def detok(ids):
    return "".join(
        i2c.get(i, "") for i in ids if i2c.get(i, "") not in ("<PAD>", "<UNK>")
    )


def generate(n=100, goal="CN", temperature=1.0, top_p=0.9):
    out = []
    start_tok = goal[0] if goal and goal[0] in c2i else "C"
    start = c2i.get(start_tok, 1)
    for _ in range(n):
        tokens = [start]
        for _ in range(max_len - 1):
            x = pad_sequences([tokens], maxlen=max_len - 1, padding="post", value=0)
            preds = model.predict(x, verbose=0)[0, len(tokens) - 1]
            logits = np.log(preds + 1e-8) / temperature
            probs = np.exp(logits)
            probs /= probs.sum()
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            cut = np.searchsorted(cum, top_p) + 1
            sel = order[:cut]
            p = probs[sel]
            p /= p.sum()
            nxt = int(np.random.choice(sel, p=p))
            tokens.append(nxt)
            if i2c.get(nxt) == "<PAD>":
                break
        out.append(detok(tokens))
    return out


def evaluate(smis, goal="CN"):
    valid, novel_valid, mols, qeds = [], [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m:
            valid.append(s)
            mols.append(m)
            try:
                qeds.append(QED.qed(m))
            except:
                qeds.append(0.0)
            if s not in train_set:
                novel_valid.append(s)
    div = 0.0
    if len(mols) > 1:
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
        s = 0
        c = 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                s += 1 - TanimotoSimilarity(fps[i], fps[j])
                c += 1
        div = s / c if c else 0
    n = len(smis)
    return {
        "Total": n,
        "Valid": len(valid),
        "Validity %": 100 * len(valid) / n if n else 0,
        "Novelty %": 100 * len(novel_valid) / n if n else 0,
        "Uniqueness %": 100 * len(set(valid)) / len(valid) if valid else 0,
        "Diversity": div,
        "Avg QED": float(np.mean(qeds)) if qeds else 0,
    }, valid


print("Generating 100 molecules with goal='CN'...")
smis = generate(n=100, goal="CN", temperature=1.0, top_p=0.9)

smi_path = os.path.join(OUT_DIR, "generated_smiles_CN.smi")
with open(smi_path, "w") as f:
    for s in smis:
        ok = "valid" if Chem.MolFromSmiles(s) else "invalid"
        f.write(f"{s}\t{ok}\n")
print(f"Wrote {smi_path}")

metrics, valid = evaluate(smis, goal="CN")
print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

if valid:
    mols = [Chem.MolFromSmiles(s) for s in valid[:30]]
    mols = [m for m in mols if m]
    if mols:
        img_path = os.path.join(OUT_DIR, "generated_molecules_CN.png")
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
        img.save(img_path)
        print(f"\nSaved {len(mols)} molecule images -> {img_path}")
