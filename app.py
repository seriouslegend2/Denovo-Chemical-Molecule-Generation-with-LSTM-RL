"""Flask dashboard for the de novo molecule generator.

Run: python3 app.py
Open: http://localhost:5000
"""

import os, io, json, base64, pickle, numpy as np, tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from rdkit import Chem
from rdkit.Chem import Draw, QED, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "processed_data", "CCAB.pkl")
MODELS = {
    "rl": os.path.join(ROOT, "saved_model", "lstm_finetuned_rl.keras"),
    "base": os.path.join(ROOT, "saved_model", "lstm_generator.keras"),
}

for g in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

print("Loading tokenizer and models...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
C2I = data["char_to_idx"]
I2C = data["idx_to_char"]
MAX_LEN = data["max_smiles_length"]
TRAIN_SET = set("".join(s) for s in data["tokenized_smiles"])

_models = {
    k: tf.keras.models.load_model(p) for k, p in MODELS.items() if os.path.exists(p)
}
print(f"Loaded models: {list(_models.keys())}   vocab={len(C2I)}   max_len={MAX_LEN}")


def detok(ids):
    return "".join(
        I2C.get(i, "") for i in ids if I2C.get(i, "") not in ("<PAD>", "<UNK>")
    )


def sample_one(model_key="rl", start="C", temperature=1.0, top_p=0.9):
    """Generate a single SMILES string."""
    model = _models[model_key]
    start_idx = C2I.get(start, 1)
    tokens = [start_idx]
    for _ in range(MAX_LEN - 1):
        x = pad_sequences([tokens], maxlen=MAX_LEN - 1, padding="post", value=0)
        preds = model.predict(x, verbose=0)[0, len(tokens) - 1]
        logits = np.log(preds + 1e-9) / max(temperature, 1e-3)
        probs = np.exp(logits)
        probs /= probs.sum()
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        sel = order[:cut]
        p = probs[sel]
        p /= p.sum()
        nxt = int(np.random.choice(sel, p=p))
        tokens.append(nxt)
        if I2C.get(nxt) == "<PAD>":
            break
    return detok(tokens)


def generate(model_key="rl", n=20, start="C", temperature=1.0, top_p=0.9):
    return [sample_one(model_key, start, temperature, top_p) for _ in range(n)]


def mol_to_png_b64(mol, size=(220, 220)):
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def evaluate(smis):
    mols, valid, novel_valid, qeds = [], [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m:
            mols.append((s, m))
            valid.append(s)
            try:
                qeds.append(QED.qed(m))
            except:
                qeds.append(0.0)
            if s not in TRAIN_SET:
                novel_valid.append(s)
    div = 0.0
    if len(mols) > 1:
        fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 1024) for _, m in mols
        ]
        s = c = 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                s += 1 - TanimotoSimilarity(fps[i], fps[j])
                c += 1
        div = s / c if c else 0
    n = len(smis)
    return {
        "total": n,
        "valid": len(valid),
        "validity_pct": round(100 * len(valid) / n, 1) if n else 0,
        "novelty_pct": round(100 * len(novel_valid) / n, 1) if n else 0,
        "uniqueness_pct": round(100 * len(set(valid)) / len(valid), 1) if valid else 0,
        "diversity": round(div, 3),
        "avg_qed": round(float(np.mean(qeds)), 3) if qeds else 0,
    }, mols


app = Flask(__name__)


@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=list(_models.keys()),
        vocab=sorted([t for t in C2I if t not in ("<PAD>", "<UNK>")]),
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    j = request.get_json(force=True)
    model_key = j.get("model", "rl")
    n = int(j.get("n", 20))
    start = j.get("start", "C") or "C"
    temperature = float(j.get("temperature", 1.0))
    top_p = float(j.get("top_p", 0.9))
    goal = j.get("goal_fragment", "").strip()

    if model_key not in _models:
        return jsonify(error=f"unknown model '{model_key}'"), 400
    n = max(1, min(n, 100))

    def event(obj):
        return json.dumps(obj) + "\n"

    @stream_with_context
    def stream():
        yield event({"type": "start", "total": n})
        smis, mols = [], []
        for i in range(n):
            s = sample_one(model_key, start, temperature, top_p)
            smis.append(s)
            m = Chem.MolFromSmiles(s)
            if m:
                mols.append((s, m))
            item = {
                "smiles": s,
                "valid": m is not None,
                "novel": m is not None and s not in TRAIN_SET,
                "contains_goal": bool(goal) and (goal in s),
                "qed": round(QED.qed(m), 3) if m else None,
                "img": mol_to_png_b64(m) if m else None,
            }
            yield event(
                {
                    "type": "item",
                    "index": i + 1,
                    "total": n,
                    "percent": round(100 * (i + 1) / n, 1),
                    "item": item,
                }
            )
        # final metrics computed over full set
        metrics, _ = evaluate(smis)
        yield event({"type": "done", "metrics": metrics})

    return Response(stream(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
