#!/usr/bin/env bash
# Start the MORLD dashboard on a fresh machine.
# Usage:  bash start.sh
set -e
cd "$(dirname "$0")"

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
  echo ">>> Creating virtualenv (.venv)"
  python3 -m venv .venv
fi

# 2. Activate + install deps (idempotent)
source .venv/bin/activate
echo ">>> Installing dependencies (first run takes a few minutes)"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# 3. Train models if not already present
if [ ! -f "saved_model/lstm_generator.keras" ]; then
  echo ">>> Training base LSTM (~30 s on GPU, ~2 min on CPU)"
  python run_train.py
fi
if [ ! -f "saved_model/lstm_finetuned_rl.keras" ]; then
  echo ">>> RL fine-tuning (~2 min)"
  python run_rl.py
fi

# 4. Launch the dashboard
echo ">>> Starting dashboard at http://localhost:5000"
exec python app.py
