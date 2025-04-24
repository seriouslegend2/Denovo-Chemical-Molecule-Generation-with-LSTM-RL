# Denovo-Chemical-Molecule-Generation-with-LSTM-RL

A deep learning pipeline for de novo molecular generation using LSTM and reinforcement learning (REINFORCE), with support for fragment-based control and automated molecule evaluation.

---

## Overview

This project enables the generation of novel, valid, and drug-like molecules as SMILES strings, using a two-stage approach:
- **Stage 1:** Train an LSTM model on a large SMILES dataset for next-token prediction.
- **Stage 2:** Fine-tune the model with reinforcement learning (REINFORCE) using a custom reward function (validity, novelty, fragment presence, QED).

### Features:
- Robust SMILES preprocessing and tokenization  
- LSTM sequence modeling  
- RL fine-tuning with chemical property rewards  
- Top-p (nucleus) sampling for diverse molecule generation  
- Automated evaluation: validity, novelty, diversity, QED, uniqueness  
- Visualization of generated molecules  

---

## Directory Structure

```
.
├── dataset/                       # Raw .smi files
├── processed_data/
│   ├── preprocessing_step_1.smi  # Canonicalized SMILES
│   ├── tokenized_data.pkl        # Tokenized SMILES and dictionaries
│   └── split_data/               # Data splits for training
├── saved_model/Orig/
│   ├── lstm_generator.h5         # Pretrained LSTM model
│   └── lstm_finetuned_rl.h5      # RL fine-tuned model
├── results/
│   ├── rewards_log.csv           # RL reward log
│   └── reward_plot.png           # Reward progression
├── generated_smiles_CN.smi       # Generated SMILES with validity
├── generated_molecules_CN.png    # Image grid of valid molecules
├── preprocess.py                 # Data preprocessing script
├── tokenize_smiles.py            # SMILES tokenization
├── split_data_tokens.py          # Data splitting
├── train_model.py                # LSTM model training
├── RLfinetune.py                 # RL fine-tuning script
├── testRLLSTM.py                 # Generation and evaluation
├── requirements.txt              # Python dependencies
└── RL-LSTM_model.ipynb           # Jupyter notebook (pipeline demo)
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.10  
- TensorFlow 2.19.0  
- RDKit 2024.9.6  
- NumPy  
- Pandas  
- Matplotlib  

---

## Pipeline Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```
- Output: `processed_data/preprocessing_step_1.smi`

### 2. Tokenization

```bash
python tokenize_smiles.py
```
- Output: `processed_data/tokenized_data.pkl`

### 3. Data Splitting

```bash
python split_data_tokens.py
```
- Output: 10 splits in `processed_data/split_data/`

### 4. LSTM Model Training

```bash
python train_model.py
```
- Output: `saved_model/Orig/lstm_generator.h5`

### 5. Reinforcement Learning Fine-Tuning

```bash
python RLfinetune.py
```
- Output: `saved_model/Orig/lstm_finetuned_rl.h5`, reward logs, and plot

### 6. Molecule Generation & Evaluation

```bash
python testRLLSTM.py
```
- Outputs:
  - `generated_smiles_CN.smi`: Generated SMILES with validity flags
  - `generated_molecules_CN.png`: Grid image of valid molecules
  - Console: Evaluation metrics (validity, novelty, diversity, QED, uniqueness)

---

## Scripts Description

| Script                | Purpose                                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------|
| `preprocess.py`       | Canonicalizes and aggregates SMILES from raw data                                       |
| `tokenize_smiles.py`  | Tokenizes SMILES using regex and builds vocabularies                                    |
| `split_data_tokens.py`| Shuffles and splits data for robust training                                            |
| `train_model.py`      | Trains LSTM with sparse categorical cross-entropy                                       |
| `RLfinetune.py`       | Fine-tunes LSTM with RL (custom reward: validity, novelty, fragment, QED)               |
| `testRLLSTM.py`       | Generates molecules, computes metrics, and visualizes results                           |

---

## Evaluation Results

| Metric              | Value   |
|---------------------|---------|
| Molecules Generated | 100     |
| **Valid Molecules** | **94**  |
| Validity (%)        | 94.0    |
| Novelty (%)         | 100.0   |
| Diversity           | 0.84    |
| Avg QED             | 0.62    |
| Uniqueness (%)      | 100.0   |

**Summary:** Out of 100 generated molecules, 94 were chemically valid, all were novel, and the set exhibited high diversity and uniqueness. The average QED (quantitative estimate of drug-likeness) was 0.62, indicating good drug-like properties.

---

## Visualization

we have file in codes folder please look over there

---

## Citation

If you use this codebase or results, please cite this repository and the relevant RDKit and TensorFlow libraries.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer.

---

**Note:** This pipeline is modular and can be adapted for different chemical fragment targets or reward functions as needed.
