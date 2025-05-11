import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define the path to the SMILES file
SMILES_FILE_PATH = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Results/generated_smiles_CN.smi"

# Read SMILES from the file
with open(SMILES_FILE_PATH, "r") as file:
    smiles_list = [line.strip() for line in file if line.strip()]  # Remove empty lines

# Filter valid SMILES (✔️-marked ones)
valid_smiles = [smiles for smiles in smiles_list if "✔️" in smiles]
# Remove the ✔️ marker for processing
valid_smiles = [smiles.replace(" ✔️", "") for smiles in valid_smiles]

# Calculate molecular weights
molecular_weights = []
for smiles in valid_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mw = Descriptors.MolWt(mol)
        molecular_weights.append(mw)
    else:
        print(f"Invalid SMILES: {smiles}")

# Create a single figure with one set of axes
os.makedirs("/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Results/", exist_ok=True)
plt.figure(figsize=(10, 6))

# Plot histogram with transparency
plt.hist(molecular_weights, bins=20, edgecolor='black', color='skyblue', alpha=0.5, density=True, label='Histogram')

# Overlay KDE plot
sns.kdeplot(data=molecular_weights, color="darkblue", lw=2, label='KDE')

# Customize the plot
plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("Density")
plt.title("Overlaid Molecular Weight Distribution (Histogram + KDE)")
plt.legend()
plt.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join("/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Results/", "molecular_weight_distribution_overlaid.png"))
plt.close()

# Print statistical summary
print("Molecular weights calculated and overlaid distribution plot saved to results/molecular_weight_distribution_overlaid.png")
print("Molecular weights:", molecular_weights)
print(f"Mean MW: {np.mean(molecular_weights):.2f}")
print(f"Median MW: {np.median(molecular_weights):.2f}")
print(f"Std Dev MW: {np.std(molecular_weights):.2f}")

# Save data to CSV (optional)
import pandas as pd
df = pd.DataFrame({"SMILES": valid_smiles, "Molecular_Weight": molecular_weights})
df.to_csv(os.path.join("/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Results/", "molecular_weights.csv"), index=False)
print("Molecular weights saved to results/molecular_weights.csv")