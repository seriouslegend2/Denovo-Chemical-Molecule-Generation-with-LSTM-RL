import os
from rdkit import Chem  # Import RDKit for SMILES canonicalization

def preprocess_and_save_smiles(input_folder, output_filepath):
    """
    Reads all .smi files from the input folder, extracts the SMILES column, canonicalizes the SMILES,
    and saves them into a single output file.
    """
    with open(output_filepath, "w") as outfile:
        outfile.write("smiles\n")  # Write the header line

        # Iterate through all .smi files in the input folder
        for file in os.listdir(input_folder):
            if file.endswith(".smi"):
                file_path = os.path.join(input_folder, file)
                with open(file_path, "r") as infile:
                    for line_number, line in enumerate(infile):
                        # Skip the header line in the input file
                        if line_number == 0 and line.strip().lower().startswith("smiles"):
                            continue
                        
                        # Extract the SMILES column (first column)
                        smiles = line.split()[0]
                        
                        # Handle missing or invalid SMILES
                        if not smiles.strip():
                            continue
                        
                        # Canonicalize SMILES
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue  # Skip invalid SMILES
                        
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        outfile.write(canonical_smiles + "\n")  # Write canonicalized SMILES

# Example usage
if __name__ == "__main__":
    input_folder = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/dataset/"  # Folder containing .smi files
    output_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/preprocessing_step_1.smi"  # Single output file
    preprocess_and_save_smiles(input_folder, output_file)
