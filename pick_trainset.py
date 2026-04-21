import os
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

# --- Path configuration ---
train_split_path = '/home/data3/xjh/DiffDock2/data/splits/timesplit_no_lig_overlap_train'
data_dir = '/home/data3/xjh/DiffDock2/data/PDBBIND_atomCorrected'
output_split_path = '/home/data3/xjh/DiffDock2/data/splits/ddpo_train_800'

# --- Parameter configuration ---
MAX_ATOMS = 1200   # Maximum number of protein + ligand atoms for memory control
TARGET_NUM = 650   # Target number of selected complexes

def main():
    # 1. Load candidate complexes from the original training split
    with open(train_split_path, 'r') as f:
        complexes = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Total complexes in original train split: {len(complexes)}")

    valid_complexes = []
    fps = []

    print("Step 1: Filtering by size and removing Boron (B) atoms...")
    for comp in tqdm(complexes):
        lig_path = os.path.join(data_dir, comp, f"{comp}_ligand.sdf")
        
        # Support both fixed and standard processed receptor files.
        rec_path_fix = os.path.join(data_dir, comp, f"{comp}_protein_processed_fix.pdb")
        rec_path_normal = os.path.join(data_dir, comp, f"{comp}_protein_processed.pdb")
        
        if os.path.exists(rec_path_fix):
            rec_path = rec_path_fix
        elif os.path.exists(rec_path_normal):
            rec_path = rec_path_normal
        else:
            continue

        if not os.path.exists(lig_path):
            continue

        # Load ligand
        supplier = Chem.SDMolSupplier(lig_path, sanitize=False)
        if not supplier or len(supplier) == 0: continue
        mol = supplier[0]
        if mol is None: continue

        # Remove ligands containing Boron (B) atoms.
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if 'B' in symbols:
            continue

        # Estimate the total atom count as a proxy for memory usage.
        ligand_atoms = mol.GetNumAtoms()
        try:
            with open(rec_path, 'r') as f:
                # Follow DiffDock by counting only alpha carbon atoms (CA).
                # In the PDB format, atom names are stored in columns 13-16.
                rec_atoms = sum(1 for line in f if line.startswith('ATOM') and line[12:16].strip() == 'CA')
        except Exception:
            continue
            
        total_atoms = ligand_atoms + rec_atoms
        if total_atoms > MAX_ATOMS:
            continue

        # Compute Morgan fingerprints for diversity-based selection.
        try:
            Chem.SanitizeMol(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            valid_complexes.append(comp)
            fps.append(fp)
        except Exception:
            continue

    print(f"\nValid complexes after filtering: {len(valid_complexes)}")

    if len(valid_complexes) == 0:
        print("Error: Still no valid complexes found. Please check paths again!")
        return

    if len(valid_complexes) < TARGET_NUM:
        print(f"Warning: Only found {len(valid_complexes)} valid complexes, taking all.")
        selected_complexes = valid_complexes
    else:
        print(f"Step 2: Selecting {TARGET_NUM} highly diverse complexes using MaxMin algorithm...")
        # Define the Tanimoto distance function.
        def distij(i, j, fps=fps):
            return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

        picker = MaxMinPicker()
        # Select TARGET_NUM maximally diverse samples.
        pickIndices = picker.LazyBitVectorPick(fps, len(fps), TARGET_NUM)
        selected_complexes = [valid_complexes[i] for i in pickIndices]

    # 3. Write the new split file
    os.makedirs(os.path.dirname(output_split_path), exist_ok=True)
    with open(output_split_path, 'w') as f:
        for comp in selected_complexes:
            f.write(comp + '\n')
            
    print(f"Success! Generated diverse training split with {len(selected_complexes)} complexes at:")
    print(output_split_path)

if __name__ == "__main__":
    main()
