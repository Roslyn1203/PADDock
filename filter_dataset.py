import os
from rdkit import Chem
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
BASE_DIR = '/home/data3/xjh/DiffDock/data/PDBBIND_atomCorrected'
TEST_SPLIT_FILE = '/home/data3/xjh/DiffDock/data/splits/timesplit_test'
NO_OVERLAP_SPLIT_FILE = '/home/data3/xjh/DiffDock/data/splits/timesplit_no_lig_overlap_val'

# VRAM threshold: tune based on available memory. Proteins with >1000 residues may easily OOM.
MAX_RESIDUES = 1000 

def has_boron(sdf_path):
    """Check whether molecule contains boron atom (B)."""
    if not os.path.exists(sdf_path):
        return True # Treat missing file as invalid sample
    
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
    try:
        mol = next(suppl)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'B' or atom.GetAtomicNum() == 5:
                    return True
    except:
        return True # Drop invalid samples if RDKit parsing fails
    return False

def get_protein_length(pdb_path):
    """Estimate protein length by counting CA (alpha-carbon) atoms."""
    if not os.path.exists(pdb_path):
        return 99999 
    
    residue_count = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            # Strictly match ATOM line with atom name CA
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                residue_count += 1
    return residue_count

def process_split(split_file, split_name):
    """Process and filter a given split file."""
    if not os.path.exists(split_file):
        print(f"File not found: {split_file}")
        return []

    with open(split_file, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    valid_pdb_ids = []
    print(f"\nScanning {split_name} ({len(pdb_ids)} complexes)...")
    
    for pdb_id in tqdm(pdb_ids):
        complex_dir = os.path.join(BASE_DIR, pdb_id)
        sdf_path = os.path.join(complex_dir, f'{pdb_id}_ligand.sdf')
        pdb_path = os.path.join(complex_dir, f'{pdb_id}_protein_processed_fix.pdb')

        # Check 1: boron atom presence
        if has_boron(sdf_path):
            # print(f"Filtered {pdb_id}: contains boron or invalid ligand")
            continue
            
        # Check 2: oversized protein
        length = get_protein_length(pdb_path)
        if length > MAX_RESIDUES:
            # print(f"Filtered {pdb_id}: protein too large (residue count {length} > {MAX_RESIDUES})")
            continue

        valid_pdb_ids.append(pdb_id)
        
    print(f"[{split_name}] Filtering finished. Valid complexes: {len(valid_pdb_ids)} / {len(pdb_ids)}")
    return valid_pdb_ids

def main():
    # 1. Filter timesplit_test
    valid_test = process_split(TEST_SPLIT_FILE, "timesplit_test")
    
    # 2. Filter timesplit_test_no_rec_overlap
    valid_no_overlap = process_split(NO_OVERLAP_SPLIT_FILE, "timesplit_test_no_rec_overlap")
    
    # 3. Save results
    out_dir = './filtered_splits'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'filtered_test.txt'), 'w') as f:
        f.write('\n'.join(valid_test))
        
    with open(os.path.join(out_dir, 'filtered_test_no_overlap.txt'), 'w') as f:
        f.write('\n'.join(valid_no_overlap))
        
    print(f"\n✅ Filtering complete. Lists saved under {out_dir}/.")

if __name__ == '__main__':
    main()