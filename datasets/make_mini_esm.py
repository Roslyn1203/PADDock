import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--esm_embeddings_path', type=str, default='data/embeddings_output', help='Source directory containing many individual .pt embedding files')
    parser.add_argument('--output_path', type=str, default='data/esm2_3billion_embeddings_mini.pt', help='Output path for the mini packed embedding file')
    parser.add_argument('--split_files', nargs='+', required=True, help='train/val/test path')
    args = parser.parse_args()

    # 1. Collect all required PDB IDs (e.g., '8hvp')
    required_ids = set()
    for split_file in args.split_files:
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    pdb_id = line.strip()
                    if pdb_id:
                        required_ids.add(pdb_id)
            print(f"Loaded split list: {split_file}")
        else:
            print(f"Warning: split list file not found: {split_file}")

    print(f"\nNeed to extract ESM features for {len(required_ids)} proteins in total...")

    # 2. Scan all embedding files and collect multi-chain features for selected IDs
    emb_dict = {}
    found_ids = set()
    
    # Get all files from source directory
    print("Scanning embedding repository, please wait...")
    all_files = os.listdir(args.esm_embeddings_path)

    for filename in tqdm(all_files):
        if not filename.endswith('.pt'):
            continue
            
        # Extract '8hvp' from filename like '8hvp_chain_1.pt'
        pdb_id = filename.split('_')[0] 
        
        # Keep only IDs included in the split lists
        if pdb_id in required_ids:
            file_path = os.path.join(args.esm_embeddings_path, filename)
            # Keep dictionary key format like '8hvp_chain_1' for DiffDock compatibility
            dict_key = filename.split('.')[0] 
            
            emb_dict[dict_key] = torch.load(file_path)['representations'][33]
            found_ids.add(pdb_id)

    # 3. Save and report statistics
    print(f"\nPacked features for {len(emb_dict)} protein chains.")
    
    missing_ids = required_ids - found_ids
    if missing_ids:
        print(f"No chain features found for {len(missing_ids)} complexes.")
        # print("Missing IDs:", missing_ids)

    # Save as a compact dictionary file
    torch.save(emb_dict, args.output_path)
    print(f"\nMini ESM feature file saved to: {args.output_path}")

if __name__ == '__main__':
    main()