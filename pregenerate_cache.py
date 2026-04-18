import os
import sys
import yaml
import torch
from types import SimpleNamespace
from functools import partial
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dist_env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 
                 'MASTER_ADDR', 'MASTER_PORT', 'NCCL_DEBUG']
for key in dist_env_vars:
    if key in os.environ:
        del os.environ[key]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Pre-generating caches in safe single-process mode...")
    
    try:
        from train_ddpo import construct_loader, t_to_sigma_compl
        from datasets.pdbbind import PDBBind 
        
        # ==================================================
        # [Phase 1] Score Model Cache (Train & Val)
        # ==================================================
        print("\n" + "="*50)
        print("🔨 [Phase 1] Building Score Model Cache (Train & Val)")
        print("="*50)
        
        score_model_dir = 'workdir/paper_score_model'
        with open(os.path.join(score_model_dir, 'model_parameters.yml')) as f:
            score_args = SimpleNamespace(**yaml.safe_load(f))
        
        score_args.batch_size = 1
        score_args.inference_steps = 10
        score_args.samples_per_complex = 8
        score_args.num_workers = 1
        
        if not hasattr(score_args, 'split_train'):
            score_args.split_train = 'data/splits/filtered_train'
        if not hasattr(score_args, 'split_val'):
            score_args.split_val = 'data/splits/filtered_val'
        if not hasattr(score_args, 'split_test'):
            score_args.split_test = 'data/splits/filtered_test'
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_to_sigma = partial(t_to_sigma_compl, args=score_args)
        
        print(f"Cache Dir: {getattr(score_args, 'cache_path', 'default')}")
        train_loader, val_loader = construct_loader(score_args, t_to_sigma, device)
        print(f"✅ Score Cache ready! (Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)})")

        # ==================================================
        # [Phase 2] Confidence Cache (Val)
        # ==================================================
        print("\n" + "="*50)
        print("🔨 [Phase 2] Building Confidence Cache (Val)")
        print("="*50)
        
        conf_model_dir = '/home/data3/xjh/DiffDock/workdir/paper_confidence_model'
        with open(os.path.join(conf_model_dir, 'model_parameters.yml')) as f:
            conf_args = SimpleNamespace(**yaml.safe_load(f))
            
        conf_args.esm_embeddings_path = score_args.esm_embeddings_path
        conf_args.cache_path = score_args.cache_path 
        
        split_val = score_args.split_val
        
        confidence_val_dataset = PDBBind(
            transform=None, root=score_args.data_dir, limit_complexes=score_args.limit_complexes,
            receptor_radius=conf_args.receptor_radius, cache_path=conf_args.cache_path, split_path=split_val,
            remove_hs=conf_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=conf_args.c_alpha_max_neighbors,
            matching=not conf_args.no_torsion, keep_original=True, popsize=conf_args.matching_popsize,
            maxiter=conf_args.matching_maxiter, all_atoms=conf_args.all_atoms,
            atom_radius=conf_args.atom_radius, atom_max_neighbors=conf_args.atom_max_neighbors,
            esm_embeddings_path=conf_args.esm_embeddings_path, require_ligand=True, num_workers=1
        )
        print(f"✅ Confidence Cache ready! (Val: {len(confidence_val_dataset)})")

        # ==================================================
        # [Phase 3] TEST Cache (core section)
        # ==================================================
        print("\n" + "="*50)
        print("🔨 [Phase 3] Building TEST Cache (Score + Confidence)")
        print("="*50)

        split_test = score_args.split_test
        print(f"Loading TEST split from: {split_test}")

        # -------- Score Test --------
        print("\n📦 Building Score TEST Cache...")
        score_test_dataset = PDBBind(
            transform=None, root=score_args.data_dir, limit_complexes=score_args.limit_complexes,
            receptor_radius=score_args.receptor_radius, cache_path=score_args.cache_path, split_path=split_test,
            remove_hs=score_args.remove_hs, max_lig_size=score_args.max_lig_size, 
            c_alpha_max_neighbors=score_args.c_alpha_max_neighbors, matching=not score_args.no_torsion,
            keep_original=True, popsize=score_args.matching_popsize, maxiter=score_args.matching_maxiter,
            all_atoms=score_args.all_atoms, atom_radius=score_args.atom_radius, 
            atom_max_neighbors=score_args.atom_max_neighbors, esm_embeddings_path=score_args.esm_embeddings_path, 
            require_ligand=True, num_workers=1
        )

        test_loader = DataLoader(score_test_dataset, batch_size=1, shuffle=False)

        
        print(f"✅ Score TEST Cache ready! ({len(score_test_dataset)})")

        # -------- Confidence Test --------
        print("\n📦 Building Confidence TEST Cache...")

        confidence_test_dataset = PDBBind(
            transform=None, root=score_args.data_dir, limit_complexes=score_args.limit_complexes,
            receptor_radius=conf_args.receptor_radius, cache_path=conf_args.cache_path, split_path=split_test,
            remove_hs=conf_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=conf_args.c_alpha_max_neighbors,
            matching=not conf_args.no_torsion, keep_original=True, popsize=conf_args.matching_popsize,
            maxiter=conf_args.matching_maxiter, all_atoms=conf_args.all_atoms,
            atom_radius=conf_args.atom_radius, atom_max_neighbors=conf_args.atom_max_neighbors,
            esm_embeddings_path=conf_args.esm_embeddings_path, require_ligand=True, num_workers=1
        )

       

        print(f"✅ Confidence TEST Cache ready! ({len(confidence_test_dataset)})")

        print("\n🎉 All caches (Train / Val / Test) generated successfully! 🎉\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()