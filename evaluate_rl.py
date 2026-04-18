import os
import glob
import yaml
import copy
import random
import csv 
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Import required modules
from datasets.pdbbind import construct_loader, PDBBind 
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.vina_scoring import calc_vina_rewards

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # ==================================================
    # DDP distributed initialization
    # ==================================================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if local_rank == 0:
        print(f"🚀 Initializing 10-Seed TEST Evaluation (World Size: {world_size})")

    # ==================================================
    # Core evaluation hyperparameters
    # ==================================================
    eval_samples_per_complex = 32  
    top_k_list = [1, 5]           
    rmsd_threshold = 2.0          
    sampling_batch_size = 16       
    num_seeds = 10                 # Evaluate with 10 random seeds
    base_seed = 2026               # Starting seed
    
    score_model_dir = 'workdir/paper_score_model'
    rl_ckpts_dir = './results/ddpo_train_600' 
    
    conf_model_dir = '/home/data3/xjh/DiffDock/workdir/paper_confidence_model'
    conf_ckpt_path = 'best_model_epoch75.pt'
    
    test_summary_path = f'{rl_ckpts_dir}/TEST_10_SEEDS_SUMMARY.csv'
    # ==================================================

    # === 1. Load configs and test dataset ===
    with open(f'{score_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    
    score_model_args.inference_steps = 10
    score_model_args.samples_per_complex = eval_samples_per_complex 
    score_model_args.batch_size = 1 
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    
    # Key change: force loader split path to test split
    score_model_args.split_val = getattr(score_model_args, 'split_test', 'data/splits/timesplit_test')
    
    if local_rank == 0: print(f"Loading Score TEST Dataset from {score_model_args.split_val}...")
    _, test_loader = construct_loader(score_model_args, t_to_sigma, device) 

    with open(f'{conf_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
        
    confidence_args.esm_embeddings_path = score_model_args.esm_embeddings_path
    
    if local_rank == 0: print('Loading Confidence TEST Dataset...')
    split_path = getattr(score_model_args, 'split_test', 'data/splits/timesplit_test')
    confidence_test_dataset = PDBBind(
        transform=None, root=score_model_args.data_dir, limit_complexes=score_model_args.limit_complexes,
        receptor_radius=confidence_args.receptor_radius, cache_path=confidence_args.cache_path, split_path=split_path,
        remove_hs=confidence_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
        matching=not confidence_args.no_torsion, keep_original=True, popsize=confidence_args.matching_popsize,
        maxiter=confidence_args.matching_maxiter, all_atoms=confidence_args.all_atoms,
        atom_radius=confidence_args.atom_radius, atom_max_neighbors=confidence_args.atom_max_neighbors,
        esm_embeddings_path=confidence_args.esm_embeddings_path, require_ligand=True, num_workers=1
    )
    confidence_complex_dict = { (d.name[0] if isinstance(d.name, list) else d.name): d for d in confidence_test_dataset }

    # === 2. Initialize network structures ===
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
    
    conf_state_dict = torch.load(f'{conf_model_dir}/{conf_ckpt_path}', map_location=device)
    confidence_model.load_state_dict(conf_state_dict, strict=True)
    confidence_model = confidence_model.to(device).eval()
    
    tr_schedule = get_t_schedule(inference_steps=score_model_args.inference_steps)

    # === 3. Evaluate only Baseline and Epoch 60 ===
    models_to_eval = ['Baseline', '60']
    
    summary_headers = [
        'Model', 'Seed', 
        'Hybrid_Top1_Success', 'Vina_Top1_Success', 'Conf_Top1_Success',
        'Hybrid_Mean_RMSD', 'Vina_Mean_RMSD', 'Conf_Mean_RMSD',
        'Hybrid_Mean_Vina', 'Vina_Mean_Vina', 'Conf_Mean_Vina',
        'Oracle_Top32_Success'
    ]

    if local_rank == 0:
        if not os.path.exists(test_summary_path):
            with open(test_summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_headers)
                writer.writeheader()

    # ==================================================
    # Outer loop: model checkpoints (Baseline -> 60)
    # ==================================================
    for target_model in models_to_eval:
        if local_rank == 0: print(f"\n" + "="*50 + f"\n🎯 Loading Model: {target_model}\n" + "="*50)
        
        # Load weights
        if target_model == 'Baseline':
            ckpt_path = os.path.join(score_model_dir, 'best_ema_inference_epoch_model.pt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(score_model_dir, 'last_model.pt')
        else:
            ckpt_path = os.path.join(rl_ckpts_dir, f'rl_model_epoch_{target_model}.pt')

        state_dict = torch.load(ckpt_path, map_location=device)
        if 'ema_weights' in state_dict and target_model == 'Baseline':
             model_state = state_dict['ema_weights'] 
        elif 'model_state_dict' in state_dict:
             model_state = state_dict['model_state_dict']
        elif 'model' in state_dict:
             model_state = state_dict['model']
        else:
             model_state = state_dict

        model.load_state_dict({k.replace('module.', ''): v for k, v in model_state.items()}, strict=True)
        model = model.to(device).eval()

        # ==================================================
        # Inner loop: run 10 different seeds
        # ==================================================
        for seed_idx in range(num_seeds):
            current_seed = base_seed + seed_idx
            set_seed(current_seed) # Change seed to vary diffusion sampling noise
            
            if local_rank == 0: print(f"\n🧬 Model: [{target_model}] | Running Seed: {current_seed} ({seed_idx+1}/{num_seeds})")

            results_records = []
            N = score_model_args.samples_per_complex
            
            csv_headers = [
                'complex_name', 
                'hybrid_top1_rmsd', 'hybrid_top1_vina', 'hybrid_top1_conf_score',
                'vina_top1_rmsd', 'vina_top1_vina', 
                'conf_top1_rmsd', 'conf_top1_vina', 'conf_top1_conf_score',
                'best_possible_vina', 'best_possible_rmsd', f'top_{N}_success_{rmsd_threshold}A'
            ]
            for k in top_k_list:
                csv_headers.extend([f'hybrid_top{k}_success_{rmsd_threshold}A', f'vina_top{k}_success_{rmsd_threshold}A', f'conf_top{k}_success_{rmsd_threshold}A'])

            temp_csv = f'{rl_ckpts_dir}/temp_test_{target_model}_seed_{current_seed}_rank_{local_rank}.csv'
            with open(temp_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()

            if is_distributed and hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'set_epoch'):
                test_loader.sampler.set_epoch(current_seed)

            pbar = tqdm(test_loader, disable=(local_rank != 0))
            for idx, orig_complex_graph in enumerate(pbar):
                orig_complex_graph = orig_complex_graph.to('cpu')
                
                for attr in ['mol', 'name']:
                    if hasattr(orig_complex_graph, attr) and isinstance(getattr(orig_complex_graph, attr), list):
                        setattr(orig_complex_graph, attr, getattr(orig_complex_graph, attr)[0])
                if hasattr(orig_complex_graph, 'original_center') and orig_complex_graph.original_center.dim() > 1:
                    orig_complex_graph.original_center = orig_complex_graph.original_center[0]

                if hasattr(orig_complex_graph, 'success') and not orig_complex_graph.success: continue
                complex_name = getattr(orig_complex_graph, 'name', f"complex_{idx}")
                if complex_name not in confidence_complex_dict: continue
                    
                conf_complex_graph = confidence_complex_dict[complex_name].to('cpu')
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                conf_data_list = [copy.deepcopy(conf_complex_graph) for _ in range(N)]
                
                try:
                    randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max)
                    with torch.no_grad():
                        sampling_output = sampling(
                            data_list=data_list, model=model, inference_steps=score_model_args.inference_steps,
                            tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                            device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                            confidence_model=confidence_model, confidence_data_list=conf_data_list, 
                            confidence_model_args=confidence_args, batch_size=sampling_batch_size
                        )
                        
                        final_data_list, conf_scores = sampling_output
                        if conf_scores is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                            conf_scores = conf_scores[:, 0]
                            
                        conf_np = conf_scores.cpu().numpy()
                        if conf_np.ndim == 2: conf_np = conf_np[:, 0]

                        vina_scores, rmsd_tensor = calc_vina_rewards(final_data_list, orig_complex_graph, score_model_args)
                        vina_np = vina_scores.cpu().numpy() 
                        rmsd_np = rmsd_tensor.cpu().numpy()

                        # Final robust bad-sample defense
                        bad_mol_mask = np.isnan(rmsd_np) | np.isinf(rmsd_np) | np.isnan(vina_np) | np.isnan(conf_np)
                        vina_np = np.nan_to_num(vina_np, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
                        conf_np = np.nan_to_num(conf_np, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
                        rmsd_np = np.nan_to_num(rmsd_np, nan=50.0, posinf=50.0, neginf=50.0) 
                        vina_np[bad_mol_mask] = -9999.0
                        conf_np[bad_mol_mask] = -9999.0
                        rmsd_np[bad_mol_mask] = 50.0

                        hybrid_scores = vina_np + conf_np
                        
                        idx_hybrid = np.argsort(hybrid_scores)[::-1] 
                        idx_vina = np.argsort(vina_np)[::-1]         
                        idx_conf = np.argsort(conf_np)[::-1]         

                        record = {
                            'complex_name': complex_name,
                            'hybrid_top1_rmsd': rmsd_np[idx_hybrid[0]], 'hybrid_top1_vina': vina_np[idx_hybrid[0]], 'hybrid_top1_conf_score': conf_np[idx_hybrid[0]],
                            'vina_top1_rmsd': rmsd_np[idx_vina[0]], 'vina_top1_vina': vina_np[idx_vina[0]],
                            'conf_top1_rmsd': rmsd_np[idx_conf[0]], 'conf_top1_vina': vina_np[idx_conf[0]], 'conf_top1_conf_score': conf_np[idx_conf[0]],
                            'best_possible_vina': np.max(vina_np), 'best_possible_rmsd': np.min(rmsd_np),
                            f'top_{N}_success_{rmsd_threshold}A': int(np.any(rmsd_np < rmsd_threshold))
                        }
                        for k in top_k_list:
                            actual_k = min(k, N)
                            record[f'hybrid_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_hybrid[:actual_k]] < rmsd_threshold))
                            record[f'vina_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_vina[:actual_k]] < rmsd_threshold))
                            record[f'conf_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_conf[:actual_k]] < rmsd_threshold))

                        with open(temp_csv, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=csv_headers)
                            writer.writerow(record)
                            
                except Exception as e:
                    pass
                finally:
                    torch.cuda.empty_cache()

            if is_distributed:
                dist.barrier()

            # Merge per-seed data on the main process
            if local_rank == 0:
                df_list = []
                for r in range(world_size):
                    t_csv = f'{rl_ckpts_dir}/temp_test_{target_model}_seed_{current_seed}_rank_{r}.csv'
                    if os.path.exists(t_csv):
                        df_list.append(pd.read_csv(t_csv))
                        os.remove(t_csv) 
                
                if len(df_list) > 0:
                    df_merged = pd.concat(df_list, ignore_index=True)
                    # Optional: save detailed records for each seed
                    # df_merged.to_csv(f'{rl_ckpts_dir}/test_detail_{target_model}_seed_{current_seed}.csv', index=False)
                    
                    hybrid_acc = df_merged[f'hybrid_top1_success_{rmsd_threshold}A'].mean() * 100
                    vina_acc = df_merged[f'vina_top1_success_{rmsd_threshold}A'].mean() * 100
                    conf_acc = df_merged[f'conf_top1_success_{rmsd_threshold}A'].mean() * 100
                    oracle = df_merged[f'top_{N}_success_{rmsd_threshold}A'].mean() * 100
                    
                    summary_record = {
                        'Model': target_model,
                        'Seed': current_seed,
                        'Hybrid_Top1_Success': round(hybrid_acc, 2), 'Vina_Top1_Success': round(vina_acc, 2), 'Conf_Top1_Success': round(conf_acc, 2),
                        'Hybrid_Mean_RMSD': round(df_merged['hybrid_top1_rmsd'].mean(), 4), 'Vina_Mean_RMSD': round(df_merged['vina_top1_rmsd'].mean(), 4), 'Conf_Mean_RMSD': round(df_merged['conf_top1_rmsd'].mean(), 4),
                        'Hybrid_Mean_Vina': round(df_merged['hybrid_top1_vina'].mean(), 4), 'Vina_Mean_Vina': round(df_merged['vina_top1_vina'].mean(), 4), 'Conf_Mean_Vina': round(df_merged['conf_top1_vina'].mean(), 4),
                        'Oracle_Top32_Success': round(oracle, 2)
                    }
                    
                    with open(test_summary_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=summary_headers)
                        writer.writerow(summary_record)

        # ==================================================
        # Model-level aggregation: mean and std over 10 seeds
        # ==================================================
        if local_rank == 0:
            df_all_seeds = pd.read_csv(test_summary_path)
            df_model = df_all_seeds[df_all_seeds['Model'] == str(target_model)]
            
            if len(df_model) == num_seeds:
                print(f"\n📊 FINAL AGGREGATION FOR MODEL [{target_model}] (Over {num_seeds} Seeds):")
                print("-" * 60)
                
                h_succ_m, h_succ_s = df_model['Hybrid_Top1_Success'].mean(), df_model['Hybrid_Top1_Success'].std()
                v_succ_m, v_succ_s = df_model['Vina_Top1_Success'].mean(), df_model['Vina_Top1_Success'].std()
                
                h_rmsd_m, h_rmsd_s = df_model['Hybrid_Mean_RMSD'].mean(), df_model['Hybrid_Mean_RMSD'].std()
                v_rmsd_m, v_rmsd_s = df_model['Vina_Mean_RMSD'].mean(), df_model['Vina_Mean_RMSD'].std()

                h_vina_m, h_vina_s = df_model['Hybrid_Mean_Vina'].mean(), df_model['Hybrid_Mean_Vina'].std()
                v_vina_m, v_vina_s = df_model['Vina_Mean_Vina'].mean(), df_model['Vina_Mean_Vina'].std()

                print(f"✅ Success Rate  | Hybrid: {h_succ_m:.1f}% ± {h_succ_s:.1f}%  |  Vina: {v_succ_m:.1f}% ± {v_succ_s:.1f}%")
                print(f"📐 Mean RMSD     | Hybrid: {h_rmsd_m:.2f} ± {h_rmsd_s:.2f}    |  Vina: {v_rmsd_m:.2f} ± {v_rmsd_s:.2f}")
                print(f"⚡ Vina Score    | Hybrid: {h_vina_m:.2f} ± {h_vina_s:.2f}    |  Vina: {v_vina_m:.2f} ± {v_vina_s:.2f}")
                print("-" * 60)

    if local_rank == 0:
        print(f"\n🎉 Test Completed! All data saved to 👉 {test_summary_path}")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()