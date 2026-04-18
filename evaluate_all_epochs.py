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

# Import required modules
from datasets.pdbbind import construct_loader, PDBBind 
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.vina_scoring_evaluate import calc_vina_rewards

def set_seed(seed=2026):
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
        set_seed(2026)
        print(f"🚀 Initializing Multi-GPU Evaluation (World Size: {world_size})")

    # ==================================================
    # Core hyperparameters
    # ==================================================
    eval_samples_per_complex = 32 # Full sampling with 32 poses
    top_k_list = [1, 5] 
    rmsd_threshold = 2.0 
    sampling_batch_size = 16 
    
    score_model_dir = 'workdir/paper_score_model'
    rl_ckpts_dir = './results/ddpo_train_600' 
    conf_model_dir = 'workdir/paper_confidence_model'
    conf_ckpt_path = 'best_model_epoch75.pt'
    
    # ==================================================
    # New: create a dedicated CSV archive folder
    # ==================================================
    csv_out_dir = os.path.join(rl_ckpts_dir, 'val_csv_results')
    if local_rank == 0 and not os.path.exists(csv_out_dir):
        os.makedirs(csv_out_dir)
        
    summary_csv_path = f'{rl_ckpts_dir}/ALL_EPOCHS_SUMMARY.csv' # Keep master summary at root

    # === 1. Load configs and datasets ===
    with open(f'{score_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
        
    score_model_args.inference_steps = 10
    score_model_args.samples_per_complex = eval_samples_per_complex 
    score_model_args.batch_size = 1 
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    
    if local_rank == 0: print("Loading Score Dataset...")
    _, val_loader = construct_loader(score_model_args, t_to_sigma, device) 
    
    with open(f'{conf_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
        
    confidence_args.esm_embeddings_path = score_model_args.esm_embeddings_path
    
    if local_rank == 0: print('Loading Confidence Dataset...')
    split_path = getattr(score_model_args, 'split_val', getattr(score_model_args, 'split_test', None))
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
    
    # === 3. Collect all epochs and prepend Baseline ===
    pt_files = glob.glob(os.path.join(rl_ckpts_dir, 'rl_model_epoch_*.pt'))
    epochs_to_eval = sorted([int(f.split('_')[-1].split('.')[0]) for f in pt_files])
    epochs_to_eval = ['Baseline'] + epochs_to_eval
    
    # Resume-from-breakpoint evaluation logic
    evaluated_epochs = []
    if os.path.exists(summary_csv_path):
        try:
            df_existing = pd.read_csv(summary_csv_path)
            evaluated_epochs = df_existing['Epoch'].astype(str).tolist()
        except: pass
        
    summary_headers = [
        'Epoch', 
        'Hybrid_Top1_Success', 'Vina_Top1_Success', 'Conf_Top1_Success',
        'Hybrid_Mean_RMSD', 'Vina_Mean_RMSD', 'Conf_Mean_RMSD',
        'Hybrid_Mean_Vina', 'Vina_Mean_Vina', 'Conf_Mean_Vina',
        'Oracle_Top32_Success'
    ]
    
    if local_rank == 0:
        print(f"\nFound targets to evaluate: {epochs_to_eval}")
        if evaluated_epochs:
            print(f"✅ Already evaluated (Will Skip): {evaluated_epochs}")
        if not os.path.exists(summary_csv_path):
            with open(summary_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_headers)
                writer.writeheader()

    # ==================================================
    # Outer evaluation loop
    # ==================================================
    for epoch in epochs_to_eval:
        if str(epoch) in evaluated_epochs:
            if local_rank == 0: print(f"\n⏭️ Skipping {epoch} (Already in summary CSV)")
            continue
            
        if local_rank == 0: print(f"\n" + "="*50 + f"\n🚀 Evaluating {epoch}\n" + "="*50)
        
        # Backward-compatible checkpoint loading
        if epoch == 'Baseline':
            ckpt_path = os.path.join(score_model_dir, 'best_ema_inference_epoch_model.pt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(score_model_dir, 'last_model.pt')
        else:
            ckpt_path = os.path.join(rl_ckpts_dir, f'rl_model_epoch_{epoch}.pt')
            
        state_dict = torch.load(ckpt_path, map_location=device)
        if 'ema_weights' in state_dict and epoch == 'Baseline':
            model_state = state_dict['ema_weights'] 
        elif 'model_state_dict' in state_dict:
            model_state = state_dict['model_state_dict']
        elif 'model' in state_dict:
            model_state = state_dict['model']
        else:
            model_state = state_dict
            
        model.load_state_dict({k.replace('module.', ''): v for k, v in model_state.items()}, strict=True)
        model = model.to(device).eval()
        
        N = score_model_args.samples_per_complex
        csv_headers = [
            'complex_name', 
            'hybrid_top1_rmsd', 'hybrid_top1_vina', 'hybrid_top1_conf_score',
            'vina_top1_rmsd', 'vina_top1_vina', 
            'conf_top1_rmsd', 'conf_top1_vina', 'conf_top1_conf_score',
            'best_possible_vina', 'best_possible_rmsd', f'top_{N}_success_{rmsd_threshold}A',
            
            # ==========================================
            # New 3 columns: store full data for all 32 molecules
            # ==========================================
            'all_32_rmsds', 
            'all_32_vinas', 
            'all_32_confs'
        ]
        for k in top_k_list:
            csv_headers.extend([f'hybrid_top{k}_success_{rmsd_threshold}A', f'vina_top{k}_success_{rmsd_threshold}A', f'conf_top{k}_success_{rmsd_threshold}A'])
            
        # Path update: write to the new csv_out_dir folder
        temp_csv = f'{csv_out_dir}/temp_val_epoch_{epoch}_rank_{local_rank}.csv'
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            
        if is_distributed and hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(0 if epoch == 'Baseline' else epoch)
            
        pbar = tqdm(val_loader, disable=(local_rank != 0))
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
                    
                    # Blind Vina evaluation
                    vina_scores, rmsd_tensor = calc_vina_rewards(final_data_list, orig_complex_graph, score_model_args)
                    vina_np = vina_scores.cpu().numpy() 
                    rmsd_np = rmsd_tensor.cpu().numpy()
                    
                    # Final robust bad-sample defense
                    bad_mol_mask = np.isnan(rmsd_np) | np.isinf(rmsd_np) | np.isnan(vina_np) | np.isnan(conf_np) | (rmsd_np >= 49.0)
                    vina_np = np.nan_to_num(vina_np, nan=-100.0, posinf=-100.0, neginf=-100.0)
                    conf_np = np.nan_to_num(conf_np, nan=-10.0, posinf=-10.0, neginf=-10.0)
                    rmsd_np = np.nan_to_num(rmsd_np, nan=40.0, posinf=40.0, neginf=40.0) 
                    vina_np[bad_mol_mask] = -100.0
                    conf_np[bad_mol_mask] = -10.0
                    rmsd_np[bad_mol_mask] = 40.0
                    
                    hybrid_scores = 0.2 * vina_np + 0.8 * conf_np
                    idx_hybrid = np.argsort(hybrid_scores)[::-1] 
                    idx_vina = np.argsort(vina_np)[::-1] 
                    idx_conf = np.argsort(conf_np)[::-1] 
                    
                    record = {
                        'complex_name': complex_name,
                        'hybrid_top1_rmsd': rmsd_np[idx_hybrid[0]], 'hybrid_top1_vina': vina_np[idx_hybrid[0]], 'hybrid_top1_conf_score': conf_np[idx_hybrid[0]],
                        'vina_top1_rmsd': rmsd_np[idx_vina[0]], 'vina_top1_vina': vina_np[idx_vina[0]],
                        'conf_top1_rmsd': rmsd_np[idx_conf[0]], 'conf_top1_vina': vina_np[idx_conf[0]], 'conf_top1_conf_score': conf_np[idx_conf[0]],
                        'best_possible_vina': np.max(vina_np), 'best_possible_rmsd': np.min(rmsd_np),
                        f'top_{N}_success_{rmsd_threshold}A': int(np.any(rmsd_np < rmsd_threshold)),
                        # ==========================================
                        # New 3 fields: log all raw values for 32 molecules
                        # ==========================================
                        'all_32_rmsds': [round(float(x), 4) for x in rmsd_np],
                        'all_32_vinas': [round(float(x), 4) for x in vina_np],
                        'all_32_confs': [round(float(x), 4) for x in conf_np]
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
            
        # ==================================================
        # Merge results on the main process
        # ==================================================
        if local_rank == 0:
            df_list = []
            for r in range(world_size):
                # Path update: read temp files from the new folder
                t_csv = f'{csv_out_dir}/temp_val_epoch_{epoch}_rank_{r}.csv'
                if os.path.exists(t_csv):
                    df_rank = pd.read_csv(t_csv)
                    df_list.append(df_rank)
                    os.remove(t_csv) 
                    
            if len(df_list) > 0:
                df_merged = pd.concat(df_list, ignore_index=True)
                # Path update: write full epoch-detail CSV into the new folder
                epoch_csv_path = f'{csv_out_dir}/val_detail_epoch_{epoch}.csv'
                df_merged.to_csv(epoch_csv_path, index=False)
                
                hybrid_acc = df_merged[f'hybrid_top1_success_{rmsd_threshold}A'].mean() * 100
                vina_acc = df_merged[f'vina_top1_success_{rmsd_threshold}A'].mean() * 100
                conf_acc = df_merged[f'conf_top1_success_{rmsd_threshold}A'].mean() * 100
                oracle = df_merged[f'top_{N}_success_{rmsd_threshold}A'].mean() * 100
                
                summary_record = {
                    'Epoch': epoch,
                    'Hybrid_Top1_Success': round(hybrid_acc, 2), 'Vina_Top1_Success': round(vina_acc, 2), 'Conf_Top1_Success': round(conf_acc, 2),
                    'Hybrid_Mean_RMSD': round(df_merged['hybrid_top1_rmsd'].mean(), 4), 'Vina_Mean_RMSD': round(df_merged['vina_top1_rmsd'].mean(), 4), 'Conf_Mean_RMSD': round(df_merged['conf_top1_rmsd'].mean(), 4),
                    'Hybrid_Mean_Vina': round(df_merged['hybrid_top1_vina'].mean(), 4), 'Vina_Mean_Vina': round(df_merged['vina_top1_vina'].mean(), 4), 'Conf_Mean_Vina': round(df_merged['conf_top1_vina'].mean(), 4),
                    'Oracle_Top32_Success': round(oracle, 2)
                }
                
                with open(summary_csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_headers)
                    writer.writerow(summary_record)
                    
                print(f"[{epoch} Summary] Hybrid: {hybrid_acc:.2f}% | Vina: {vina_acc:.2f}% | Conf: {conf_acc:.2f}% | Oracle: {oracle:.2f}%")
                
    if local_rank == 0:
        print("\n🎉 Process Finished! Check your master summary file:")
        print(f"👉 {summary_csv_path}")
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()