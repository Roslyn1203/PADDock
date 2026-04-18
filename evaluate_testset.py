import os
import glob
import yaml
import copy
import random
import csv 
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from argparse import Namespace
from functools import partial

import torch
from torch_geometric.loader import DataLoader

import datasets.process_mols as process_mols_module
import datasets.pdbbind as pdbbind_module
from datasets.pdbbind import PDBBind 
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.vina_scoring_evaluate import calc_vina_rewards

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_protein_mode(protein_mode):
    protein_mode = protein_mode.lower()
    aliases = {
        'holo': 'holo',
        'crystal': 'holo',
        'processed': 'holo',
        'apo': 'esmfold',
        'apo_esmfold': 'esmfold',
        'esm': 'esmfold',
        'esmfold': 'esmfold',
        'af2': 'alphafold',
        'alphafold': 'alphafold',
    }
    if protein_mode not in aliases:
        raise ValueError(f"Unsupported protein_mode: {protein_mode}")
    return aliases[protein_mode]


def get_protein_filename(complex_name, protein_mode):
    filename_map = {
        'holo': f'{complex_name}_protein_processed_fix.pdb',
        'esmfold': f'{complex_name}_protein_esmfold_aligned_tr_fix.pdb',
        'alphafold': f'{complex_name}_protein_alphafold_aligned_tr_fix.pdb',
    }
    return filename_map[protein_mode]


def get_protein_path(data_dir, complex_name, protein_mode):
    primary_path = os.path.join(data_dir, complex_name, get_protein_filename(complex_name, protein_mode))
    if os.path.exists(primary_path):
        return primary_path

    if protein_mode != 'holo':
        fallback_path = os.path.join(data_dir, complex_name, get_protein_filename(complex_name, 'holo'))
        if os.path.exists(fallback_path):
            print(f"⚠️ {complex_name} missing {protein_mode} protein, fallback to Holo: {os.path.basename(fallback_path)}")
            return fallback_path

    return primary_path


def install_protein_mode_patch(data_dir, protein_mode):
    original_parse_pdb = process_mols_module.parsePDB
    original_parse_receptor_process = process_mols_module.parse_receptor
    original_parse_receptor_dataset = pdbbind_module.parse_receptor

    def parse_pdb_for_mode(pdbid, pdbbind_dir):
        target_root = pdbbind_dir if pdbbind_dir is not None else data_dir
        protein_path = get_protein_path(target_root, pdbid, protein_mode)
        return process_mols_module.parse_pdb_from_path(protein_path)

    def parse_receptor_for_mode(pdbid, pdbbind_dir):
        return parse_pdb_for_mode(pdbid, pdbbind_dir)

    process_mols_module.parsePDB = parse_pdb_for_mode
    process_mols_module.parse_receptor = parse_receptor_for_mode
    pdbbind_module.parse_receptor = parse_receptor_for_mode

    return original_parse_pdb, original_parse_receptor_process, original_parse_receptor_dataset


def restore_protein_mode_patch(originals):
    original_parse_pdb, original_parse_receptor_process, original_parse_receptor_dataset = originals
    process_mols_module.parsePDB = original_parse_pdb
    process_mols_module.parse_receptor = original_parse_receptor_process
    pdbbind_module.parse_receptor = original_parse_receptor_dataset


def parse_numeric_array(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def get_cache_path_for_mode(base_cache_path, protein_mode, confidence=False):
    if protein_mode == 'holo':
        return f"{base_cache_path}_confidence" if confidence else base_cache_path

    suffix = f"_{protein_mode}_confidence" if confidence else f"_{protein_mode}"
    return f"{base_cache_path}{suffix}"


def get_mode_file_suffix(protein_mode):
    return '' if protein_mode == 'holo' else f'_{protein_mode}'


def ensure_topk_best_rmsd_columns(df, top_k_list, samples_per_complex):
    required_columns = [
        f'{method}_top{k}_best_rmsd'
        for method in ['hybrid', 'vina', 'conf']
        for k in top_k_list
    ]
    if all(col in df.columns for col in required_columns):
        return df

    if not all(col in df.columns for col in ['all_32_rmsds', 'all_32_vinas', 'all_32_confs']):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f'Missing required top-k RMSD columns and raw arrays: {missing}')

    def compute_row_metrics(row):
        rmsd_np = parse_numeric_array(row['all_32_rmsds'])
        vina_np = parse_numeric_array(row['all_32_vinas'])
        conf_np = parse_numeric_array(row['all_32_confs'])

        actual_n = min(samples_per_complex, len(rmsd_np), len(vina_np), len(conf_np))
        rmsd_np = rmsd_np[:actual_n]
        vina_np = vina_np[:actual_n]
        conf_np = conf_np[:actual_n]

        hybrid_scores = 0.2 * vina_np + 0.8 * conf_np
        orderings = {
            'hybrid': np.argsort(hybrid_scores)[::-1],
            'vina': np.argsort(vina_np)[::-1],
            'conf': np.argsort(conf_np)[::-1],
        }

        metrics = {}
        for method, ordering in orderings.items():
            for k in top_k_list:
                actual_k = min(k, len(ordering))
                metrics[f'{method}_top{k}_best_rmsd'] = float(np.min(rmsd_np[ordering[:actual_k]]))
        return pd.Series(metrics)

    computed_metrics = df.apply(compute_row_metrics, axis=1)
    for col in computed_metrics.columns:
        df[col] = computed_metrics[col]
    return df


def ensure_topk_best_vina_columns(df, top_k_list, samples_per_complex):
    required_columns = [
        f'{method}_top{k}_best_vina_at_best_rmsd'
        for method in ['hybrid', 'vina', 'conf']
        for k in top_k_list
    ]
    if all(col in df.columns for col in required_columns):
        return df

    if not all(col in df.columns for col in ['all_32_rmsds', 'all_32_vinas', 'all_32_confs']):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f'Missing required top-k Vina columns and raw arrays: {missing}')

    def compute_row_metrics(row):
        rmsd_np = parse_numeric_array(row['all_32_rmsds'])
        vina_np = parse_numeric_array(row['all_32_vinas'])
        conf_np = parse_numeric_array(row['all_32_confs'])

        actual_n = min(samples_per_complex, len(rmsd_np), len(vina_np), len(conf_np))
        rmsd_np = rmsd_np[:actual_n]
        vina_np = vina_np[:actual_n]
        conf_np = conf_np[:actual_n]

        hybrid_scores = 0.2 * vina_np + 0.8 * conf_np
        orderings = {
            'hybrid': np.argsort(hybrid_scores)[::-1],
            'vina': np.argsort(vina_np)[::-1],
            'conf': np.argsort(conf_np)[::-1],
        }

        metrics = {}
        for method, ordering in orderings.items():
            for k in top_k_list:
                actual_k = min(k, len(ordering))
                topk_indices = ordering[:actual_k]
                best_idx = topk_indices[np.argmin(rmsd_np[topk_indices])]
                metrics[f'{method}_top{k}_best_vina_at_best_rmsd'] = float(vina_np[best_idx])
        return pd.Series(metrics)

    computed_metrics = df.apply(compute_row_metrics, axis=1)
    for col in computed_metrics.columns:
        df[col] = computed_metrics[col]
    return df


def build_summary_record(df_merged, target_model, current_seed, rmsd_threshold, samples_per_complex):
    df_merged = ensure_topk_best_rmsd_columns(df_merged, [1, 5], samples_per_complex)
    df_merged = ensure_topk_best_vina_columns(df_merged, [1, 5], samples_per_complex)

    hybrid_top1_acc = df_merged[f'hybrid_top1_success_{rmsd_threshold}A'].mean() * 100
    vina_top1_acc = df_merged[f'vina_top1_success_{rmsd_threshold}A'].mean() * 100
    conf_top1_acc = df_merged[f'conf_top1_success_{rmsd_threshold}A'].mean() * 100

    hybrid_top5_acc = df_merged[f'hybrid_top5_success_{rmsd_threshold}A'].mean() * 100
    vina_top5_acc = df_merged[f'vina_top5_success_{rmsd_threshold}A'].mean() * 100
    conf_top5_acc = df_merged[f'conf_top5_success_{rmsd_threshold}A'].mean() * 100

    oracle = df_merged[f'top_{samples_per_complex}_success_{rmsd_threshold}A'].mean() * 100

    return {
        'Model': target_model,
        'Seed': current_seed,
        'Hybrid_Top1_Success': round(hybrid_top1_acc, 2),
        'Vina_Top1_Success': round(vina_top1_acc, 2),
        'Conf_Top1_Success': round(conf_top1_acc, 2),
        'Hybrid_Top5_Success': round(hybrid_top5_acc, 2),
        'Vina_Top5_Success': round(vina_top5_acc, 2),
        'Conf_Top5_Success': round(conf_top5_acc, 2),
        'Hybrid_Median_RMSD': round(df_merged['hybrid_top1_rmsd'].median(), 4),
        'Vina_Median_RMSD': round(df_merged['vina_top1_rmsd'].median(), 4),
        'Conf_Median_RMSD': round(df_merged['conf_top1_rmsd'].median(), 4),
        'Hybrid_Top5_Median_RMSD': round(df_merged['hybrid_top5_best_rmsd'].median(), 4),
        'Vina_Top5_Median_RMSD': round(df_merged['vina_top5_best_rmsd'].median(), 4),
        'Conf_Top5_Median_RMSD': round(df_merged['conf_top5_best_rmsd'].median(), 4),
        'Hybrid_Mean_Vina': round(df_merged['hybrid_top1_vina'].mean(), 4),
        'Vina_Mean_Vina': round(df_merged['vina_top1_vina'].mean(), 4),
        'Conf_Mean_Vina': round(df_merged['conf_top1_vina'].mean(), 4),
        'Hybrid_Top5_Mean_Vina': round(df_merged['hybrid_top5_best_vina_at_best_rmsd'].mean(), 4),
        'Vina_Top5_Mean_Vina': round(df_merged['vina_top5_best_vina_at_best_rmsd'].mean(), 4),
        'Conf_Top5_Mean_Vina': round(df_merged['conf_top5_best_vina_at_best_rmsd'].mean(), 4),
        'Oracle_Top32_Success': round(oracle, 2)
    }

def main():
    # ==================================================
    # Single-GPU mode initialization
    # ==================================================
    parser = argparse.ArgumentParser(description="Single GPU Test Evaluation")
    parser.add_argument('--model', type=str, required=True, help="Model to evaluate: 'Baseline' or '60'")
    parser.add_argument(
        '--protein_mode',
        type=str,
        default='holo',
        help="Protein mode: holo / esmfold / alphafold. 'apo' is mapped to esmfold automatically.",
    )
    cmd_args = parser.parse_args()
    
    target_model = cmd_args.model
    protein_mode = normalize_protein_mode(cmd_args.protein_mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🚀 Initializing SINGLE-GPU TEST Evaluation for Model: [{target_model}] | Protein Mode: [{protein_mode}] on {device}")

    # ==================================================
    # Core evaluation hyperparameters
    # ==================================================
    eval_samples_per_complex = 32  
    top_k_list = [1, 5]           
    rmsd_threshold = 2.0          
    sampling_batch_size = 16       
    num_seeds = 3                 
    base_seed = 2026               

    score_model_dir = 'workdir/paper_score_model'
    rl_ckpts_dir = './results/ddpo_train_test_600'
    conf_model_dir = 'workdir/paper_confidence_model'
    conf_ckpt_path = 'best_model_epoch75.pt'
    
    test_csv_dir = os.path.join(rl_ckpts_dir, 'test_csv_results')
    if not os.path.exists(test_csv_dir):
        os.makedirs(test_csv_dir)

    mode_file_suffix = get_mode_file_suffix(protein_mode)
    test_summary_path = f'{rl_ckpts_dir}/TEST_{num_seeds}_SEEDS_SUMMARY_{target_model}{mode_file_suffix}.csv'

    # === 1. Load configs and test dataset ===
    with open(f'{score_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    
    score_model_args.inference_steps = 10
    score_model_args.samples_per_complex = eval_samples_per_complex 
    score_model_args.batch_size = 1 
    score_model_args.protein_mode = protein_mode
    score_model_args.cache_path = get_cache_path_for_mode(score_model_args.cache_path, protein_mode, confidence=False)
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    
    split_path_test = getattr(score_model_args, 'split_test', 'data/splits/timesplit_test')
    
    print(f"Loading Score TEST Dataset from {split_path_test}...")
    patch_originals = install_protein_mode_patch(score_model_args.data_dir, protein_mode)
    try:
        score_test_dataset = PDBBind(
            transform=None, root=score_model_args.data_dir, limit_complexes=score_model_args.limit_complexes,
            receptor_radius=score_model_args.receptor_radius, cache_path=score_model_args.cache_path, split_path=split_path_test,
            remove_hs=score_model_args.remove_hs, max_lig_size=score_model_args.max_lig_size, 
            c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors, matching=not score_model_args.no_torsion,
            keep_original=True, popsize=score_model_args.matching_popsize, maxiter=score_model_args.matching_maxiter,
            all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius, 
            atom_max_neighbors=score_model_args.atom_max_neighbors, esm_embeddings_path=score_model_args.esm_embeddings_path, 
            require_ligand=True, num_workers=1
        )
    finally:
        restore_protein_mode_patch(patch_originals)
    test_loader = DataLoader(dataset=score_test_dataset, batch_size=1, shuffle=False)

    with open(f'{conf_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
        
    confidence_args.esm_embeddings_path = score_model_args.esm_embeddings_path
    confidence_args.protein_mode = protein_mode
    confidence_args.cache_path = get_cache_path_for_mode(confidence_args.cache_path, protein_mode, confidence=True)
    
    print('Loading Confidence TEST Dataset...')
    patch_originals = install_protein_mode_patch(score_model_args.data_dir, protein_mode)
    try:
        confidence_test_dataset = PDBBind(
            transform=None, root=score_model_args.data_dir, limit_complexes=score_model_args.limit_complexes,
            receptor_radius=confidence_args.receptor_radius, cache_path=confidence_args.cache_path, split_path=split_path_test,
            remove_hs=confidence_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
            matching=not confidence_args.no_torsion, keep_original=True, popsize=confidence_args.matching_popsize,
            maxiter=confidence_args.matching_maxiter, all_atoms=confidence_args.all_atoms,
            atom_radius=confidence_args.atom_radius, atom_max_neighbors=confidence_args.atom_max_neighbors,
            esm_embeddings_path=confidence_args.esm_embeddings_path, require_ligand=True, num_workers=1
        )
    finally:
        restore_protein_mode_patch(patch_originals)
    confidence_complex_dict = { (d.name[0] if isinstance(d.name, list) else d.name): d for d in confidence_test_dataset }

    # === 2. Initialize network structures ===
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
    
    conf_state_dict = torch.load(f'{conf_model_dir}/{conf_ckpt_path}', map_location=device)
    confidence_model.load_state_dict(conf_state_dict, strict=True)
    confidence_model = confidence_model.to(device).eval()
    
    tr_schedule = get_t_schedule(inference_steps=score_model_args.inference_steps)

    summary_headers = [
        'Model', 'Seed', 
        'Hybrid_Top1_Success', 'Vina_Top1_Success', 'Conf_Top1_Success',
        'Hybrid_Top5_Success', 'Vina_Top5_Success', 'Conf_Top5_Success',
        'Hybrid_Median_RMSD', 'Vina_Median_RMSD', 'Conf_Median_RMSD',
        'Hybrid_Top5_Median_RMSD', 'Vina_Top5_Median_RMSD', 'Conf_Top5_Median_RMSD',
        'Hybrid_Mean_Vina', 'Vina_Mean_Vina', 'Conf_Mean_Vina',
        'Hybrid_Top5_Mean_Vina', 'Vina_Top5_Mean_Vina', 'Conf_Top5_Mean_Vina',
        'Oracle_Top32_Success'
    ]

    evaluated_seeds = []
    rebuild_summary = False
    if os.path.exists(test_summary_path):
        try:
            df_existing = pd.read_csv(test_summary_path)
            if not all(col in df_existing.columns for col in summary_headers):
                print(f"⚠️ Detected legacy summary format, rebuilding from existing detail CSV files: {test_summary_path}")
                rebuild_summary = True
            else:
                evaluated_seeds = df_existing['Seed'].tolist()
                print(f"✅ Found existing summary. Evaluated Seeds: {evaluated_seeds}")
        except: pass

    if not os.path.exists(test_summary_path) or rebuild_summary:
        with open(test_summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_headers)
            writer.writeheader()

    if target_model == 'Baseline':
        ckpt_path = os.path.join(score_model_dir, 'best_ema_inference_epoch_model.pt')
        if not os.path.exists(ckpt_path): ckpt_path = os.path.join(score_model_dir, 'last_model.pt')
    else:
        ckpt_path = os.path.join(rl_ckpts_dir, f'rl_model_epoch_{target_model}.pt')

    state_dict = torch.load(ckpt_path, map_location=device)
    model_state = state_dict.get('ema_weights', state_dict.get('model_state_dict', state_dict.get('model', state_dict)))
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_state.items()}, strict=True)
    model = model.to(device).eval()

    # ==================================================
    # Inner loop: run multiple seeds
    # ==================================================
    for seed_idx in range(num_seeds):
        current_seed = base_seed + seed_idx
        
        # First guard: skip if summary already contains this record
        if current_seed in evaluated_seeds and not rebuild_summary:
            print(f"\n⏭️ Skipping Seed {current_seed} (Already evaluated in summary)")
            continue

        print(f"\n🧬 Model: [{target_model}] | Processing Seed: {current_seed} ({seed_idx+1}/{num_seeds})")
        detail_csv_path = f'{test_csv_dir}/test_detail_{target_model}{mode_file_suffix}_seed_{current_seed}.csv'

        # Second guard: if detail file exists, skip inference and reuse it
        if os.path.exists(detail_csv_path):
            print(f"   ⚡ Found existing detail file {os.path.basename(detail_csv_path)}")
            print("   ⚡ Skip model inference and rebuild summary directly from saved data...")
        else:
            set_seed(current_seed) 
            N = score_model_args.samples_per_complex
            csv_headers = [
                'complex_name', 
                'hybrid_top1_rmsd', 'hybrid_top1_vina', 'hybrid_top1_conf_score',
                'vina_top1_rmsd', 'vina_top1_vina', 
                'conf_top1_rmsd', 'conf_top1_vina', 'conf_top1_conf_score',
                'best_possible_vina', 'best_possible_rmsd', f'top_{N}_success_{rmsd_threshold}A',
                'all_32_rmsds', 'all_32_vinas', 'all_32_confs'
            ]
            for k in top_k_list:
                csv_headers.extend([
                    f'hybrid_top{k}_success_{rmsd_threshold}A', f'vina_top{k}_success_{rmsd_threshold}A', f'conf_top{k}_success_{rmsd_threshold}A',
                    f'hybrid_top{k}_best_rmsd', f'vina_top{k}_best_rmsd', f'conf_top{k}_best_rmsd'
                ])

            with open(detail_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()

            for idx, orig_complex_graph in enumerate(tqdm(test_loader)):
                orig_complex_graph = orig_complex_graph.to('cpu')
                
                for attr in ['mol', 'name']:
                    if hasattr(orig_complex_graph, attr) and isinstance(getattr(orig_complex_graph, attr), list):
                        setattr(orig_complex_graph, attr, getattr(orig_complex_graph, attr)[0])
                if hasattr(orig_complex_graph, 'original_center') and orig_complex_graph.original_center.dim() > 1:
                    orig_complex_graph.original_center = orig_complex_graph.original_center[0]

                if hasattr(orig_complex_graph, 'success') and not orig_complex_graph.success: continue
                complex_name = getattr(orig_complex_graph, 'name', f"complex_{idx}")
                if complex_name not in confidence_complex_dict: continue

                selected_protein_path = get_protein_path(score_model_args.data_dir, complex_name, protein_mode)
                if not os.path.exists(selected_protein_path):
                    print(f"⚠️ Skipping {complex_name}: protein file not found for mode={protein_mode} at {selected_protein_path}")
                    continue
                orig_complex_graph.protein_path = selected_protein_path
                    
                conf_complex_graph = confidence_complex_dict[complex_name].to('cpu')
                conf_complex_graph.protein_path = selected_protein_path
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
                            'all_32_rmsds': [round(float(x), 4) for x in rmsd_np],
                            'all_32_vinas': [round(float(x), 4) for x in vina_np],
                            'all_32_confs': [round(float(x), 4) for x in conf_np]
                        }
                        for k in top_k_list:
                            actual_k = min(k, N)
                            record[f'hybrid_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_hybrid[:actual_k]] < rmsd_threshold))
                            record[f'vina_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_vina[:actual_k]] < rmsd_threshold))
                            record[f'conf_top{k}_success_{rmsd_threshold}A'] = int(np.any(rmsd_np[idx_conf[:actual_k]] < rmsd_threshold))
                            record[f'hybrid_top{k}_best_rmsd'] = round(float(np.min(rmsd_np[idx_hybrid[:actual_k]])), 4)
                            record[f'vina_top{k}_best_rmsd'] = round(float(np.min(rmsd_np[idx_vina[:actual_k]])), 4)
                            record[f'conf_top{k}_best_rmsd'] = round(float(np.min(rmsd_np[idx_conf[:actual_k]])), 4)

                        with open(detail_csv_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=csv_headers)
                            writer.writerow(record)
                            
                except Exception as e:
                    pass
                finally:
                    torch.cuda.empty_cache()

        # Read detail data, compute metrics, and append to summary
        if os.path.exists(detail_csv_path):
            df_merged = pd.read_csv(detail_csv_path)
            summary_record = build_summary_record(
                df_merged=df_merged,
                target_model=target_model,
                current_seed=current_seed,
                rmsd_threshold=rmsd_threshold,
                samples_per_complex=score_model_args.samples_per_complex,
            )
            
            with open(test_summary_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_headers)
                writer.writerow(summary_record)
            evaluated_seeds.append(current_seed)

    # ==================================================
    # Model-level aggregation: detailed final report
    # ==================================================
    if os.path.exists(test_summary_path):
        df_all_seeds = pd.read_csv(test_summary_path)
        actual_seeds_run = len(df_all_seeds)
        
        if actual_seeds_run > 0:
            print(f"\n📊 FINAL AGGREGATION FOR MODEL [{target_model}] (Over {actual_seeds_run} Seeds):")
            print("=" * 80)
            
            h_succ_m, h_succ_s = df_all_seeds['Hybrid_Top1_Success'].mean(), df_all_seeds['Hybrid_Top1_Success'].std()
            v_succ_m, v_succ_s = df_all_seeds['Vina_Top1_Success'].mean(), df_all_seeds['Vina_Top1_Success'].std()
            c_succ_m, c_succ_s = df_all_seeds['Conf_Top1_Success'].mean(), df_all_seeds['Conf_Top1_Success'].std()
            
            h_top5_m, h_top5_s = df_all_seeds['Hybrid_Top5_Success'].mean(), df_all_seeds['Hybrid_Top5_Success'].std()
            v_top5_m, v_top5_s = df_all_seeds['Vina_Top5_Success'].mean(), df_all_seeds['Vina_Top5_Success'].std()
            c_top5_m, c_top5_s = df_all_seeds['Conf_Top5_Success'].mean(), df_all_seeds['Conf_Top5_Success'].std()
            
            ora_m, ora_s = df_all_seeds['Oracle_Top32_Success'].mean(), df_all_seeds['Oracle_Top32_Success'].std()

            h_med_m, h_med_s = df_all_seeds['Hybrid_Median_RMSD'].mean(), df_all_seeds['Hybrid_Median_RMSD'].std()
            v_med_m, v_med_s = df_all_seeds['Vina_Median_RMSD'].mean(), df_all_seeds['Vina_Median_RMSD'].std()
            c_med_m, c_med_s = df_all_seeds['Conf_Median_RMSD'].mean(), df_all_seeds['Conf_Median_RMSD'].std()
            h_top5_med_m, h_top5_med_s = df_all_seeds['Hybrid_Top5_Median_RMSD'].mean(), df_all_seeds['Hybrid_Top5_Median_RMSD'].std()
            v_top5_med_m, v_top5_med_s = df_all_seeds['Vina_Top5_Median_RMSD'].mean(), df_all_seeds['Vina_Top5_Median_RMSD'].std()
            c_top5_med_m, c_top5_med_s = df_all_seeds['Conf_Top5_Median_RMSD'].mean(), df_all_seeds['Conf_Top5_Median_RMSD'].std()

            h_vina_m, h_vina_s = df_all_seeds['Hybrid_Mean_Vina'].mean(), df_all_seeds['Hybrid_Mean_Vina'].std()
            v_vina_m, v_vina_s = df_all_seeds['Vina_Mean_Vina'].mean(), df_all_seeds['Vina_Mean_Vina'].std()
            c_vina_m, c_vina_s = df_all_seeds['Conf_Mean_Vina'].mean(), df_all_seeds['Conf_Mean_Vina'].std()
            h_top5_vina_m, h_top5_vina_s = df_all_seeds['Hybrid_Top5_Mean_Vina'].mean(), df_all_seeds['Hybrid_Top5_Mean_Vina'].std()
            v_top5_vina_m, v_top5_vina_s = df_all_seeds['Vina_Top5_Mean_Vina'].mean(), df_all_seeds['Vina_Top5_Mean_Vina'].std()
            c_top5_vina_m, c_top5_vina_s = df_all_seeds['Conf_Top5_Mean_Vina'].mean(), df_all_seeds['Conf_Top5_Mean_Vina'].std()

            def fmt(m, s): return f"{m:>5.1f}%" if pd.isna(s) else f"{m:>5.1f}% ± {s:<4.1f}%"
            def fmt_f(m, s): return f"{m:>5.2f}" if pd.isna(s) else f"{m:>5.2f}  ± {s:<4.2f}"

            print("Hybrid:")
            print(f"  Top-1 | Success: {fmt(h_succ_m, h_succ_s)} | Med RMSD: {fmt_f(h_med_m, h_med_s)} | Mean Vina: {fmt_f(h_vina_m, h_vina_s)}")
            print(f"  Top-5 | Success: {fmt(h_top5_m, h_top5_s)} | Med RMSD: {fmt_f(h_top5_med_m, h_top5_med_s)} | Mean Vina: {fmt_f(h_top5_vina_m, h_top5_vina_s)}")
            print("Vina:")
            print(f"  Top-1 | Success: {fmt(v_succ_m, v_succ_s)} | Med RMSD: {fmt_f(v_med_m, v_med_s)} | Mean Vina: {fmt_f(v_vina_m, v_vina_s)}")
            print(f"  Top-5 | Success: {fmt(v_top5_m, v_top5_s)} | Med RMSD: {fmt_f(v_top5_med_m, v_top5_med_s)} | Mean Vina: {fmt_f(v_top5_vina_m, v_top5_vina_s)}")
            print("Conf:")
            print(f"  Top-1 | Success: {fmt(c_succ_m, c_succ_s)} | Med RMSD: {fmt_f(c_med_m, c_med_s)} | Mean Vina: {fmt_f(c_vina_m, c_vina_s)}")
            print(f"  Top-5 | Success: {fmt(c_top5_m, c_top5_s)} | Med RMSD: {fmt_f(c_top5_med_m, c_top5_med_s)} | Mean Vina: {fmt_f(c_top5_vina_m, c_top5_vina_s)}")
            print("-" * 80)
            print(f"Oracle (32): {fmt(ora_m, ora_s)}")
            print("=" * 80)
            print(f"🎉 Test Completed! All detailed data saved in 👉 {test_csv_dir}")

if __name__ == '__main__':
    main()
