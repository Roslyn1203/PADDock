import os
import time
import yaml
import torch
import numpy as np
import copy
import gc
import math
import glob
import re
import csv
from functools import partial
from argparse import Namespace
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Normal
from datetime import timedelta # newly added

# Import project modules
from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.utils import get_model
from utils.sampling_rl import sampling_rl, randomize_position
from utils.vina_scoring import calc_vina_rewards
from torch_geometric.data import Batch
from utils.ddpo_utils import save_debug_visualization, get_step_log_prob, manual_all_reduce_grads, set_train_mode_with_frozen_bn, BatchSizeManager
from rdkit import Chem
import shutil 
import os
import csv
from utils.guidance import GammaScheduler
def train_rl_epoch(args, model, train_loader, optimizer, t_to_sigma, device, tr_schedule, rot_schedule, tor_schedule):
    scaler = GradScaler() 
    
    local_loss = 0
    local_vina = 0
    local_rmsd = 0
    local_reward = 0
    valid_steps = 0 
    
    rank = int(os.environ.get("LOCAL_RANK", 0))
    iterator = tqdm(train_loader) if rank == 0 else train_loader
    bs_manager = BatchSizeManager(rank)

    for idx, batch_data in enumerate(iterator):
        t0 = time.time()
        
        # 1. Data preparation
        if isinstance(batch_data, list): orig_complex_graph = batch_data[0]
        else: orig_complex_graph = batch_data.clone()
        orig_complex_graph = orig_complex_graph.to('cpu')

        if hasattr(orig_complex_graph, 'mol') and isinstance(orig_complex_graph.mol, list):
            orig_complex_graph.mol = orig_complex_graph.mol[0]
        if hasattr(orig_complex_graph, 'name') and isinstance(orig_complex_graph['name'], list):
            orig_complex_graph.name = orig_complex_graph['name'][0]
        if hasattr(orig_complex_graph, 'original_center') and orig_complex_graph.original_center.dim() > 1:
            orig_complex_graph.original_center = orig_complex_graph.original_center[0]

        complex_name = orig_complex_graph.name

        # === Key change: unified skip logic must use manual_all_reduce_grads ===
        
        # Internal helper for synchronization when skipping a sample
        def skip_synchronize():
            optimizer.zero_grad() # clear gradients
            # Manually sync zero gradients to avoid deadlock across ranks
            manual_all_reduce_grads(model) 
            # No optimizer step when skipping
        
        # Check cache and decide behavior
        cached_bs = bs_manager.get(complex_name)
        if cached_bs is not None and cached_bs <= 0:
            skip_synchronize()
            continue

        # Pre-filter checks
        if cached_bs is None:
            should_skip = False
            # A. Boron atom filter
            if hasattr(orig_complex_graph, 'mol'):
                atoms = [atom.GetSymbol() for atom in orig_complex_graph.mol.GetAtoms()]
                if 'B' in atoms:
                    print(f"[Rank {rank}] ⚠️ Skip Boron complex {idx}")
                    bs_manager.update(complex_name, -1)
                    should_skip = True
            
            # B. Atom-count filter
            if not should_skip:
                num_atoms = orig_complex_graph['receptor'].pos.shape[0] + orig_complex_graph['ligand'].pos.shape[0]
                if num_atoms > 1500:
                    print(f"[Rank {rank}] Too huge ({num_atoms}), caching 0.")
                    bs_manager.update(complex_name, 0)
                    should_skip = True
            
            if should_skip:
                skip_synchronize()
                continue

        t1 = time.time()
        # === New: read GDD priors from modified dataset ===
        # Assume attributes like spheres are attached to graph in construct_loader
        # (implementation depends on your custom dataset modification strategy)
        desired_sphere = getattr(orig_complex_graph, 'desired_sphere', None)
        desired_rot = getattr(orig_complex_graph, 'desired_rot', None)
        desired_tor = getattr(orig_complex_graph, 'desired_tor', None)
        # 4. Sampling
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.samples_per_complex)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
        num_atoms_per_sample = data_list[0]['ligand'].pos.shape[0]

        model.eval() 
        try:
            with torch.no_grad():
                final_data_list, trajectories = sampling_rl(
                    data_list=data_list, model=model, inference_steps=args.inference_steps,
                    tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                    device=device, t_to_sigma=t_to_sigma, model_args=args, batch_size=args.samples_per_complex 
                )
        except RuntimeError:
             # Keep synchronization even when sampling fails
             torch.cuda.empty_cache()
             skip_synchronize()
             continue
             
        t2 = time.time()
        # # ==========================================
        # # [New] Save debug visualization (only rank 0 and first batch)
        # # ==========================================
        # if idx == 4: 
        #     # print("[Debug] Saving visualization (Receptor + Ligands)...")
        #     # orig_complex_graph has already been moved to CPU
        #     save_debug_visualization(orig_complex_graph, final_data_list, args, save_dir="./debug_vis")
        # ==========================================
        # 5. Reward (single call to get both Vina and RMSD)
        vina_rewards, rmsd_tensor = calc_vina_rewards(final_data_list, orig_complex_graph, args)
        
        # --- 5. Reward processing ---
        vina_rewards = vina_rewards.to(device, dtype=torch.float32)
        rmsd_tensor = rmsd_tensor.to(device, dtype=torch.float32)
        
        # Tuned setting: use RMSD weight for in-pocket selection
        rmsd_weight = 1.0

        # Introduce nonlinear reward signal
        total_rewards = (vina_rewards - (rmsd_weight * rmsd_tensor))

        # Separate bad samples from valid samples
        bad_mask = rmsd_tensor > 10.0
        valid_mask = ~bad_mask
        
        # Initialize zero advantages
        advantages = torch.zeros_like(total_rewards)
        
        # Initialize default stats
        mean_reward = torch.tensor(0.0, device=total_rewards.device)
        std_reward = torch.tensor(0.0, device=total_rewards.device)
        
        # Fix: mean can be computed when at least one valid sample exists
        if valid_mask.sum() > 0:
            valid_rewards = total_rewards[valid_mask]
            mean_reward = valid_rewards.mean() # compute true reward mean whenever valid samples exist
            
            # Fix: variance/normalization requires more than one valid sample
            if valid_mask.sum() > 1:
                std_reward = valid_rewards.std()
                
                # Apply Z-score normalization only on valid samples
                advantages[valid_mask] = (valid_rewards - mean_reward) / (std_reward + 1e-6)
                advantages = advantages * 2.0
            # (if only one valid sample exists, zero advantage is expected and correct)

        # Keep as a final safety clamp
        advantages = torch.clamp(advantages, min=-3.0, max=3.0)

        # Keep metrics for logging
        if valid_mask.sum() > 0:
            current_vina_mean = vina_rewards[valid_mask].mean().item()
            current_rmsd_mean = rmsd_tensor[valid_mask].mean().item()
        else:
            current_vina_mean = 0.0
            current_rmsd_mean = 0.0
            
        current_reward_mean = mean_reward.item()

        t3 = time.time()

        # 6. Update
        set_train_mode_with_frozen_bn(model)

        def execute_update_pass(mini_bs):
            optimizer.zero_grad(set_to_none=True) # memory-saving: set_to_none=True
            total_samples = args.samples_per_complex
            loss_accum = 0
            
            num_steps = len(trajectories)
            start_train_idx = int(num_steps * 0.7) 
            actual_trainable_steps = num_steps - start_train_idx
            num_mini_batches = math.ceil(total_samples / mini_bs)

            for i in range(0, total_samples, mini_bs):
                end_i = min(i + mini_bs, total_samples)
                
                # --- Memory optimization: explicit cleanup every iteration ---
                if i > 0: torch.cuda.empty_cache() 
                
                current_data_slice = data_list[i : end_i]
                current_adv = advantages[i : end_i].to(device) 
                
                # Precompute mask to avoid repeated work in t-loop
                # --- Core fix: remove incorrect Vina>0 interception, keep only RMSD safeguard ---
                bad_mask = (rmsd_tensor[i : end_i] > 10.0)
                current_adv[bad_mask] = 0.0 

                loss_this_batch = 0 
                
                for t_idx, step_data in enumerate(trajectories):
                    if t_idx < start_train_idx:
                        continue 
                    mini_replay_batch = Batch.from_data_list(current_data_slice).to(device)
                    # --- Memory optimization: move CPU tensors to device immediately ---
                    t_tr, t_rot, t_tor = step_data['t_tr'], step_data['t_rot'], step_data['t_tor']
                    dt_tr, dt_rot, dt_tor = step_data['dt_tr'], step_data['dt_rot'], step_data['dt_tor']

                    # --- Memory optimization: move actions to GPU only when needed ---
                    m_act_tr = step_data['action_tr'][i : end_i].to(device)
                    m_act_rot = step_data['action_rot'][i : end_i].to(device)
                    m_pos = step_data['obs_pos'][i*num_atoms_per_sample : end_i*num_atoms_per_sample].to(device)
                    
                    m_act_tor = None
                    m_tor_indices = []
                    if 'action_tor' in step_data and step_data['action_tor'] is not None:
                        m_tor_indices = step_data['tor_indices'][i : end_i]
                        s_t = sum(step_data['tor_indices'][:i])
                        e_t = s_t + sum(m_tor_indices)
                        m_act_tor = step_data['action_tor'][s_t : e_t].to(device)

                    # Load old log-probabilities and move to GPU
                    lp_tr_o = step_data['log_prob_tr'][i:end_i].to(device)
                    lp_rot_o = step_data['log_prob_rot'][i:end_i].to(device)
                    lp_tor_o = step_data['log_prob_tor'][i:end_i].to(device)
                    curr_old_lp = lp_tr_o + lp_rot_o + lp_tor_o

                    with autocast(dtype=torch.bfloat16):
                        # Use already moved device tensors
                        new_lp_tr, new_lp_rot, new_lp_tor = get_step_log_prob(
                            model, mini_replay_batch, 
                            m_pos, m_act_tr, m_act_rot, m_act_tor, m_tor_indices,
                            t_tr, t_rot, t_tor, dt_tr, dt_rot, dt_tor, t_to_sigma, args, device
                        )
                        
                        new_lp_total = new_lp_tr + new_lp_rot + new_lp_tor
                        ratio = torch.exp(new_lp_total - curr_old_lp)
                        
                        # # ... (alignment test logic omitted) ...
                        # # Policy alignment test (omitted)
                        # if idx == 0 and i == 0 and t_idx == start_train_idx:
                        #     diff = torch.abs(new_lp_total - curr_old_lp).mean().item()
                        #     print(f"\n[RL Alignment Test] Mean LogProb Diff: {diff:.6f}")
                        #     if diff > 1e-2:
                        #         print(f"[Warning] ❌ Policy mismatch detected! Ratio Mean: {ratio.mean().item():.4f}")
                        #         # Print first three samples to inspect mismatch source
                        #         print(f"New LP: {new_lp_total[:3].detach().cpu().numpy()}")
                        #         print(f"Old LP: {curr_old_lp[:3].detach().cpu().numpy()}")
                        #     else:
                        #         print(f"[Success] ✅ Policy alignment passed.")
                        loss_step = -torch.min(ratio * current_adv, torch.clamp(ratio, 0.8, 1.2) * current_adv).mean()
                        loss_bw = loss_step / num_mini_batches

                    scaler.scale(loss_bw).backward()
                    loss_this_batch += loss_step.item()
                    
                    # --- Memory optimization: aggressive cleanup ---
                    del new_lp_total, ratio, loss_step, loss_bw, m_act_tr, m_act_rot, m_pos, m_act_tor, curr_old_lp
                
                # Cleanup large objects for current batch
                del mini_replay_batch, current_data_slice, current_adv
                loss_accum += (loss_this_batch / actual_trainable_steps)
            
            return loss_accum / num_mini_batches
        # Retry loop
        if cached_bs is not None and cached_bs > 0:
            try_list = [cached_bs] 
            is_profiling = False
        else:
            try_list = [16, 8, 4]
            is_profiling = True

        success = False
        
        # Key: use no_sync context
        with model.no_sync():
            for try_bs in try_list:
                try:
                    current_loss = execute_update_pass(try_bs)
                    if is_profiling:
                        bs_manager.update(complex_name, try_bs)
                    success = True
                    break 
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if not is_profiling:
                            print(f"[Rank {rank}] ❌ Unexpected OOM with cached BS={try_bs}.")
                            bs_manager.update(complex_name, try_bs // 2)
                            break
                        continue
                    else:
                        raise e

        if not success:
            if is_profiling:
                bs_manager.update(complex_name, 0)
            optimizer.zero_grad() 
        
        # Key: all branches converge here for manual synchronization
        manual_all_reduce_grads(model)
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        t4 = time.time()

        if success:
            local_loss += current_loss
            local_vina += current_vina_mean
            local_rmsd += current_rmsd_mean
            local_reward += current_reward_mean
            valid_steps += 1
            
            if valid_steps % 10 == 0 and rank == 0:
                print(f"Iter {idx} | Reward: {current_reward_mean:.4f} | Loss: {current_loss:.4f} | Vina: {current_vina_mean:.2f} | RMSD: {current_rmsd_mean:.2f}")
            if rank == 0:
                print(f"Time: Load={t1-t0:.2f}, Sample={t2-t1:.2f}, Score={t3-t2:.2f}, Update={t4-t3:.2f}")       

    # Epoch End
    metrics_tensor = torch.tensor([
        local_loss, local_vina, local_rmsd, local_reward, valid_steps
    ], device=device, dtype=torch.float32)

    if dist.is_initialized():
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    global_sum_loss = metrics_tensor[0].item()
    global_sum_vina = metrics_tensor[1].item()
    global_sum_rmsd = metrics_tensor[2].item()
    global_sum_reward = metrics_tensor[3].item()
    global_total_steps = metrics_tensor[4].item()

    if global_total_steps > 0:
        avg_loss = global_sum_loss / global_total_steps
        avg_vina = global_sum_vina / global_total_steps
        avg_rmsd = global_sum_rmsd / global_total_steps
        avg_rew  = global_sum_reward / global_total_steps
    else:
        avg_loss, avg_vina, avg_rmsd, avg_rew = 0.0, 0.0, 0.0, 0.0

    return avg_loss, avg_vina, avg_rmsd, avg_rew



def main_rl():
    # --- A. Distributed initialization ---
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=180))
        device = torch.device(f'cuda:{local_rank}')
        is_distributed = True
        if local_rank == 0:
            print(f"Distributed training enabled. Local Rank: {local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False
        print("Single GPU mode.")

    # --- B. Parameter setup ---
    model_dir = 'workdir/paper_score_model'
    with open(f'{model_dir}/model_parameters.yml') as f:
        args = Namespace(**yaml.full_load(f))
    
    args.batch_size = 1
    args.inference_steps = 10      
    args.samples_per_complex = 16   
    
    # Speed-test setting: increase learning rate for short-run validation
    args.lr = 5e-6
    args.n_epochs = 100
    args.num_workers = 8

    # Result save path
    save_dir = "./results/ddpo_train_600"
    os.makedirs(save_dir, exist_ok=True)

    # New: initialize metric logger (main process only)
    csv_file_path = os.path.join(save_dir, "training_stats.csv")

    # --- C & D. Data loading and model initialization ---
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, _ = construct_loader(args, t_to_sigma, device)

    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- E. Resume training logic (auto-resume) ---
    start_epoch = 0
    existing_ckpts = glob.glob(os.path.join(save_dir, "rl_model_epoch_*.pt"))
    latest_ckpt_path = None
    latest_epoch = -1

    if existing_ckpts:
        for path in existing_ckpts:
            match = re.search(r'rl_model_epoch_(\d+).pt', path)
            if match:
                ep = int(match.group(1))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_ckpt_path = path

    if latest_ckpt_path is not None:
        if dist.get_rank() == 0 if is_distributed else True:
            print(f"Found existing checkpoint: {latest_ckpt_path}. Resuming from Epoch {latest_epoch + 1}...")
        
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            if is_distributed:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            if is_distributed:
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            start_epoch = latest_epoch + 1
    else:
        ckpt_path = f'{model_dir}/best_ema_inference_epoch_model.pt'
        if os.path.exists(ckpt_path):
            if dist.get_rank() == 0 if is_distributed else True:
                print(f"No RL checkpoint found. Loading pretrained base model: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            if is_distributed:
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

    # New: write CSV header for fresh run, append for resumed run
    if (is_distributed == False or dist.get_rank() == 0):
        write_mode = 'a' if start_epoch > 0 else 'w'
        with open(csv_file_path, write_mode, newline='') as f:
            writer = csv.writer(f)
            if write_mode == 'w':
                writer.writerow(['Epoch', 'Loss', 'Vina_Score', 'RMSD', 'Total_Reward'])

    # --- F. Training loop ---
    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    # args should include gamma_scheduler setting, e.g. 'default', 'onoff', 'warmup_cooldown'
    args.gamma_scheduler = getattr(args, 'gamma_scheduler', 'default') 
    
    tr_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, args.inference_steps)
    rot_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, args.inference_steps)
    tor_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, args.inference_steps)
    for epoch in range(start_epoch, args.n_epochs):
        if is_distributed and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)
            
        if dist.get_rank() == 0 if is_distributed else True:
            print(f"\n=== Starting RL Epoch {epoch} ===")
            
        avg_loss, avg_vina, avg_rmsd, avg_rew = train_rl_epoch(
            args, model, train_loader, optimizer, 
            t_to_sigma, device, 
            tr_schedule, rot_schedule, tor_schedule,
            tr_gamma_schedule, rot_gamma_schedule, tor_gamma_schedule
        )
        
        # --- Metric aggregation and saving ---
        if dist.get_rank() == 0 if is_distributed else True:
            print(f"Epoch {epoch} Done. Loss: {avg_loss:.4f}, Vina: {avg_vina:.4f}, RMSD: {avg_rmsd:.4f}, Reward: {avg_rew:.4f}")
            
            # Write metrics to CSV for later analysis/plotting
            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss, avg_vina, avg_rmsd, avg_rew])
            
            # Save checkpoint
            save_path = os.path.join(save_dir, f"rl_model_epoch_{epoch}.pt")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args 
            }
            torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main_rl()