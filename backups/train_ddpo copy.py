import copy
import math
import os
import torch
import yaml
import numpy as np
from functools import partial
from torch_geometric.loader import DataLoader, DataListLoader # ensure imported
from torch.distributions import Normal
from argparse import Namespace
from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
# your sampling_rl
from utils.sampling_rl import sampling_rl, randomize_position
from tqdm import tqdm
from torch_geometric.data import Batch, Data


def set_train_mode_with_frozen_bn(model):
    """
    Enable training mode while freezing all BatchNorm layers.
    Standard setup for fine-tuning and RL.
    """
    model.train() # enable global train mode (dropout, gradients, etc.)
    
    # Iterate through submodules and force BN layers back to eval mode
    for module in model.modules():
        # Match both e3nn BatchNorm and native PyTorch BatchNorm
        if isinstance(module, torch.nn.BatchNorm1d) or \
           isinstance(module, torch.nn.BatchNorm2d) or \
           isinstance(module, torch.nn.BatchNorm3d) or \
           "BatchNorm" in module.__class__.__name__: # for e3nn.nn.BatchNorm
            
            module.eval() 
            # Optionally disable gradients for extra safety
            for param in module.parameters():
                param.requires_grad = False
# ----------------------------------------------------------------------
# 1. Log-probability computation (core revised version)
# ----------------------------------------------------------------------
def get_step_log_prob(model, replay_batch, step_data, t_tr, t_rot, t_tor, dt_tr, dt_rot, dt_tor, t_to_sigma, model_args, device):
    """
    replay_batch: a large graph with N replicas (Batch Size = samples_per_complex)
    step_data: trajectory data at current timestep t
    """
    # 1. Restore state s_t
    # step_data['obs_pos'] is a list of tensors (molecules may have different atom counts)
    # Concatenate them to match PyG batch format
    # Assume sampling_rl stores list of [num_atoms, 3] tensors
    
    # obs_pos is the position list for N molecules
    obs_pos_list = step_data['obs_pos'] 
    
    # Merge position list into one large tensor [Total_Atoms_In_Batch, 3]
    # Ensure order matches graph order in replay_batch
    current_pos = torch.cat(obs_pos_list, dim=0).to(device)
    
    # Key fix: update positions in replay_batch
    # This in-place update is safe since replay_batch is refreshed each loop
    replay_batch['ligand'].pos = current_pos 

    # 2. Model forward pass
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
    # Note: set_time needs batch size, here replay_batch.num_graphs
    set_time(replay_batch, t_tr, t_rot, t_tor, replay_batch.num_graphs, model_args.all_atoms, device)
    
    # Get scores
    tr_score, rot_score, tor_score = model(replay_batch)

    # 3. Compute distribution parameters
    tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min), device=device))
    
    # --- Translation ---
    tr_mu = (tr_g ** 2 * dt_tr * tr_score)
    tr_std = (tr_g * np.sqrt(dt_tr))
    
    tr_dist = Normal(tr_mu, tr_std)
    
    # Get action (a_t) and move to GPU
    action_tr = step_data['action_tr'].to(device)
    
    # Compute log-probabilities
    # tr_score and action_tr are both [N, 3], so dimensions align
    new_tr_log_prob = tr_dist.log_prob(action_tr).sum(dim=-1)

    # --- Torsion ---
    new_tor_log_prob = torch.zeros(replay_batch.num_graphs, device=device)
    
    if not model_args.no_torsion and 'action_tor' in step_data:
        # Keep torsion logic unchanged
        tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min), device=device))
        tor_mu = (tor_g ** 2 * dt_tor * tor_score)
        tor_std = (tor_g * np.sqrt(dt_tor))
        
        tor_dist = Normal(tor_mu, tor_std)
        action_tor = step_data['action_tor'].to(device)
        
        raw_tor_log_prob = tor_dist.log_prob(action_tor)
        
        new_tor_log_prob_list = []
        start_idx = 0
        indices = step_data['tor_indices']
        
        for n_tor in indices:
            if n_tor > 0:
                mol_lp = raw_tor_log_prob[start_idx : start_idx + n_tor].sum()
                start_idx += n_tor
            else:
                mol_lp = torch.tensor(0.0, device=device)
            new_tor_log_prob_list.append(mol_lp)
        
        new_tor_log_prob = torch.stack(new_tor_log_prob_list)

    return new_tr_log_prob, new_tor_log_prob

# ----------------------------------------------------------------------
# 2. Mock Reward
# ----------------------------------------------------------------------
def calc_mock_reward(data_list):
    rewards = []
    for graph in data_list:
        # Simple test: make all atoms move toward origin (minimize norm)
        # Reward = - Mean Distance to Origin
        pos = graph['ligand'].pos
        score = -1.0 * pos.norm(dim=1).mean().item()
        rewards.append(score)
    return torch.tensor(rewards)

# ----------------------------------------------------------------------
# 3. Train epoch (core revised version)
# ----------------------------------------------------------------------
def train_rl_epoch(args, model, train_loader, optimizer, t_to_sigma, device, tr_schedule, rot_schedule, tor_schedule):
    # model.train()
    total_reward = 0
    total_loss = 0
    complex_count = 0
    
    # Outer loop: each step processes one protein (Batch Size = 1)
    for idx, batch_data in enumerate(tqdm(train_loader)):
        
        # 1. Data preprocessing: normalize into PyG Batch or Data object
        # For DataListLoader, batch_data is a list
        # For DataLoader, batch_data is a Batch object
        if isinstance(batch_data, list):
            # DataListLoader compatibility: take first element
            orig_complex_graph = batch_data[0] 
        else:
            # DataLoader compatibility (batch size=1 gives the target graph)
            # DataLoader returns Batch; treat as single Data or clone directly
            orig_complex_graph = batch_data.clone()
        
        # Keep data on CPU for easier deepcopy
        orig_complex_graph = orig_complex_graph.to('cpu')

        # -----------------------------------------------------------
        # Step 1: Build RL batch (duplicate N copies)
        # -----------------------------------------------------------
        # Critical: duplicate one protein graph by samples_per_complex
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.samples_per_complex)]
        
        # Randomly initialize positions
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
        
        # -----------------------------------------------------------
        # Step 2: Rollout (Sampling)
        # -----------------------------------------------------------
        model.eval()
        with torch.no_grad():
            final_data_list, trajectories = sampling_rl(
                data_list=data_list,
                model=model,
                inference_steps=args.inference_steps,
                tr_schedule=tr_schedule,
                rot_schedule=rot_schedule,
                tor_schedule=tor_schedule,
                device=device,
                t_to_sigma=t_to_sigma,
                model_args=args,
                batch_size=args.samples_per_complex 
            )
            
        # -----------------------------------------------------------
        # Step 3: Compute Reward
        # -----------------------------------------------------------
        rewards = calc_mock_reward(final_data_list).to(device)
        
        # Advantage normalization (per-prompt)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        total_reward += mean_reward.item()
        complex_count += 1
        
        # -----------------------------------------------------------
        # Step 4: Update (PPO)
        # -----------------------------------------------------------
        # Key fix: construct replay_batch
        # Need a Batch with N graphs for model input during updates
        # Use PyG DataLoader to convert data_list back to Batch
        # This guarantees correct replay_batch.batch indices [0,0,0,1,1,1,...]
        # model.train()
        set_train_mode_with_frozen_bn(model)
        rl_loader = DataLoader(data_list, batch_size=args.samples_per_complex, shuffle=False)
        replay_batch = next(iter(rl_loader)).to(device) # get the only batch
        
        optimizer.zero_grad()
        loss_complex = 0
        num_steps = len(trajectories)
        
        # Iterate over timesteps
        for t_idx, step_data in enumerate(trajectories):
            # Delta time
            t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
            dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx+1] if t_idx < num_steps - 1 else tr_schedule[t_idx]
            dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx+1] if t_idx < num_steps - 1 else rot_schedule[t_idx]
            dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx+1] if t_idx < num_steps - 1 else tor_schedule[t_idx]
            
            # Old Log Prob
            old_log_prob = step_data['log_prob_tr'].to(device) + step_data['log_prob_tor'].to(device)
            
            # New log-probabilities (with gradients)
            # Note: pass replay_batch (size N), not batch_data (size 1)
            new_log_prob_tr, new_log_prob_tor = get_step_log_prob(
                model, replay_batch, step_data, 
                t_tr, t_rot, t_tor, 
                dt_tr, dt_rot, dt_tor, 
                t_to_sigma, args, device
            )
            new_log_prob = new_log_prob_tr + new_log_prob_tor
            
            # PPO Loss
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            
            # Maximize reward by minimizing negative loss
            loss_step = -torch.min(surr1, surr2).mean()
            loss_complex += loss_step

        loss_complex = loss_complex / num_steps
        
        # Backward pass
        loss_complex.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if idx % 10 == 0:
            print(f"Iter {idx} | Mean Reward: {mean_reward.item():.4f} | Loss: {loss_complex.item():.4f}")

    return total_loss / (complex_count + 1e-8), total_reward / (complex_count + 1e-8)

# ----------------------------------------------------------------------
# 4. Main Function
# ----------------------------------------------------------------------
def main_rl():
    model_dir = 'workdir/paper_score_model' # change to your model path
    if not os.path.exists(f'{model_dir}/model_parameters.yml'):
        # If yml is missing, build args manually (debug only)
        # In practice, point to the correct config path
        pass 
        
    with open(f'{model_dir}/model_parameters.yml') as f:
        args = Namespace(**yaml.full_load(f))
    
    # Force RL parameters
    args.batch_size = 1  # DataLoader reads one protein at a time
    args.samples_per_complex = 8 # tune by VRAM, 16/32 for larger GPUs
    args.inference_steps = 20
    args.lr = 5e-6 # RL learning rate is typically lower than pretraining
    args.n_epochs = 50
    args.num_workers = 0 
    
    # Path checks
    if not hasattr(args, 'split_train'): args.split_train = 'data/splits/timesplit_no_lig_overlap_train'
    if not hasattr(args, 'split_val'): args.split_val = 'data/splits/timesplit_no_lig_overlap_val'
    # Use a new cache path to avoid conflicts with original pipeline
    args.cache_path = 'data/cache_rl_debug' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Data loading (use your adjusted construct_loader without noise)
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    # Note: construct_loader here should be the modified version without noise transform
    # Or set transform to None manually if needed
    train_loader, val_loader = construct_loader(args, t_to_sigma, device) 
    print(f"Train loader size: {len(train_loader)}")

    # Model loading
    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    ckpt_path = f'{model_dir}/best_ema_inference_epoch_model.pt'
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained model.")
    else:
        print("Warning: Pretrained model not found, using random init.")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    
    for epoch in range(args.n_epochs):
        print(f"\n=== Starting RL Epoch {epoch} ===")
        avg_loss, avg_rew = train_rl_epoch(
            args, model, train_loader, optimizer, 
            t_to_sigma, device, 
            tr_schedule, rot_schedule, tor_schedule
        )
        
        # Save
        save_path = f"results/rl_train/rl_model_epoch_{epoch}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main_rl()