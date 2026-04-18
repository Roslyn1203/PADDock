from utils.diffusion_utils import set_time
import torch
import numpy as np


def get_step_log_prob(model, replay_batch, current_pos_tensor, action_tr_tensor, action_rot_tensor, action_tor_tensor, 
                      tor_indices, t_tr, t_rot, t_tor, dt_tr, dt_rot, dt_tor, t_to_sigma, model_args, device):
    model.eval()
    
    # 1. Prepare inputs
    replay_batch['ligand'].pos = current_pos_tensor.to(device)
    set_time(replay_batch, t_tr, t_rot, t_tor, replay_batch.num_graphs, model_args.all_atoms, device)
    
    # 2. Memory-optimized forward pass (single call site)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        tr_score_bf, rot_score_bf, tor_score_bf = model(replay_batch)
    
    # Memory tip: cast to float32 immediately for math operations
    tr_score = tr_score_bf.float()
    rot_score = rot_score_bf.float()
    tor_score = tor_score_bf.float()
    
    # Numerical checks and cleanup
    def safe_clamp(tensor):
        t = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
        return torch.clamp(t, min=-1e4, max=1e4)

    tr_score = safe_clamp(tr_score)
    rot_score = safe_clamp(rot_score)
    tor_score = safe_clamp(tor_score)
    
    # 3. Prepare sigma values and constants (explicit float32 to avoid implicit numpy casting)
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
    tr_sigma_v = torch.tensor(tr_sigma, device=device, dtype=torch.float32)
    rot_sigma_v = torch.tensor(rot_sigma, device=device, dtype=torch.float32)
    tor_sigma_v = torch.tensor(tor_sigma, device=device, dtype=torch.float32)
    
    tr_g = tr_sigma_v * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min), device=device, dtype=torch.float32))
    rot_g = 2 * rot_sigma_v * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min), device=device, dtype=torch.float32))
    tor_g = tor_sigma_v * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min), device=device, dtype=torch.float32))

    dt_tr_t = torch.tensor(dt_tr, device=device, dtype=torch.float32)
    dt_rot_t = torch.tensor(dt_rot, device=device, dtype=torch.float32)
    dt_tor_t = torch.tensor(dt_tor, device=device, dtype=torch.float32)

    # --- A. Translation ---
    tr_mu = (tr_g ** 2 * dt_tr_t * tr_score)
    tr_std = torch.clamp((tr_g * torch.sqrt(dt_tr_t)), min=1e-6) 
    new_tr_log_prob = torch.distributions.Normal(tr_mu, tr_std).log_prob(action_tr_tensor.to(device)).sum(dim=-1)

    # --- B. Rotation ---
    rot_mu = (rot_g ** 2 * dt_rot_t * rot_score)
    rot_std = torch.clamp((rot_g * torch.sqrt(dt_rot_t)), min=1e-6)
    new_rot_log_prob = torch.distributions.Normal(rot_mu, rot_std).log_prob(action_rot_tensor.to(device)).sum(dim=-1)
    
    # --- C. Torsion ---
    new_tor_log_prob = torch.zeros(replay_batch.num_graphs, device=device)
    if not model_args.no_torsion and action_tor_tensor is not None:
        tor_mu = (tor_g ** 2 * dt_tor_t * tor_score)
        tor_std = torch.clamp((tor_g * torch.sqrt(dt_tor_t)), min=1e-6)

        raw_tor_log_prob = torch.distributions.Normal(tor_mu, tor_std).log_prob(action_tor_tensor.to(device))
        
        split_sizes = tor_indices.tolist() if torch.is_tensor(tor_indices) else tor_indices
        split_log_probs = torch.split(raw_tor_log_prob, split_sizes)
        
        new_tor_log_prob = torch.stack([
            chunk.sum() if chunk.numel() > 0 else torch.tensor(0.0, device=device) 
            for chunk in split_log_probs
        ])

    return new_tr_log_prob, new_rot_log_prob, new_tor_log_prob