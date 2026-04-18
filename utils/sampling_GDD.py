import numpy as np
import torch
import random
import copy
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles, AngleCalcMethod

# === Import GDD operators ===
from utils.guidance import (
    tr_guider,
    in_tr_region,
    compute_tr_gamma,
    get_guided_tr_update,
    get_rot_state,
    rot_guider,
    in_rot_region,
    get_guided_rotation_matrix,
    get_tor_state,
    tor_guider,
    in_torus_region,
    compute_tor_gamma,
    get_guided_tor_update,
)

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # (keep unchanged)
    if not no_torsion:
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = modify_conformer_torsion_angles(
                complex_graph['ligand'].pos,
                complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask],
                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        if not no_random:
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update


# === Added kwargs required by GDD guidance ===
def sampling_rl(
    data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
    batch_size=32, visualization_list=None,
    tr_guiding=False, desired_sphere=None, tr_gamma_schedule=None,
    rot_guiding=False, desired_rot=None, rot_gamma_schedule=None,
    tor_guiding=False, desired_tor=None, tor_gamma_schedule=None,
    dynamic_gamma=False, angle_calc_method=AngleCalcMethod.TOR_CALC_2,
    Rm_update_method="m0", neg_vdir=True, mask_n_tor=None, mask_n_distribution="random"
):
    N = len(data_list)
    trajectory_data = [] 

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        # Extract gamma for current step
        tr_gamma = tr_gamma_schedule[t_idx] if tr_gamma_schedule else 0.0
        rot_gamma = rot_gamma_schedule[t_idx] if rot_gamma_schedule else 0.0
        tor_gamma = tor_gamma_schedule[t_idx] if tor_gamma_schedule else 0.0

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        new_data_list = []
        
        step_obs_pos = []       
        step_tr_action = []     
        step_tr_log_prob = []   
        step_rot_action = []    
        step_rot_log_prob = []  
        step_tor_action = []    
        step_tor_log_prob = []  
        step_tor_indices = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            step_obs_pos.append(complex_graph_batch['ligand'].pos.detach().cpu())

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    tr_score, rot_score, tor_score = model(complex_graph_batch)
            
            def soft_clean(s):
                return torch.nan_to_num(s.detach(), nan=0.0, posinf=1e3, neginf=-1e3).to(torch.float32).cpu()

            tr_score_c = soft_clean(tr_score)
            rot_score_c = soft_clean(rot_score)
            tor_score_c = soft_clean(tor_score)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
            dt_tr_t = torch.tensor(dt_tr, dtype=torch.float32)
            dt_rot_t = torch.tensor(dt_rot, dtype=torch.float32)
            dt_tor_t = torch.tensor(dt_tor, dtype=torch.float32)

            cpu_batch = complex_graph_batch.to('cpu').to_data_list()

            # --- A. Translation ---
            tr_mu = (tr_g ** 2 * dt_tr_t * tr_score_c)
            tr_std = (tr_g * torch.sqrt(dt_tr_t)).clamp(min=1e-6) 
            tr_perturb_raw = tr_mu + tr_std * torch.randn_like(tr_mu)
            
            guided_tr_actions = []
            for i, graph in enumerate(cpu_batch):
                tr_state = torch.mean(graph["ligand"].pos, dim=0, keepdim=True).numpy()
                tr_update_np = tr_perturb_raw[i:i+1].numpy()
                
                if tr_guiding and not in_tr_region(tr_state, desired_sphere):
                    tr_vdir, tr_distance = tr_guider(pos=tr_state, sph=desired_sphere)
                    current_gamma = tr_gamma
                    if dynamic_gamma:
                        current_gamma, _ = compute_tr_gamma(graph["receptor"].diameter.item() if hasattr(graph["receptor"], "diameter") else 100.0, tr_update_np, tr_vdir, tr_distance)
                    
                    guided_tr_update = get_guided_tr_update(tr_state, tr_vdir, tr_distance, tr_update_np, current_gamma, Rm_update_method)
                else:
                    guided_tr_update = tr_update_np
                    
                guided_tr_actions.append(torch.tensor(guided_tr_update, dtype=torch.float32))

            tr_perturb = torch.cat(guided_tr_actions, dim=0)
            step_tr_action.append(tr_perturb)
            # Core rule: evaluate guided action with model-native distribution tr_mu/tr_std
            step_tr_log_prob.append(torch.distributions.Normal(tr_mu, tr_std).log_prob(tr_perturb).sum(dim=-1))

            # --- B. Rotation ---
            rot_mu = (rot_g ** 2 * dt_rot_t * rot_score_c)
            rot_std = (rot_g * torch.sqrt(dt_rot_t)).clamp(min=1e-6)
            rot_perturb_raw = rot_mu + rot_std * torch.randn_like(rot_mu)
            
            guided_rot_actions = []
            for i, graph in enumerate(cpu_batch):
                rot_state = get_rot_state(graph["ligand"].pos).numpy()
                rot_update_np = rot_perturb_raw[i:i+1].numpy()
                
                if rot_guiding and not in_rot_region(state=rot_state, region=desired_rot):
                    rot_vdir, rot_distance = rot_guider(rot_state, desired_rot)
                    rot_mat = get_guided_rotation_matrix(rot_state, rot_vdir, rot_distance, rot_update_np, rot_gamma)
                    # Convert GDD 3x3 rotation matrix back to DiffDock axis-angle vector
                    guided_rot_update = R.from_matrix(rot_mat).as_rotvec() 
                else:
                    guided_rot_update = rot_update_np.squeeze()
                
                guided_rot_actions.append(torch.tensor(guided_rot_update, dtype=torch.float32))

            rot_perturb = torch.stack(guided_rot_actions, dim=0)
            step_rot_action.append(rot_perturb)
            step_rot_log_prob.append(torch.distributions.Normal(rot_mu, rot_std).log_prob(rot_perturb).sum(dim=-1))

            # --- C. Torsion ---
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                tor_mu = (tor_g ** 2 * dt_tor_t * tor_score_c)
                tor_std = (tor_g * torch.sqrt(dt_tor_t)).clamp(min=1e-6)
                tor_perturb_raw = tor_mu + tor_std * torch.randn_like(tor_mu)
                
                guided_tor_actions = []
                cur_idx = 0
                for i, graph in enumerate(cpu_batch):
                    n_tor = graph['ligand'].edge_mask.sum().item()
                    step_tor_indices.append(n_tor)
                    if n_tor == 0:
                        continue
                        
                    tor_update_np = tor_perturb_raw[cur_idx : cur_idx + n_tor].numpy()
                    
                    if tor_guiding:
                        tor_state = get_tor_state(
                            positions=graph["ligand"].pos,
                            edge_index=graph["ligand", "ligand"].edge_index,
                            edge_mask=graph["ligand"].edge_mask,
                            mol=graph.mol[0],
                            method=angle_calc_method
                        )
                        if not in_torus_region(tau=tor_state, region=desired_tor):
                            tor_vdir, tor_distance = tor_guider(state=tor_state, region=desired_tor)
                            if neg_vdir:
                                tor_vdir = -tor_vdir
                                
                            if mask_n_tor and mask_n_tor > 0:
                                n_angles = len(tor_state)
                                _mask_n_tor = min(mask_n_tor, n_angles)
                                if mask_n_distribution == "random":
                                    unique_indices = random.sample(range(n_angles), _mask_n_tor)
                                elif mask_n_distribution == "weighted":
                                    p = desired_tor[-1][0]
                                    unique_indices = np.random.choice(range(n_angles), size=_mask_n_tor, replace=False, p=(p / np.sum(p)))
                                for index in unique_indices:
                                    tor_vdir[index] = 0
                                    
                            current_gamma = tor_gamma
                            if dynamic_gamma:
                                current_gamma, _ = compute_tor_gamma(tor_update_np, tor_vdir, tor_distance)
                                
                            guided_tor_update = get_guided_tor_update(tor_state, tor_vdir, tor_distance, tor_update_np, current_gamma, Rm_update_method)
                        else:
                            guided_tor_update = tor_update_np
                    else:
                        guided_tor_update = tor_update_np
                        
                    guided_tor_actions.append(torch.tensor(guided_tor_update, dtype=torch.float32))
                    cur_idx += n_tor
                
                if len(guided_tor_actions) > 0:
                    tor_perturb = torch.cat(guided_tor_actions, dim=0)
                else:
                    tor_perturb = torch.empty((0,))
                    
                step_tor_action.append(tor_perturb)
                
                # Recompute torsion log probability
                raw_tor_lp = torch.distributions.Normal(tor_mu, tor_std).log_prob(tor_perturb)
                start_idx = 0
                batch_tor_lps = []
                # Get torsion indices for current batch
                for n_tor in step_tor_indices[-len(cpu_batch):]:
                    batch_tor_lps.append(raw_tor_lp[start_idx : start_idx + n_tor].sum() if n_tor > 0 else torch.tensor(0.0))
                    start_idx += n_tor
                step_tor_log_prob.append(torch.stack(batch_tor_lps))

            # --- D. Apply ---
            cur_tor_idx = 0
            for i, graph in enumerate(cpu_batch):
                current_tr_p = step_tr_action[-1][i:i+1]
                current_rot_p = step_rot_action[-1][i:i+1].squeeze(0)
                p_tor = None
                if not model_args.no_torsion:
                    n_tor = step_tor_indices[-len(cpu_batch) + i]
                    if n_tor > 0:
                        p_tor = step_tor_action[-1][cur_tor_idx : cur_tor_idx + n_tor]
                        cur_tor_idx += n_tor
                
                # Use original modify_conformer since guidance is already fused into actions
                new_graph = modify_conformer(graph, current_tr_p, current_rot_p, p_tor.numpy() if p_tor is not None else None)
                new_data_list.append(new_graph)

        # Save trajectory step
        step_info = {
            't_idx': t_idx,
            't_tr': t_tr, 't_rot': t_rot, 't_tor': t_tor,
            'dt_tr': dt_tr, 'dt_rot': dt_rot, 'dt_tor': dt_tor,
            'obs_pos': torch.cat(step_obs_pos, dim=0).cpu(),
            'action_tr': torch.cat(step_tr_action, dim=0).cpu(),
            'log_prob_tr': torch.cat(step_tr_log_prob, dim=0).cpu(),
            'action_rot': torch.cat(step_rot_action, dim=0).cpu(),
            'log_prob_rot': torch.cat(step_rot_log_prob, dim=0).cpu(),
            'log_prob_tor': (torch.cat(step_tor_log_prob, dim=0) if not model_args.no_torsion else torch.zeros(N)).cpu(),
            'tor_indices': step_tor_indices if not model_args.no_torsion else [],
        }
        if not model_args.no_torsion: 
            step_info['action_tor'] = torch.cat(step_tor_action, dim=0).cpu()
        
        trajectory_data.append(step_info)
        data_list = new_data_list

    return data_list, trajectory_data