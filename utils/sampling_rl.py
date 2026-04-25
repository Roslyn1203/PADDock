import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update


def sampling_rl(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
                batch_size=32, visualization_list=None, confidence_model=None, confidence_data_list=None,
                confidence_model_args=None):
    N = len(data_list)
    trajectory_data = []

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

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

            # Record current state
            step_obs_pos.append(complex_graph_batch['ligand'].pos.detach().cpu())

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            # 1. Model inference (use autocast to match training setup)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    tr_score, rot_score, tor_score = model(complex_graph_batch)
            
            # 2. Clean immediately and cast back to float32 for math ops
            def soft_clean(s):
                return torch.nan_to_num(s.detach(), nan=0.0, posinf=1e3, neginf=-1e3).to(torch.float32).cpu()

            tr_score_c = soft_clean(tr_score)
            rot_score_c = soft_clean(rot_score)
            tor_score_c = soft_clean(tor_score)
            
            # 3. Prepare coefficients (use torch to avoid precision drift)
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
            
            dt_tr_t = torch.tensor(dt_tr, dtype=torch.float32)
            dt_rot_t = torch.tensor(dt_rot, dtype=torch.float32)
            dt_tor_t = torch.tensor(dt_tor, dtype=torch.float32)

            # --- A. Translation ---
            tr_mu = (tr_g ** 2 * dt_tr_t * tr_score_c)
            tr_std = (tr_g * torch.sqrt(dt_tr_t)).clamp(min=1e-6) 
            tr_perturb = tr_mu + tr_std * torch.randn_like(tr_mu)
            
            step_tr_action.append(tr_perturb)
            step_tr_log_prob.append(torch.distributions.Normal(tr_mu, tr_std).log_prob(tr_perturb).sum(dim=-1))

            # --- B. Rotation ---
            rot_mu = (rot_g ** 2 * dt_rot_t * rot_score_c)
            rot_std = (rot_g * torch.sqrt(dt_rot_t)).clamp(min=1e-6)
            rot_perturb = rot_mu + rot_std * torch.randn_like(rot_mu)
            
            step_rot_action.append(rot_perturb)
            step_rot_log_prob.append(torch.distributions.Normal(rot_mu, rot_std).log_prob(rot_perturb).sum(dim=-1))

            # --- C. Torsion ---
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                tor_mu = (tor_g ** 2 * dt_tor_t * tor_score_c)
                tor_std = (tor_g * torch.sqrt(dt_tor_t)).clamp(min=1e-6)
                
                tor_perturb_raw = tor_mu + tor_std * torch.randn_like(tor_mu)
                step_tor_action.append(tor_perturb_raw)
                
                raw_tor_lp = torch.distributions.Normal(tor_mu, tor_std).log_prob(tor_perturb_raw)
                
                start_idx = 0
                batch_tor_lps = []
                for graph in complex_graph_batch.to_data_list():
                    n_tor = graph['ligand'].edge_mask.sum().item()
                    step_tor_indices.append(n_tor)
                    batch_tor_lps.append(raw_tor_lp[start_idx : start_idx + n_tor].sum() if n_tor > 0 else torch.tensor(0.0))
                    start_idx += n_tor
                step_tor_log_prob.append(torch.stack(batch_tor_lps))
            
            # --- D. Apply ---
            cpu_batch = complex_graph_batch.to('cpu').to_data_list()
            cur_tor_idx = 0
            for i, graph in enumerate(cpu_batch):
                current_tr_p = step_tr_action[-1][i:i+1]
                current_rot_p = step_rot_action[-1][i:i+1].squeeze(0)
                p_tor = None
                if not model_args.no_torsion:
                    n_tor = step_tor_indices[-len(cpu_batch) + i]
                    p_tor = step_tor_action[-1][cur_tor_idx : cur_tor_idx + n_tor]
                    cur_tor_idx += n_tor
                new_graph = modify_conformer(graph, current_tr_p, current_rot_p, p_tor.numpy() if p_tor is not None else None)
                new_data_list.append(new_graph)

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((new_data_list[idx]['ligand'].pos + new_data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

        # Save trajectory step and ensure tensors are offloaded from GPU
        step_info = {
            't_idx': t_idx,
            't_tr': t_tr, 't_rot': t_rot, 't_tor': t_tor,
            'dt_tr': dt_tr, 'dt_rot': dt_rot, 'dt_tor': dt_tor,
            # Must call .cpu()
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

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size, shuffle=False)) \
                if confidence_data_list is not None else None
            confidence = []
            for complex_graph_batch in loader:
                b = complex_graph_batch.num_graphs
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, trajectory_data, confidence
