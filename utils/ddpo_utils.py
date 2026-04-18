import os
import shutil
import copy
import torch
from rdkit import Chem
from utils.diffusion_utils import set_time
import numpy as np
from utils.diffusion_utils import t_to_sigma

def save_debug_visualization(orig_graph, generated_graphs, args, save_dir="./debug_vis"):
    """
    Save GT ligand, sampled ligands, and receptor to SDF/PDB files for debugging.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Get basic metadata
        name = orig_graph.name
        if isinstance(name, list): name = name[0]
        
        # === A. Save receptor ===
        # Reuse receptor path resolution logic from vina_scoring
        pdb_path = None
        if hasattr(orig_graph, 'protein_path'):
             p_path = orig_graph.protein_path
             pdb_path = p_path[0] if isinstance(p_path, list) else p_path
        else:
             # Try inferring from data_dir
             possible_path = os.path.join(args.data_dir, name, f'{name}_protein_processed_fix.pdb')
             if os.path.exists(possible_path):
                 pdb_path = possible_path
             else:
                 possible_path = os.path.join(args.data_dir, name, f'{name}_processed_protein.pdb')
                 if os.path.exists(possible_path):
                     pdb_path = possible_path
        
        if pdb_path and os.path.exists(pdb_path):
            target_pdb = os.path.join(save_dir, f"{name}_protein.pdb")
            if not os.path.exists(target_pdb):
                shutil.copy(pdb_path, target_pdb)
                # print(f"[Debug] Copied Receptor to {target_pdb}")
        else:
            print(f"[Debug] Warning: Receptor PDB not found for {name}")

        # === B. Save GT ligand ===
        mol = orig_graph.mol
        if isinstance(mol, list): mol = mol[0]
        gt_mol = copy.deepcopy(mol) 
        
        # Get center
        center = torch.zeros(3)
        if hasattr(orig_graph, 'original_center'):
            center = orig_graph.original_center
            if center.dim() > 1: center = center[0]
        center = center.cpu()

        gt_pos_tensor = orig_graph['ligand'].pos.cpu() + center
        
        # Basic atom-count check and optional dehydrogenation logic
        if gt_mol.GetNumAtoms() != gt_pos_tensor.shape[0]:
            gt_mol = Chem.RemoveHs(gt_mol)
            
        if gt_mol.GetNumAtoms() == gt_pos_tensor.shape[0]:
            conf = gt_mol.GetConformer()
            for i in range(gt_mol.GetNumAtoms()):
                x, y, z = gt_pos_tensor[i].tolist()
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            
            gt_path = os.path.join(save_dir, f"{name}_gt_ligand.sdf")
            if not os.path.exists(gt_path):
                with Chem.SDWriter(gt_path) as w:
                    w.write(gt_mol)
                # print(f"[Debug] Saved GT Ligand to {gt_path}")
        
        # === C. Save sampled ligands ===
        for i, graph in enumerate(generated_graphs):
            pred_pos = graph['ligand'].pos.detach().cpu() + center
            pred_mol = copy.deepcopy(gt_mol) # reuse processed GT molecule (possibly dehydrogenated)
            
            if pred_mol.GetNumAtoms() == pred_pos.shape[0]:
                conf = pred_mol.GetConformer()
                for j in range(pred_mol.GetNumAtoms()):
                    x, y, z = pred_pos[j].tolist()
                    conf.SetAtomPosition(j, (float(x), float(y), float(z)))
                
                sample_path = os.path.join(save_dir, f"{name}_sample_{i}.sdf")
                if not os.path.exists(sample_path):
                    with Chem.SDWriter(sample_path) as w:
                        w.write(pred_mol)
            
        # print(f"[Debug] Visualization saved to {save_dir}")
        
    except Exception as e:
        print(f"[Debug] Save visualization failed: {e}")

def get_step_log_prob(model, replay_batch, current_pos_tensor, action_tr_tensor, action_rot_tensor, action_tor_tensor, 
                      tor_indices, t_tr, t_rot, t_tor, dt_tr, dt_rot, dt_tor, t_to_sigma, model_args, device):
    model.eval()
    
    # 1. Prepare inputs
    replay_batch['ligand'].pos = current_pos_tensor.to(device)
    set_time(replay_batch, t_tr, t_rot, t_tor, replay_batch.num_graphs, model_args.all_atoms, device)
    
    # 2. Memory-optimized forward pass (single call site)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        tr_score_bf, rot_score_bf, tor_score_bf = model(replay_batch)
    
    # Memory tip: cast to float32 immediately for math ops to avoid bf16 graph inefficiency
    tr_score = tr_score_bf.float()
    rot_score = rot_score_bf.float()
    tor_score = tor_score_bf.float()
    
    # Numerical validation and cleanup
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

class BatchSizeManager:
    def __init__(self, rank, cache_dir='./batch-sizes'):
        # 1. Ensure directory exists
        os.makedirs(cache_dir, exist_ok=True)

        self.filename = os.path.join(
            cache_dir, f'batch_size_cache_rank{rank}.csv'
        )
        self.cache = {}

        # 2. Ensure file exists (create empty file if missing)
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()

        self.load()
        
    def load(self):
        if os.path.exists(self.filename):
            print(f"[Rank {os.environ.get('LOCAL_RANK', 0)}] "
                  f"Loading BatchSize Cache from {self.filename}...")
            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        self.cache[row[0]] = int(row[1])
    
    def get(self, name):
        return self.cache.get(name)
    
    def update(self, name, bs):
        self.cache[name] = bs
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, bs])


def manual_all_reduce_grads(model):
    """
    Unified synchronization utility.
    Manually sync all parameter gradients to prevent DDP deadlocks caused by OOM or skipped steps.
    """
    if not dist.is_initialized():
        return
    
    world_size = dist.get_world_size()
    
    for param in model.parameters():
        if param.requires_grad:
            # If no gradient exists (e.g., skipped step), initialize zeros for synchronization
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            
            # Aggregate gradients across all ranks
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Take mean over world size
            param.grad /= world_size
    
def set_train_mode_with_frozen_bn(model):
    """
    Enable training mode while freezing all BatchNorm layers.
    This is standard practice for fine-tuning and RL.
    """
    model.train() # enable global train mode (dropout, gradients, etc.)
    
    # Iterate over all submodules and force BN layers to eval mode
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


# --- Add near file header ---
def ddp_sync_skip(model):
    """
    Execute a dummy backward pass when a batch is skipped to keep DDP synchronization healthy.
    """
    # Only needed in DDP mode
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # Sum any trainable params, multiply by 0, then backward
        # This produces zero gradients but still satisfies DDP communication requirements
        dummy = sum([p.sum() for p in model.parameters() if p.requires_grad])
        (dummy * 0.0).backward()