import os
import time
import yaml
import torch
import numpy as np
import copy
import csv
import glob
from functools import partial
from argparse import Namespace
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

# Import project modules
from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.utils import get_model
from utils.sampling_rl import sampling_rl, randomize_position
from utils.vina_scoring import calc_vina_rewards


class BatchSizeManager:
    def __init__(self, rank, cache_dir='./batch-size'):
        # 1. Ensure directory exists
        os.makedirs(cache_dir, exist_ok=True)

        self.filename = os.path.join(
            cache_dir, f'batch_size_cache_rank{rank}.csv'
        )
        self.cache = {}

        # 2. Ensure file exists (create empty one if missing)
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()

        self.load()
        
    def load(self):
        if os.path.exists(self.filename):
            print(f"[Rank {os.environ.get('LOCAL_RANK', 0)}] "
                  f"Loading Cache from {self.filename}...")
            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        self.cache[row[0]] = int(row[1])
    
    def get(self, name):
        return self.cache.get(name)
    
    def update(self, name, bs):
        # Write only when value is new/changed to avoid redundant I/O
        if name not in self.cache or self.cache[name] != bs:
            self.cache[name] = bs
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, bs])

# ----------------------------------------------------------------------
# Main screening logic
# ----------------------------------------------------------------------
def screen_dataset(args, model, loader, t_to_sigma, device, tr_schedule, rot_schedule, tor_schedule):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    # Show progress bar only on rank 0
    iterator = tqdm(loader, desc="Screening") if rank == 0 else loader
    bs_manager = BatchSizeManager(rank)

    processed_count = 0
    bad_count_total = 0

    # Set model to eval mode
    model.eval()

    with torch.no_grad(): # gradients are not needed during screening
        for idx, batch_data in enumerate(iterator):
            
            # 1. Parse input data
            if isinstance(batch_data, list): orig_complex_graph = batch_data[0]
            else: orig_complex_graph = batch_data.clone()
            orig_complex_graph = orig_complex_graph.to('cpu')

            # Normalize attributes
            if hasattr(orig_complex_graph, 'mol') and isinstance(orig_complex_graph.mol, list):
                orig_complex_graph.mol = orig_complex_graph.mol[0]
            if hasattr(orig_complex_graph, 'name') and isinstance(orig_complex_graph['name'], list):
                orig_complex_graph.name = orig_complex_graph['name'][0]
            if hasattr(orig_complex_graph, 'original_center') and orig_complex_graph.original_center.dim() > 1:
                orig_complex_graph.original_center = orig_complex_graph.original_center[0]

            complex_name = orig_complex_graph.name

            # 2. Check cache: skip if already processed
            cached_val = bs_manager.get(complex_name)
            if cached_val is not None:
                print(f"[Rank {rank}] Skipping {complex_name} (Already checked: {cached_val})")
                continue

            # 3. Basic filtering (boron/oversized systems)
            should_skip = False
            if hasattr(orig_complex_graph, 'mol'):
                atoms = [atom.GetSymbol() for atom in orig_complex_graph.mol.GetAtoms()]
                if 'B' in atoms: # boron atoms are unsupported
                    bs_manager.update(complex_name, -1) # -1 marks special skip
                    should_skip = True
            
            if not should_skip:
                num_atoms = orig_complex_graph['receptor'].pos.shape[0] + orig_complex_graph['ligand'].pos.shape[0]
                if num_atoms > 1500: # too large
                    bs_manager.update(complex_name, 999)
                    should_skip = True
            
            if should_skip:
                continue

            # 4. Sampling (generate args.samples_per_complex samples)
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.samples_per_complex)]
            randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
            
            try:
                # Run sampling
                final_data_list, _ = sampling_rl(
                    data_list=data_list, model=model, inference_steps=args.inference_steps,
                    tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                    device=device, t_to_sigma=t_to_sigma, model_args=args, 
                    batch_size=args.samples_per_complex 
                )
                # === New: robust cleanup ===
                # If samples diverge, replace them with valid ones
                # final_data_list, _ = robust_clean_trajectories(
                #     final_data_list, _, orig_complex_graph, device
                # )
                # ==========================
            except Exception as e:
                print(f"[Rank {rank}] Error sampling {complex_name}: {e}")
                bs_manager.update(complex_name, -1) # errors are marked as bad
                continue

            # 5. Scoring (compute RMSD)
            # We only need rmsd_tensor; vina_rewards is returned together but unused
            _, rmsd_tensor = calc_vina_rewards(final_data_list, orig_complex_graph, args)
            
            # Convert to numpy for processing
            rmsd_vals = rmsd_tensor.cpu().numpy()

            # 6. Core screening logic
            # Count samples with RMSD >= 15
            num_bad_samples = np.sum(rmsd_vals >= 15.0)
            bs_manager.update(complex_name, num_bad_samples)
            
               
            processed_count += 1
            if processed_count % 10 == 0 and rank == 0:
                print(f"Processed: {processed_count}, Found Bad: {bad_count_total}")

    print(f"[Rank {rank}] Screening finished. Total Bad found locally: {bad_count_total}")

# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def main():
    # --- A. Distributed initialization ---
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=180))
        device = torch.device(f'cuda:{local_rank}')
        is_distributed = True
        if local_rank == 0:
            print(f"Distributed screening enabled. Local Rank: {local_rank}")
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False
        print("Single GPU mode.")

    # --- B. Load parameters ---
    model_dir = 'workdir/paper_score_model' # base model path
    if not os.path.exists(f'{model_dir}/model_parameters.yml'):
        raise FileNotFoundError(f"Model config not found at {model_dir}/model_parameters.yml")
        
    with open(f'{model_dir}/model_parameters.yml') as f:
        args = Namespace(**yaml.full_load(f))
    
    # Forced parameter settings
    args.batch_size = 1 # inference batch size must be 1
    args.inference_steps = 10 # keep aligned with training setup
    args.samples_per_complex = 8 # generate 8 samples for screening
    
    # Path settings (keep consistent with training code)
    args.split_train = 'data/splits/timesplit_no_lig_overlap_val'
    args.cache_path = 'data/cache_rl_debug' 
    args.num_workers = 4

    # --- C. Data loading ---
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    # Build data loader
    # Note: construct_loader usually attaches DistributedSampler automatically
    loader, _ = construct_loader(args, t_to_sigma, device)

    # --- D. Model loading (base DiffDock weights) ---
    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    
    # Load base weights
    ckpt_path = f'{model_dir}/best_ema_inference_epoch_model.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base model weights not found at {ckpt_path}")
        
    print(f"[Rank {local_rank}] Loading base model from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # Note: no DDP(model) wrapping is needed during screening (no gradient sync required)
    # Only DistributedSampler is needed for data partitioning.

    # --- E. Prepare schedule ---
    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule

    # --- F. Start screening ---
    if local_rank == 0:
        print(">>> Start Screening (Filtering based on RMSD >= 15) <<<")
        
    screen_dataset(
        args, model, loader, t_to_sigma, device, 
        tr_schedule, rot_schedule, tor_schedule
    )
    
    if is_distributed:
        dist.barrier() # wait for all ranks to finish
        dist.destroy_process_group()

if __name__ == '__main__':
    main()