import os
import time
import copy  # Required for safe molecule duplication
import numpy as np
import torch
from multiprocessing import Pool
import subprocess
import signal
from contextlib import contextmanager

from rdkit import Chem
from meeko import MoleculePreparation
from openbabel import pybel
from vina import Vina

# --- Import SpyRMSD ---
from spyrmsd import rmsd as spy_rmsd
from spyrmsd import molecule as spy_mol
HAS_SPYRMSD = True
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="meeko.preparation")
# --- Helper utilities ---
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def prepare_receptor(pdb_path):
    """Prepare receptor PDBQT file."""
    pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')
    if os.path.exists(pdbqt_path):
        return pdbqt_path
    
    try:
        cmd = ['obabel', pdb_path, '-O', pdbqt_path, '-xr', '--partialcharge', 'gasteiger']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return pdbqt_path
    except:
        pass
    
    if pybel:
        try:
            mol = next(pybel.readfile("pdb", pdb_path))
            mol.write("pdbqt", pdbqt_path, overwrite=True)
            return pdbqt_path
        except:
            pass
    return None

def get_symmetry_rmsd(rdkit_mol, ref_coords, pred_coords):
    """Compute symmetry-aware RMSD using spyrmsd."""
    if not HAS_SPYRMSD: return None
    try:
        with time_limit(2):
            mol = spy_mol.Molecule.from_rdkit(rdkit_mol)
            RMSD = spy_rmsd.symmrmsd(
                ref_coords, 
                pred_coords, 
                mol.atomicnums, 
                mol.atomicnums, 
                mol.adjacency_matrix, 
                mol.adjacency_matrix,
            )
            return RMSD
    except:
        return None

def run_vina_api_worker(args):
    """
    Worker: blind Vina scoring for Val/Test.
    Safely skip invalid NaN/Inf samples to avoid deadlocks and return raw Vina outputs for valid samples.
    """
    mol_idx, complex_name, lig_mol, pos_tensor, gt_pos_tensor, receptor_pdbqt_path, original_center_tensor = args
    
    # Default fallback values for invalid samples
    default_score = 15.0  
    default_rmsd = 50.0   
    
    vina_mol = None 
    
    try:
        # === 1. Coordinate preparation ===
        offset = (original_center_tensor.numpy() if isinstance(original_center_tensor, torch.Tensor) 
                  else original_center_tensor).astype(np.double)
        pred_rel_pos = (pos_tensor.numpy() if isinstance(pos_tensor, torch.Tensor) 
                        else pos_tensor).astype(np.double)
        
        pred_abs_pos = pred_rel_pos + offset

        # ====================================================================
        # Guardrail: skip NaN/Inf coordinates immediately
        # If diffusion outputs invalid coordinates, skip both Vina and RMSD safely.
        # This is valid because no ground-truth leakage is used here.
        # ====================================================================
        if np.isnan(pred_abs_pos).any() or np.isinf(pred_abs_pos).any():
            return default_score, default_rmsd
            
        current_center = pred_abs_pos.mean(axis=0)

        gt_rel_pos = (gt_pos_tensor.numpy() if isinstance(gt_pos_tensor, torch.Tensor) 
                      else gt_pos_tensor).astype(np.double)
        gt_abs_pos = gt_rel_pos + offset 

        try:
            vina_mol = copy.deepcopy(lig_mol)
        except:
            vina_mol = lig_mol

        # === 2. Compute true RMSD (record-only metric) ===
        rmsd_val = default_rmsd
        try:
            num_model = pred_abs_pos.shape[0]
            gt_mol_heavy = Chem.RemoveHs(lig_mol)
            check_mol = gt_mol_heavy if gt_mol_heavy.GetNumAtoms() == num_model else vina_mol
            
            if check_mol.GetNumAtoms() == num_model:
                rmsd_val = get_symmetry_rmsd(check_mol, gt_abs_pos, pred_abs_pos)
                if rmsd_val is None:
                    rmsd_val = np.sqrt(((pred_abs_pos - gt_abs_pos) ** 2).sum(axis=1).mean())
        except:
            rmsd_val = default_rmsd

        # === 3. Blind Vina scoring: score all valid samples regardless of RMSD ===
        # The previous rmsd_val <= 10.0 gating has been fully removed.
        score = default_score
        
        try:
            conf = vina_mol.GetConformer()
            for i in range(pred_abs_pos.shape[0]):
                conf.SetAtomPosition(i, (float(pred_abs_pos[i][0]), float(pred_abs_pos[i][1]), float(pred_abs_pos[i][2])))

            preparator = MoleculePreparation()
            preparator.prepare(vina_mol)
            pdbqt_string = preparator.write_pdbqt_string()

            v = Vina(sf_name='vina', cpu=1, verbosity=0)
            v.set_receptor(receptor_pdbqt_path)
            v.set_ligand_from_string(pdbqt_string)
            
            center_list = [float(current_center[0]), float(current_center[1]), float(current_center[2])]
            v.compute_vina_maps(center=center_list, box_size=[22.0, 22.0, 22.0], spacing=0.5)
            
            score = v.score()[0]
        except Exception as e:
            score = default_score

        # === 4. Disable reward smoothing ===
        # For Val/Test, output raw Vina scores directly to CSV.
        return score, rmsd_val

    except Exception as e:
        # Fallback if any unexpected error occurs
        return default_score, default_rmsd
        
def calc_vina_rewards(data_list, orig_complex_graph, args):
    """
    Compute Vina rewards and RMSD.
    """
    complex_name = orig_complex_graph['name']
    if isinstance(complex_name, list): complex_name = complex_name[0]
    
    device = data_list[0]['ligand'].pos.device
    
    # === 1. Robust PDB path resolution ===
    pdb_path = None
    if hasattr(orig_complex_graph, 'protein_path'):
         p_path = orig_complex_graph.protein_path
         temp_path = p_path[0] if isinstance(p_path, list) else p_path
         # Verify that the path stored in graph actually exists
         if os.path.exists(temp_path):
             pdb_path = temp_path
             
    if pdb_path is None:
         # Fallback: construct path manually
         pdb_path = os.path.join(args.data_dir, complex_name, f'{complex_name}_protein_processed_fix.pdb')
         if not os.path.exists(pdb_path):
             pdb_path = os.path.join(args.data_dir, complex_name, f'{complex_name}_processed_protein.pdb')

    # Guardrail 1: if PDB is missing, return failure values directly
    if not os.path.exists(pdb_path):
        print(f"\n[CRITICAL] PDB not found for {complex_name} at {pdb_path}")
        return torch.zeros(len(data_list)).to(device), torch.full((len(data_list),), 20.0).to(device)

    # === 2. Generate and strictly validate PDBQT ===
    receptor_pdbqt = prepare_receptor(pdb_path)

    # Guardrail 2: if PDBQT generation fails or file is missing, return failure values
    if not receptor_pdbqt or not os.path.exists(receptor_pdbqt):
        print(f"\n[CRITICAL] PDBQT generation failed! Expected file missing: {receptor_pdbqt}")
        return torch.zeros(len(data_list)).to(device), torch.full((len(data_list),), 20.0).to(device)

    # Prepare center tensor (move to CPU)
    if hasattr(orig_complex_graph, 'original_center'):
        original_center = orig_complex_graph.original_center
        if original_center.dim() > 1: original_center = original_center[0]
        original_center_cpu = original_center.detach().cpu()
    else:
        original_center_cpu = torch.zeros(3)

    # Prepare ground-truth positions (move to CPU)
    gt_pos_cpu = orig_complex_graph['ligand'].pos.detach().cpu() 
    base_mol = orig_complex_graph.mol 
    tasks = []
    
    # ...(the following for-loop remains unchanged)...
    for i, graph in enumerate(data_list):
        # Prepare predicted positions (move to CPU)
        pred_pos_cpu = graph['ligand'].pos.detach().cpu() # critical step
        
        tasks.append((
            i, 
            complex_name,
            base_mol, 
            pred_pos_cpu,       # CPU Tensor
            gt_pos_cpu,         # CPU Tensor
            receptor_pdbqt,
            original_center_cpu # CPU Tensor
        ))

    num_processes = min(len(data_list), 16)
    
    with Pool(num_processes) as pool:
        results = pool.map(run_vina_api_worker, tasks)
    
    vina_scores = []
    rmsds = []
    for s, r in results:
        vina_scores.append(s)
        rmsds.append(r)
    
    vina_rewards = torch.tensor([-s for s in vina_scores], dtype=torch.float32).to(device)
    rmsd_tensor = torch.tensor(rmsds, dtype=torch.float32).to(device)
    
    return vina_rewards, rmsd_tensor