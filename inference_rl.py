import copy
import os
import torch
from argparse import ArgumentParser, Namespace
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones

from utils.sampling_rl import sampling_rl as sampling_fn
from utils.sampling import randomize_position
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm
import yaml

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input')
parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='SMILES or path to molecule')

parser.add_argument('--out_dir', type=str, default='results/rl_inference', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save pdb file with steps')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model')
parser.add_argument('--ckpt', type=str, default='rl_model_epoch_70.pt', help='Checkpoint to use for the score model')
parser.add_argument('--use_confidence', action='store_true', default=False, help='Enable confidence-based reranking')
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps actually performed')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Load Score Model arguments
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))

if args.use_confidence:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
else:
    confidence_args = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Process input list
if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = set_nones(df['complex_name'].tolist())
    protein_path_list = set_nones(df['protein_path'].tolist())
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())
    ligand_description_list = set_nones(df['ligand_description'].tolist())
else:
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]

complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    os.makedirs(write_dir, exist_ok=True)

# Preprocess data
test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embeddings=score_model_args.esm_embeddings_path is not None,
                                receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                atom_max_neighbors=score_model_args.atom_max_neighbors)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

if args.use_confidence and not confidence_args.use_original_model_cache:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list,
                                               protein_files=protein_path_list,
                                               ligand_descriptions=ligand_description_list,
                                               protein_sequences=protein_sequence_list,
                                               lm_embeddings=confidence_args.esm_embeddings_path is not None,
                                               receptor_radius=confidence_args.receptor_radius,
                                               remove_hs=confidence_args.remove_hs,
                                               c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                               all_atoms=confidence_args.all_atoms,
                                               atom_radius=confidence_args.atom_radius,
                                               atom_max_neighbors=confidence_args.atom_max_neighbors,
                                               precomputed_lm_embeddings=test_dataset.lm_embeddings)
else:
    confidence_test_dataset = None

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

# Load Score Model
model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

if args.use_confidence:
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

failures, skipped = 0, 0
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))

for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    if not orig_complex_graph.success[0]:
        skipped += 1
        print(f"HAPPENING | Skipping {test_dataset.complex_names[idx]}")
        continue
    try:
        # Duplicate N copies for parallel pose generation
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        if confidence_test_dataset is not None:
            confidence_complex_graph = confidence_test_dataset[idx]
            if not confidence_complex_graph.success:
                skipped += 1
                print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                continue
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None
        
        # Randomly initialize positions
        randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max)
        lig = orig_complex_graph.mol[0]

        # Initialize visualization (optional)
        visualization_list = None
        if args.save_visualisation:
            visualization_list = []
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)

        data_list, trajectories, confidence = sampling_fn(
            data_list=data_list, 
            model=model,
            inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
            tr_schedule=tr_schedule, 
            rot_schedule=tr_schedule, 
            tor_schedule=tr_schedule,
            device=device, 
            t_to_sigma=t_to_sigma, 
            model_args=score_model_args,
            batch_size=args.batch_size,
            visualization_list=visualization_list,
            confidence_model=confidence_model,
            confidence_data_list=confidence_data_list,
            confidence_model_args=confidence_args,
        )
        
        # Restore coordinates to original frame
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
        if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]

        # Save results
        write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)

            if confidence is not None:
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))
            else:
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'sample_{rank}.sdf'))

        # Save visualization
        if args.save_visualisation and visualization_list is not None:
             for rank, pdb_file in enumerate(visualization_list):
                 pdb_file.write(os.path.join(write_dir, f'sample_{rank}_reverseprocess.pdb'))

    except Exception as e:
        print("Failed on", orig_complex_graph["name"], e)
        # Print full traceback for RL debugging
        import traceback
        traceback.print_exc()
        failures += 1

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')
