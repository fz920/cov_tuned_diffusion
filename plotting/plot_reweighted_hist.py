import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from training_utils import load_target_dist, load_dataset
from tqdm import tqdm
from utils.path_config import CHECKPOINTS_DIR, FIGURES_DIR

def main():
    parser = argparse.ArgumentParser()
    # Sampling settings
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                      help='List of sample indices to combine (uses --sample_index if not provided)')
    parser.add_argument('--cov_form', type=str, default='ddpm',
                      choices=['ddpm', 'isotropic', 'full'],
                      help='Covariance form to use for plotting')
    parser.add_argument('--save_energy', action='store_true',
                      help='Save computed energies to a pickle file for future use')
    parser.add_argument('--use_saved_energy', action='store_true',
                      help='Use previously saved energy values instead of recomputing them')

    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_steps', type=int, default=100)
    args = parser.parse_args()

    # If sample_indices is not provided, use sample_index
    if args.sample_indices is None:
        args.sample_indices = [args.sample_index]

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique suffix for filenames based on sample indices
    if len(args.sample_indices) > 1:
        indices_str = f"indices_{min(args.sample_indices)}-{max(args.sample_indices)}"
    else:
        indices_str = f"index_{args.sample_indices[0]}"

    # Path for saved energy file
    energy_dir = CHECKPOINTS_DIR / 'energy' / args.dataset
    energy_file = energy_dir / f'{args.cov_form}_{args.num_steps}steps_{indices_str}.pkl'

    # Check if we should use saved energy values
    if args.use_saved_energy and energy_file.exists():
        print(f"Loading precomputed energy values from {energy_file}")
        with open(energy_file, 'rb') as f:
            energy_data = pickle.load(f)
        
        # Extract energy values and weights
        model_energy = energy_data['model_energy'].to(device)
        true_energy = energy_data['true_energy'].to(device)
        w = energy_data['weights'].to(device)
        log_w = energy_data['log_weights'].to(device)
        
        # Check if the loaded data is compatible with current parameters
        if energy_data['num_samples'] < args.num_samples:
            print(f"Warning: Saved energy file contains fewer samples ({energy_data['num_samples']}) than requested ({args.num_samples})")
            args.num_samples = energy_data['num_samples']
            
        # Ensure we're using the correct number of samples
        model_energy = model_energy[:args.num_samples]
        true_energy = true_energy[:args.num_samples]
        w = w[:args.num_samples]
        log_w = log_w[:args.num_samples]
        
        # Normalize the weights just in case
        w = w / w.sum()
        
        print(f"Reverse ESS (%) for {args.cov_form}: {1 / torch.sum(w ** 2) / args.num_samples * 100:.2f}%")
        
    else:
        # If not using saved energy or file doesn't exist, compute energies
        if args.use_saved_energy and not energy_file.exists():
            print(f"Warning: Requested to use saved energy file {energy_file} but it doesn't exist. Computing energies instead.")
        
        # Define sample paths
        samples_base_path = CHECKPOINTS_DIR / 'samples' / args.dataset
        
        # Load and combine samples from multiple files
        all_samples = []
        all_log_weights = []

        for sample_idx in args.sample_indices:
            backward_pkl_path = samples_base_path / 'backward' / f'{args.num_steps}steps_{sample_idx}sample.pkl'
            
            # Load samples and weights
            with open(backward_pkl_path, 'rb') as f:
                backward_results = pickle.load(f)
            
            # Extract samples and weights for the selected covariance form
            # Concatenate all batches of samples and weights from this file
            file_samples = torch.cat(backward_results[args.cov_form]['samples'], dim=0)
            file_log_weights = torch.cat(backward_results[args.cov_form]['log_weights'], dim=0)
            
            all_samples.append(file_samples)
            all_log_weights.append(file_log_weights)
        
        # Concatenate all samples and weights from all files
        all_samples = torch.cat(all_samples, dim=0)
        all_log_weights = torch.cat(all_log_weights, dim=0)
        
        # get true target density and true samples
        true_target_dist = load_target_dist(args.dataset)
        true_samples = load_dataset(args.dataset, partition='train')
        true_samples = true_samples[:args.num_samples].to(device)

        # Limit to requested number of samples if needed
        x = all_samples[:args.num_samples].to(device)
        log_w = all_log_weights[:args.num_samples].to(device)

        # normalize the importance weights

        w = torch.exp(log_w - log_w.max())
        w = w / w.sum()

        print(f"Reverse ESS (%) for {args.cov_form}: {1 / torch.sum(w ** 2) / args.num_samples * 100:.2f}%")

        # get energy
        if args.dataset == 'aldp':
            # Evaluate energy in batches to avoid memory issues
            batch_size = 10000
            num_batches = (args.num_samples + batch_size - 1) // batch_size
            
            model_energy = []
            true_energy = []
            
            # Process model samples in batches
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, args.num_samples)
                batch_x = x[start_idx:end_idx]
                
                # Get energy for this batch
                batch_energy = true_target_dist.energy(batch_x)
                model_energy.append(batch_energy)
        
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, args.num_samples)
                batch_true = true_samples[start_idx:end_idx]
        
                # Get energy for this batch
                batch_energy = true_target_dist.energy(batch_true)
                true_energy.append(batch_energy)
            
            # Concatenate results
            model_energy = torch.cat(model_energy)
            true_energy = torch.cat(true_energy)
            
            # Save energy values to file if requested
            if args.save_energy:
                # Create energy directory if it doesn't exist
                energy_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model and true energy values along with weights
                energy_data = {
                    'model_energy': model_energy.detach().cpu(),
                    'true_energy': true_energy.detach().cpu(),
                    'weights': w.detach().cpu(),
                    'log_weights': log_w.detach().cpu(),
                    'covariance_form': args.cov_form,
                    'num_steps': args.num_steps,
                    'sample_indices': args.sample_indices,
                    'num_samples': args.num_samples
                }
                
                with open(energy_file, 'wb') as f:
                    pickle.dump(energy_data, f)
                
                print(f"Energy values saved to {energy_file}")
        else:
            model_energy = true_target_dist.energy(x)
            true_energy = true_target_dist.energy(true_samples)

    print(f"Total proposed samples: {model_energy.shape[0]}")

    model_energy_np = model_energy.detach().cpu().numpy()
    true_energy_np = true_energy.detach().cpu().numpy()

    w_np = w.detach().cpu().numpy()

    # labels and titles for the plot
    dataset_title = {
        'dw4': 'DW-4',
        'lj13': 'LJ-13',
        'lj55': 'LJ-55',
        'aldp': 'Alanine Dipeptide'
    }

    bins_dataset = {
        'dw4': 90,
        'lj13': 100,
        'lj55': 100,
        'aldp': 200
    }

    # plot the histogram
    plt.figure(figsize=(7, 4))

    # 1) True log probability (Ground Truth)
    plt.hist(true_energy_np, bins=bins_dataset[args.dataset], density=True, 
            alpha=0.5,
            label='Ground Truth')

    # 2) Model log probability (unweighted)
    plt.hist(model_energy_np, bins=bins_dataset[args.dataset], density=True, 
            alpha=0.5,
            label=f'Diffusion Samples (Unweighted)')

    # 3) Reweighted model log probability
    plt.hist(model_energy_np, bins=bins_dataset[args.dataset], density=True, 
            weights=w_np, histtype='step', linewidth=2, alpha=1, color='red',
            label=f'Diffusion Samples (Reweighted)')

    if args.dataset == 'dw4':
        plt.xlim([-27.5, -14])
    elif args.dataset == 'lj13':
        plt.xlim([-65, -15])
    elif args.dataset == 'lj55':
        plt.xlim([-100, -10])
    elif args.dataset == 'aldp':
        plt.xlim([-60, -10])

    # Increase font sizes
    plt.xlabel('Energy', fontsize=18)
    plt.ylabel('Normalized Density', fontsize=18)
    plt.legend(loc='upper right', fontsize=10.5)
    plt.title(f'{dataset_title[args.dataset]} - Reweighted Histogram', fontsize=18)

    # # Increase tick label sizes
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    
    plt.tight_layout()

    # Create figures directory if it doesn't exist
    figures_dir = FIGURES_DIR / 'hist'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save figure with the sample indices in the filename
    pdf_path = figures_dir / f'{args.dataset}_{args.cov_form}_reweight_{args.num_steps}steps_{indices_str}.pdf'
    png_path = figures_dir / f'{args.dataset}_{args.cov_form}_reweight_{args.num_steps}steps_{indices_str}.png'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)

    # Also save a high-resolution PNG version
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    
    plt.close()


if __name__ == '__main__':
    main()
