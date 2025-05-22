import torch
import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import ScoreNet, compute_forward_ess, compute_reverse_ess
from training_utils import load_target_dist, load_dataset

def load_parameters(dataset, net, params_index, num_steps, alpha):
    """Load parameters that were optimized with specific alpha value."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    param_checkpoint_path = (
        f"/rds/user/fz287/hpc-work/dissertation/checkpoints/params_checkpoints/"
        f"{dataset}/{net}_score_params_{params_index}_{num_steps}steps_alpha{alpha}.pth"
    )
    
    checkpoint = torch.load(param_checkpoint_path, map_location=device)
    return checkpoint

def main():
    parser = argparse.ArgumentParser()
    # Model and dataset settings
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=100)
    
    # Experiment settings
    parser.add_argument('--alphas', type=float, nargs='+', default=[1.0, 2.0])
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[10000, 50000, 100000])
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs to average over')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output settings
    parser.add_argument('--output_dir', type=str, 
                        default=str(Path(__file__).parent.parent / 'consistency_sampling/figures'))
    parser.add_argument('--save_name', type=str, default=None,
                        help='Name for the output files (without extension)')
    args = parser.parse_args()
    
    # Set the save name if not provided
    if args.save_name is None:
        args.save_name = f'{args.dataset}_alpha_comparison_ess'
    
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the score model
    config_path = (
        fstr(Path(__file__).parent.parent / 'consistency_sampling/model/configs/')
        f'{args.dataset}_{args.net}_config.yaml'
    )
    with open(config_path, 'r') as f:
        score_model_config = yaml.safe_load(f)
        
    score_checkpoint_path = (
        CHECKPOINTS_DIR / 'model_checkpoints/'
        f'{args.dataset}/score/{args.net}_score_{args.model_index}.pth'
    )
    
    score_model = ScoreNet(dataset=args.dataset, device=device, 
                           model_config=score_model_config, 
                           net=args.net).to(device)
    
    score_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint['model_state_dict'])
    score_model.eval()
    score_model.requires_grad_(False)
    
    # Load the true target distribution
    true_target_dist = load_target_dist(args.dataset)
    
    # Load the dataset
    true_data = load_dataset(args.dataset, partition='test')
    
    # Results dictionary to store ESS values
    results = {
        'forward': {alpha: {n: [] for n in args.sample_sizes} for alpha in args.alphas},
        'reverse': {alpha: {n: [] for n in args.sample_sizes} for alpha in args.alphas}
    }
    
    # For each alpha value, load parameters and compute ESS for different sample sizes
    for alpha in args.alphas:
        print(f"Computing ESS with parameters optimized for alpha={alpha}")
        params = load_parameters(args.dataset, args.net, args.params_index, args.num_steps, alpha)
        
        for n_samples in args.sample_sizes:
            print(f"Sample size: {n_samples}")
            
            # Run multiple times to get stable estimates
            for run in range(args.num_runs):
                print(f"Run {run+1}/{args.num_runs}")
                
                # Forward ESS calculation
                with torch.no_grad():
                    # Sample initial points from true data
                    indices = torch.randperm(true_data.shape[0])[:n_samples]
                    x0 = true_data[indices].to(device)
                    log_prob_x0 = true_target_dist.log_prob(x0)
                    
                    # Forward process (data -> noise)
                    _, _, log_w_f = score_model.est_forward_ess(
                        x0, log_prob_x0, args.num_steps,
                        nus=params['nus'],
                        time_steps=params['time_steps'] if 'time_steps' in params else None,
                        progress_bar=True
                    )
                    
                    # Backward process (noise -> samples)
                    samples, _, _, log_w_b = score_model.ddpm_sampler(
                        args.num_steps,
                        true_target_dist,
                        num_samples=n_samples,
                        nus=params['nus'],
                        time_steps=params['time_steps'] if 'time_steps' in params else None,
                        progress_bar=True
                    )
                    
                    # Compute ESS
                    forward_ess = compute_forward_ess(log_w_f)
                    reverse_ess = compute_reverse_ess(log_w_b)
                    
                    # Store normalized ESS (as percentage)
                    results['forward'][alpha][n_samples].append(forward_ess.item() / n_samples * 100)
                    results['reverse'][alpha][n_samples].append(reverse_ess.item() / n_samples * 100)
                    
                    print(f"  Forward ESS: {forward_ess.item() / n_samples * 100:.2f}%")
                    print(f"  Reverse ESS: {reverse_ess.item() / n_samples * 100:.2f}%")
    
    # Compute mean and standard error for plotting
    for direction in ['forward', 'reverse']:
        for alpha in args.alphas:
            for n_samples in args.sample_sizes:
                values = results[direction][alpha][n_samples]
                results[direction][alpha][n_samples] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    # Dictionary to map dataset codes to human-readable names
    dataset_title = {
        'dw4': 'DW-4',
        'lj13': 'LJ-13',
        'lj55': 'LJ-55',
        'aldp': 'Alanine Dipeptide'
    }

    # Plot results in a similar style to plot_ess.py
    plt.figure(figsize=(7, 4))
    
    # Define colors for different alpha values
    # Darker colors for forward, lighter for reverse
    forward_colors = {
        1.0: '#1f77b4',  # blue
        2.0: '#d62728',  # red
    }

    reverse_colors = {
        1.0: '#76b7b2',  # lighter blue
        2.0: '#ff9896',  # lighter red
    }

    markers = {1.0: 'o', 2.0: 's'}
    linestyles = {'forward': '-', 'reverse': '--'}
    
    # Plot both forward and reverse ESS
    for alpha in args.alphas:
        sample_sizes = list(args.sample_sizes)

        # Forward ESS
        y_forward = [results['forward'][alpha][n]['mean'] for n in args.sample_sizes]
        std_forward = [results['forward'][alpha][n]['std'] for n in args.sample_sizes]
        
        plt.errorbar(
            sample_sizes, 
            y_forward,
            yerr=std_forward,
            fmt=markers[alpha] + linestyles['forward'],
            color=forward_colors[alpha],
            ecolor=forward_colors[alpha],
            elinewidth=1.5,
            capsize=5,
            markersize=6,
            label=fr"$\alpha={alpha}$ Forward"
        )
        
        # Reverse ESS
        y_reverse = [results['reverse'][alpha][n]['mean'] for n in args.sample_sizes]
        std_reverse = [results['reverse'][alpha][n]['std'] for n in args.sample_sizes]
        
        plt.errorbar(
            sample_sizes, 
            y_reverse,
            yerr=std_reverse,
            fmt=markers[alpha] + linestyles['reverse'],
            color=reverse_colors[alpha],
            ecolor=reverse_colors[alpha],
            elinewidth=1.5,
            capsize=5,
            markersize=6,
            label=fr"$\alpha={alpha}$ Reverse"
        )
    
    # Format the plot
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('ESS (%)', fontsize=14)
    
    dataset_name = dataset_title.get(args.dataset, args.dataset.upper())
    plt.title(f'ESS (Mean ± StdDev) for {dataset_name}', fontsize=16)
    
    plt.grid(alpha=0.3)
    plt.xticks(range(len(args.sample_sizes)), [f'{n}' for n in args.sample_sizes])
    plt.ylim(bottom=0)

    plt.xscale('log')
    
    # Place legend below the figure, similar to plot_ess.py
    plt.legend(fontsize=11, ncol=len(args.alphas), loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), framealpha=0.9)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot
    save_path_png = os.path.join(args.output_dir, f'{args.save_name}.png')
    save_path_pdf = os.path.join(args.output_dir, f'{args.save_name}.pdf')
    
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    
    print(f"Plots saved to:")
    print(f"  - {save_path_png}")
    print(f"  - {save_path_pdf}")
    
    # Save numerical results
    import json

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
    json_path = os.path.join(args.output_dir, f'{args.save_name}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)
    print(f"  - {json_path}")

if __name__ == '__main__':
    main()
