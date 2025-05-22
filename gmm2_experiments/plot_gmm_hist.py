import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from gmm import create_gmm
from score_model import ScoreNet

def load_covariance_params(args, input_dim, cov_form, num_steps):
    """
    Load covariance parameters for a specific form
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cov_form == 'full':
        filename = f"/rds/user/fz287/hpc-work/dissertation/checkpoints/gmm2_checkpoints/params_checkpoints/{input_dim}D_gmm2_score_params_{num_steps}steps_{args.sample_index}_{cov_form}_with_time_stepsFalse_rank{args.rank}.pth"
    else:
        filename = f"/rds/user/fz287/hpc-work/dissertation/checkpoints/gmm2_checkpoints/params_checkpoints/{input_dim}D_gmm2_score_params_{num_steps}steps_{args.sample_index}_{cov_form}_with_time_stepsFalse.pth"

    checkpoint = torch.load(filename, map_location=device)
    return checkpoint

def load_and_process_data(input_dim, args):
    """
    Load and process the data for a specific dimension.
    If data is not available, generate it using ddpm_sampler_low_rank_model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the score model
    score_model = ScoreNet(input_dim=input_dim, n_layers=7, hidden_size=512).to(device)
    score_model.load_state_dict(torch.load(CHECKPOINTS_DIR / 'gmm2_checkpoints/model_checkpoints/{input_dim}D_gmm2_score_ckpt_7layers_512hidden_size.pth', map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    # Create the GMM (true distribution)
    gmm = create_gmm(input_dim, device=device)

    # Define paths
    backward_pkl_path = CHECKPOINTS_DIR / 'gmm2_checkpoints/sample_checkpoints/{input_dim}D_gmm2_backward_{args.num_steps}steps_{args.sample_index}sample_{args.cov_form}_rank{args.rank}.pkl'

    # First try to load data from saved file
    try:
        with open(backward_pkl_path, 'rb') as f:
            backward_results = pickle.load(f)
        print(f"Loaded samples from {backward_pkl_path}")
        
        # Extract samples and weights
        try:
            samples = backward_results['samples']
            weights = backward_results['weights']
        except KeyError:
            print("Error: Sample file does not contain expected data structure")
            raise
            
    except (FileNotFoundError, KeyError):
        print(f"No data file found at {backward_pkl_path}, generating samples on the fly...")
        
        # Initialize covariance predictor model
        rank = args.rank if args.rank is not None else input_dim
        print(f"Using low-rank covariance model with rank {rank}")
        
        # Generate samples using low rank model
        ckpt = load_covariance_params(args, input_dim, args.cov_form, args.num_steps)
        samples, weights, reverse_ess = score_model.ddpm_sampler(
            num_steps=args.num_steps,
            num_samples=args.num_samples,
            true_gmm=gmm,
            cov_form=args.cov_form,
            progress_bar=True,
            cov_params=ckpt["cov_params"]
        )

        # Save generated samples
        # if args.save_generated:
        #     os.makedirs(os.path.dirname(backward_pkl_path), exist_ok=True)
        #     torch.save({
        #         'samples': samples,
        #         'weights': weights
        #     }, backward_pkl_path)
        #     print(f"Saved generated samples to {backward_pkl_path}")

    # Limit to requested number of samples
    x = samples[:args.num_samples].to(device)
    w = weights[:args.num_samples].to(device)

    ess_pct = reverse_ess / args.num_samples * 100

    print(f"Reverse ESS (%) for {args.cov_form} {input_dim}D: {ess_pct:.2f}%")

    # Generate true samples
    true_samples = gmm.sample(args.num_samples)

    # Compute log probabilities
    with torch.no_grad():
        model_log_probs = gmm.log_prob(x)
        true_log_probs = gmm.log_prob(true_samples)

    return {
        'model_log_probs': model_log_probs.detach().cpu().numpy(),
        'true_log_probs': true_log_probs.detach().cpu().numpy(),
        'weights': w.detach().cpu().numpy(),
        'ess_pct': ess_pct
    }

def plot_histograms(args):
    """
    Plot true, unweighted, and reweighted histograms of log probabilities
    side by side for multiple input dimensions
    """
    all_data = {}
    
    # Load data for each dimension
    for dim in args.input_dims:
        data = load_and_process_data(dim, args)
        if data is not None:
            all_data[dim] = data
    
    if not all_data:
        print("No data loaded. Exiting.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, len(all_data), figsize=(14, 4), sharey=True)
    
    # If only one dimension, make axes iterable
    if len(all_data) == 1:
        axes = [axes]
    
    # To track handles for the legend
    legend_handles = []
    legend_labels = []
    
    # Define colors to use consistently across plots
    colors = ['blue', '#ff7f0e', '#2ca02c']
    
    for i, (dim, data) in enumerate(all_data.items()):
        ax = axes[i]
        model_log_probs = data['model_log_probs']
        true_log_probs = data['true_log_probs']
        weights = data['weights']
        
        # Determine sensible range based on data
        min_val = min(model_log_probs.min(), true_log_probs.min())
        max_val = max(model_log_probs.max(), true_log_probs.max())
        range_adjust = (max_val - min_val) * 0.1
        plot_range = [min_val - range_adjust, max_val + range_adjust]
        
        # Determine number of bins based on input dimension
        # num_bins = min(50 + dim // 10, 100)  # More bins for higher dimensions
        num_bins = 100

        # 1) True log probability (Ground Truth)
        h1 = ax.hist(true_log_probs, bins=num_bins, range=plot_range, density=True, 
                     histtype='step', linewidth=2, alpha=0.9,
                     color=colors[0], label='Ground Truth')
                     
        # 2) Model log probability (unweighted)
        h2 = ax.hist(model_log_probs, bins=num_bins, range=plot_range, density=True, 
                     histtype='step', linewidth=2, alpha=0.9,
                     color=colors[1], label='Diffusion Samples (Unweighted)')
                     
        # 3) Reweighted model log probability
        h3 = ax.hist(model_log_probs, bins=num_bins, range=plot_range, density=True, 
                     weights=weights, histtype='step', linewidth=2, alpha=0.9,
                     color=colors[2], label='Diffusion Samples (Reweighted)')
        
        # Only save handles from the first plot
        if i == 0:
            # Create proxy artists for the legend
            from matplotlib.lines import Line2D

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
            legend_handles = [
                Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color=colors[2], lw=2)
            ]
            legend_labels = [
                'Ground Truth',
                'Diffusion Samples (Unweighted)',
                'Diffusion Samples (Reweighted)'
            ]
        
        ax.set_xlabel('Log Probability', fontsize=14)
        ax.set_title(f'{dim}D GMM ({args.num_steps} steps)', fontsize=16)
        ax.grid(alpha=0.3)

        # Set the same x-limits for all histograms
        if dim == 50:
            ax.set_xlim([-45, -5])
        elif dim == 100:
            ax.set_xlim([-80, -20])
    
    # Set y label only for the leftmost plot
    axes[0].set_ylabel('Normalized Density', fontsize=14)
    
    # Create a single legend below the plots
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=12
    )
    
    plt.tight_layout()
    # Add more bottom space for the legend
    plt.subplots_adjust(bottom=0.2)
    
    # Create figures directory if it doesn't exist
    figures_dir = FIGURES_DIR / 'hist'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save figure
    dims_str = '_'.join(str(d) for d in args.input_dims)
    save_path = f'{figures_dir}/{dims_str}D_gmm_{args.cov_form}_reweight_{args.num_steps}steps_rank{args.rank}.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dims', nargs='+', type=int, default=[50, 100],
                        help='Dimensionalities of the GMM to plot')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to use for histograms')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of sampling steps')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of sample batch to use')
    parser.add_argument('--cov_form', type=str, default='isotropic',
                        choices=['ddpm', 'isotropic', 'diagonal', 'full'],
                        help='Covariance form to use for plotting')
    parser.add_argument('--rank', type=int, default=None,
                        help='Rank of full covariance matrix (if applicable)')
    parser.add_argument('--save_generated', action='store_true', default=True,
                        help='Save generated samples to disk')
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    plot_histograms(args)

if __name__ == '__main__':
    main() 