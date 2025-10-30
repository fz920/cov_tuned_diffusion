import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from score_model import ScoreNet
from gmm import create_gmm

from utils.path_config import (
    FIGURES_DIR,
    get_gmm2_ess_summary_path,
    get_gmm2_model_checkpoint_path,
    get_gmm2_params_checkpoint_path,
)

def load_covariance_params(args, cov_form, idx, num_steps):
    """
    Load covariance parameters for a specific form
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params_path = get_gmm2_params_checkpoint_path(
        input_dim=args.input_dim,
        num_steps=num_steps,
        params_index=idx,
        cov_form=cov_form,
        tune_time_steps=args.tune_time_steps,
        rank=args.rank if cov_form == "full" else None,
    )
    return torch.load(params_path, map_location=device)


def compute_ess(args):
    """
    Compute forward ESS for different covariance forms and number of steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the score model
    score_model = ScoreNet(input_dim=args.input_dim, n_layers=7, hidden_size=512).to(device)
    score_ckpt_path = get_gmm2_model_checkpoint_path(
        input_dim=args.input_dim, n_layers=7, hidden_size=512
    )
    score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    # Create the GMM
    gmm = create_gmm(args.input_dim, device=device)

    results_forward_ess = {
      cov_form: {n: [] for n in args.num_steps_list}
      for cov_form in args.cov_forms
    }

    results_reverse_ess = {
      cov_form: {n: [] for n in args.num_steps_list}
      for cov_form in args.cov_forms
    }

    # Sample from the GMM once
    x0 = gmm.sample(args.num_samples)
    log_prob_x0 = gmm.log_prob(x0)

    for num_steps in args.num_steps_list:
        for cov_form in args.cov_forms:
            score_model.cov_form = cov_form
            for idx in args.params_index_list:
                if cov_form == 'ddpm':
                    cov_params = None
                    time_steps = None
                else:
                    ckpt = load_covariance_params(args, cov_form, idx, num_steps)
                    cov_params = ckpt['cov_params']
                    time_steps = ckpt['time_steps']

                with torch.no_grad():
                    _, forward_ess, _ = score_model.est_forward_ess(
                        x0, log_prob_x0, num_steps,
                        cov_params=cov_params,
                        time_steps=time_steps,
                        cov_form=cov_form,
                        progress_bar=True
                    )
                    pct = forward_ess.item() / args.num_samples * 100
                    results_forward_ess[cov_form][num_steps].append(pct)

                    _, _, reverse_ess = score_model.ddpm_sampler(
                        num_steps, args.num_samples,
                        cov_params=cov_params,
                        time_steps=time_steps,
                        cov_form=cov_form,
                        true_gmm=gmm,
                        progress_bar=True
                    )
                    pct = reverse_ess.item() / args.num_samples * 100
                    results_reverse_ess[cov_form][num_steps].append(pct)

    # compute mean and quantiles
    summary_forward_ess = {}
    for cov_form, by_steps in results_forward_ess.items():
        arrs = by_steps
        median, p25, p75, mean, std = [], [], [], [], []
        for n in args.num_steps_list:
            a = np.array(arrs[n])
            median.append(np.median(a))
            p25.append(np.percentile(a, 25))
            p75.append(np.percentile(a, 75))
            mean.append(np.mean(a))
            std.append(np.std(a))
        summary_forward_ess[cov_form] = {"median": median, "p25": p25, "p75": p75, "mean": mean, "std": std}
    
    summary_reverse_ess = {}
    for cov_form, by_steps in results_reverse_ess.items():
        arrs = by_steps
        median, p25, p75, mean, std = [], [], [], [], []
        for n in args.num_steps_list:
            a = np.array(arrs[n])
            median.append(np.median(a))
            p25.append(np.percentile(a, 25))
            p75.append(np.percentile(a, 75))
            mean.append(np.mean(a))
            std.append(np.std(a))
        summary_reverse_ess[cov_form] = {"median": median, "p25": p25, "p75": p75, "mean": mean, "std": std}

    # save both summaries to file
    summary_path = get_gmm2_ess_summary_path(
        input_dim=args.input_dim,
        num_steps=args.num_steps_list,
        params_indices=args.params_index_list,
        rank=args.rank,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'forward_ess': summary_forward_ess,
        'reverse_ess': summary_reverse_ess
    }, summary_path)

    return summary_forward_ess, summary_reverse_ess

def plot_ess(all_summaries, args):
    """
    Plot both forward and reverse ESS for multiple input dimensions side by side
    with a shared legend at the bottom.
    
    Args:
        all_summaries: Dictionary with keys being input dimensions and values being
                      dictionaries with 'forward_ess' and 'reverse_ess' summaries
    """
    fig, axes = plt.subplots(1, len(all_summaries), figsize=(14, 4))
    
    forward_colors = {
        'ddpm': '#7570b3',  # purple
        'isotropic': '#1f77b4',  # strong blue
        'diagonal': '#ff7f0e',  # orange
        'full':  '#d62728',  # strong red
    }

    reverse_colors = {
        'ddpm': '#c2a5cf',  # light purple
        'isotropic': '#76b7b2',  # muted cyan/light blue (pairs w/ blue)
        'diagonal': '#ffbb78',  # light orange
        'full':  '#ff9896',  # pinkish (pairs w/ red)
    }

    markers = {'ddpm': 'd', 'isotropic': 'o', 'diagonal': 's', 'full': '^'}
    full_label = 'Full' + (f' (rank {args.rank})' if args.rank is not None else '')
    labels = {'ddpm': 'DDPM', 'isotropic': 'Isotropic', 'diagonal': 'Diagonal', 'full': full_label}
    linestyles = {'forward': '-', 'reverse': '--'}
    
    # Keep track of handles and labels for the legend
    all_handles = []
    all_labels = []
    
    for i, (dim, summaries) in enumerate(all_summaries.items()):
        ax = axes[i] if len(all_summaries) > 1 else axes
        summary_forward_ess = summaries['forward_ess']
        summary_reverse_ess = summaries['reverse_ess']
        
        handles = []
        
        for cov_form in args.cov_forms:
            x = args.num_steps_list
            marker = markers[cov_form]

            # Forward ESS
            fwd = summary_forward_ess[cov_form]
            y = np.array(fwd["mean"])
            std = np.array(fwd["std"])
            h1 = ax.errorbar(
                x, y,
                yerr=std,
                fmt=marker + linestyles['forward'],
                color=forward_colors[cov_form],
                ecolor=forward_colors[cov_form],
                elinewidth=1.5,
                capsize=5,
                markersize=6,
                label=f"{labels[cov_form]} Forward"
            )
            handles.append(h1)

            # Reverse ESS
            rev = summary_reverse_ess[cov_form]
            y = np.array(rev["mean"])
            std = np.array(rev["std"])
            h2 = ax.errorbar(
                x, y,
                yerr=std,
                fmt=marker + linestyles['reverse'],
                color=reverse_colors[cov_form],
                ecolor=reverse_colors[cov_form],
                elinewidth=1.5,
                capsize=5,
                markersize=6,
                label=f"{labels[cov_form]} Reverse"
            )
            handles.append(h2)

        ax.set_xlabel('Number of Steps', fontsize=14)
        ax.set_title(f'ESS for {dim}D GMM', fontsize=16)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Only add handles from the first plot to avoid duplicates
        if i == 0:
            all_handles = handles
            all_labels = [h.get_label() for h in handles]
    
    # Set y-label only for the leftmost plot
    if len(all_summaries) > 1:
        axes[0].set_ylabel('ESS (%)', fontsize=14)
    else:
        axes.set_ylabel('ESS (%)', fontsize=14)
    
    # Create a single legend below the plots
    fig.legend(
        all_handles, 
        all_labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.14),
        ncol=4, 
        fontsize=16
    )
    
    plt.tight_layout()
    # Add more bottom space for the legend
    plt.subplots_adjust(bottom=0.2)
    
    if args.save_path:
        plt.savefig(args.save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {args.save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dims', nargs='+', type=int, default=[50, 100],
                         help='Input dimensions to plot')
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument(
        '--num_steps_list', nargs='+', type=int,
        default=[5, 10, 20, 30, 40, 50],
        help="List of step counts, e.g. --num_steps_list 5 10 20"
    )
    parser.add_argument(
        '--params_index_list', nargs='+', type=int,
        default=[0, 0, 0],
        help="Which checkpoint indices to load"
    )
    parser.add_argument('--rank', type=int, default=None, help='Rank of the full covariance matrix')
    parser.add_argument('--tune_time_steps', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.cov_forms = ['ddpm', 'isotropic', 'diagonal', 'full']

    if args.save_path is None:
        dims_str = '_'.join(map(str, args.input_dims))
        figures_dir = FIGURES_DIR / 'gmm2' / 'ess'
        figures_dir.mkdir(parents=True, exist_ok=True)
        args.save_path = figures_dir / f'ess_plot_{dims_str}D.pdf'
    else:
        args.save_path = Path(args.save_path).expanduser().resolve()
        args.save_path.parent.mkdir(parents=True, exist_ok=True)

    # Dictionary to store summaries for all dimensions
    all_summaries = {}
    
    # Compute or load ESS for each input dimension
    for input_dim in args.input_dims:
        # Temporarily set input_dim in args for compute_ess
        args.input_dim = input_dim
        
        summary_file_path = get_gmm2_ess_summary_path(
            input_dim=input_dim,
            num_steps=args.num_steps_list,
            params_indices=args.params_index_list,
            rank=args.rank,
        )

        if summary_file_path.exists():
            print(f"Loading existing summary from {summary_file_path}")
            ess_checkpoints = torch.load(summary_file_path)
            summary_forward_ess = ess_checkpoints['forward_ess']
            summary_reverse_ess = ess_checkpoints['reverse_ess']
        else:
            summary_forward_ess, summary_reverse_ess = compute_ess(args)
        
        all_summaries[input_dim] = {
            'forward_ess': summary_forward_ess,
            'reverse_ess': summary_reverse_ess
        }

    # Plot results for all dimensions
    plot_ess(all_summaries, args)

if __name__ == '__main__':
    main()
