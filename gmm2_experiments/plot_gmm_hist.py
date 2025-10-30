import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from gmm import create_gmm
from score_model import ScoreNet
from utils.path_config import (
    FIGURES_DIR,
    get_gmm2_model_checkpoint_path,
    get_gmm2_params_checkpoint_path,
    get_gmm2_sample_path,
)


def load_covariance_params(args, input_dim, cov_form, num_steps):
    """Load tuned covariance parameters for the requested configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_path = get_gmm2_params_checkpoint_path(
        input_dim=input_dim,
        num_steps=num_steps,
        params_index=args.sample_index,
        cov_form=cov_form,
        tune_time_steps=False,
        rank=args.rank if cov_form == "full" else None,
    )
    return torch.load(params_path, map_location=device)


def _to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, device=device)


def load_and_process_data(input_dim, args):
    """
    Load and process the data for a specific dimension.

    If pre-generated samples are not available they are generated on the fly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the score model
    score_model = ScoreNet(input_dim=input_dim, n_layers=7, hidden_size=512).to(device)
    score_ckpt_path = get_gmm2_model_checkpoint_path(
        input_dim=input_dim, n_layers=7, hidden_size=512
    )
    score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    # Create the GMM (true distribution)
    gmm = create_gmm(input_dim, device=device)

    # Define paths
    backward_pkl_path = get_gmm2_sample_path(
        input_dim=input_dim,
        num_steps=args.num_steps,
        sample_index=args.sample_index,
        cov_form=args.cov_form,
        direction="backward",
        rank=args.rank if args.cov_form == "full" else None,
    )

    reverse_ess = None

    # First try to load data from saved file
    try:
        with open(backward_pkl_path, "rb") as f:
            backward_results = pickle.load(f)
        print(f"Loaded samples from {backward_pkl_path}")

        # Extract samples and weights
        samples = backward_results["samples"]
        weights = backward_results["weights"]
        reverse_ess = backward_results.get("reverse_ess")

    except (FileNotFoundError, KeyError):
        print(
            f"No data file found at {backward_pkl_path}, generating samples on the fly..."
        )

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
            cov_params=ckpt["cov_params"],
        )

    # Convert to tensors on the correct device
    samples = _to_tensor(samples, device)
    weights = _to_tensor(weights, device)

    # Limit to requested number of samples
    x = samples[: args.num_samples]
    w = weights[: args.num_samples]

    ess_pct = None
    if reverse_ess is not None:
        ess_pct = reverse_ess / args.num_samples * 100
        print(f"Reverse ESS (%) for {args.cov_form} {input_dim}D: {ess_pct:.2f}%")

    # Generate true samples and compute log probabilities
    true_samples = gmm.sample(args.num_samples)

    with torch.no_grad():
        model_log_probs = gmm.log_prob(x)
        true_log_probs = gmm.log_prob(true_samples)

    return {
        "model_log_probs": model_log_probs.detach().cpu().numpy(),
        "true_log_probs": true_log_probs.detach().cpu().numpy(),
        "weights": w.detach().cpu().numpy(),
        "ess_pct": ess_pct,
    }


def plot_histograms(args):
    """
    Plot true, unweighted, and reweighted histograms of log probabilities
    side by side for multiple input dimensions.
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
    colors = ["blue", "#ff7f0e", "#2ca02c"]

    for i, (dim, data) in enumerate(all_data.items()):
        ax = axes[i]
        model_log_probs = data["model_log_probs"]
        true_log_probs = data["true_log_probs"]
        weights = data["weights"]

        # Determine sensible range based on data
        min_val = min(model_log_probs.min(), true_log_probs.min())
        max_val = max(model_log_probs.max(), true_log_probs.max())
        range_adjust = (max_val - min_val) * 0.1
        plot_range = [min_val - range_adjust, max_val + range_adjust]

        # Determine number of bins based on input dimension
        num_bins = 100

        # 1) True log probability (Ground Truth)
        ax.hist(
            true_log_probs,
            bins=num_bins,
            range=plot_range,
            density=True,
            histtype="step",
            linewidth=2,
            alpha=0.9,
            color=colors[0],
            label="Ground Truth",
        )

        # 2) Model log probability (unweighted)
        ax.hist(
            model_log_probs,
            bins=num_bins,
            range=plot_range,
            density=True,
            histtype="step",
            linewidth=2,
            alpha=0.9,
            color=colors[1],
            label="Diffusion Samples (Unweighted)",
        )

        # 3) Reweighted model log probability
        ax.hist(
            model_log_probs,
            bins=num_bins,
            range=plot_range,
            density=True,
            weights=weights,
            histtype="step",
            linewidth=2,
            alpha=0.9,
            color=colors[2],
            label="Diffusion Samples (Reweighted)",
        )

        # Only save handles from the first plot
        if i == 0:
            legend_handles = [
                Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color=colors[2], lw=2),
            ]
            legend_labels = [
                "Ground Truth",
                "Diffusion Samples (Unweighted)",
                "Diffusion Samples (Reweighted)",
            ]

        ax.set_xlabel("Log Probability", fontsize=14)
        ax.set_title(f"{dim}D GMM ({args.num_steps} steps)", fontsize=16)
        ax.grid(alpha=0.3)

        # Set the same x-limits for all histograms
        if dim == 50:
            ax.set_xlim([-45, -5])
        elif dim == 100:
            ax.set_xlim([-80, -20])

    # Set y label only for the leftmost plot
    axes[0].set_ylabel("Normalized Density", fontsize=14)

    # Create a single legend below the plots
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    figures_dir = FIGURES_DIR / "gmm2" / "hist"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dims_str = "_".join(str(d) for d in args.input_dims)
    rank_suffix = f"_rank{args.rank}" if args.rank is not None else ""
    filename = (
        f"{dims_str}D_gmm_{args.cov_form}_reweight_{args.num_steps}steps{rank_suffix}.pdf"
    )
    save_path = figures_dir / filename
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dims",
        nargs="+",
        type=int,
        default=[50, 100],
        help="Dimensionalities of the GMM to plot",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to use for histograms",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of sample batch to use",
    )
    parser.add_argument(
        "--cov_form",
        type=str,
        default="isotropic",
        choices=["ddpm", "isotropic", "diagonal", "full"],
        help="Covariance form to use for plotting",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank of full covariance matrix (if applicable)",
    )
    parser.add_argument(
        "--save_generated",
        action="store_true",
        default=True,
        help="Save generated samples to disk",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    plot_histograms(args)


if __name__ == "__main__":
    main()
