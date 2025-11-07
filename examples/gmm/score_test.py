import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gmm import create_gmm
from score_model import ScoreNet
from cov_tuned_diffusion.utils.path_config import FIGURES_DIR, get_gmm2_model_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Quick sanity check for trained GMM2 score models.")
    parser.add_argument("--input-dim", type=int, default=2, help="Dimensionality of the GMM.")
    parser.add_argument("--n-layers", type=int, default=7, help="Number of GNN layers.")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden dimension of the network.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to draw.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of reverse diffusion steps.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store diagnostic plots (defaults to figures/gmm2/diagnostics).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_gmm2_model_checkpoint_path(
        input_dim=args.input_dim,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    score_model = ScoreNet(
        input_dim=args.input_dim,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
    ).to(device)
    score_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    gmm = create_gmm(args.input_dim, device=device)

    with torch.no_grad():
        samples, _weights, ess = score_model.ddpm_sampler(
            num_steps=args.num_steps,
            num_samples=args.num_samples,
            true_gmm=gmm,
        )

        true_samples = gmm.sample(args.num_samples)
        model_log_prob = gmm.log_prob(samples)
        true_log_prob = gmm.log_prob(true_samples)

    reverse_ess = ess.item() / args.num_samples * 100
    print(f"Reverse ESS: {reverse_ess:.2f}%")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (FIGURES_DIR / "gmm2" / "diagnostics")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_log_prob_np = model_log_prob.detach().cpu().numpy()
    true_log_prob_np = true_log_prob.detach().cpu().numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(model_log_prob_np, bins=50, alpha=0.6, label="Model", density=True)
    plt.hist(true_log_prob_np, bins=50, alpha=0.6, label="Ground Truth", density=True)
    plt.xlabel("log p(x)")
    plt.ylabel("Density")
    plt.title(f"Log-probability histogram ({args.input_dim}D)")
    plt.legend()
    log_prob_path = output_dir / f"log_prob_hist_{args.input_dim}D.pdf"
    plt.tight_layout()
    plt.savefig(log_prob_path, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to {log_prob_path}")

    samples_np = samples.detach().cpu().numpy()
    true_samples_np = true_samples.detach().cpu().numpy()

    if args.input_dim == 1:
        plt.figure(figsize=(7, 4))
        plt.hist(samples_np[:, 0], bins=100, alpha=0.6, label="Model", density=True)
        plt.hist(true_samples_np[:, 0], bins=100, alpha=0.6, label="Ground Truth", density=True)
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Sample histogram")
        plt.legend()
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(
            true_samples_np[:, 0],
            true_samples_np[:, 1],
            alpha=0.3,
            s=5,
            label="Ground Truth",
        )
        plt.scatter(
            samples_np[:, 0],
            samples_np[:, 1],
            alpha=0.3,
            s=5,
            label="Model",
        )
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Sample scatter (first two dims)")
        plt.legend()

    scatter_path = output_dir / f"samples_visualization_{args.input_dim}D.png"
    plt.tight_layout()
    plt.savefig(scatter_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved sample visualization to {scatter_path}")


if __name__ == "__main__":
    main()
