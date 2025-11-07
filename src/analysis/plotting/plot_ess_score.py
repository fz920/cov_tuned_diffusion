import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from cov_tuned_diffusion.utils.path_config import CHECKPOINTS_DIR, FIGURES_DIR


def calculate_ess(log_w: torch.Tensor, mode: str = "reverse") -> torch.Tensor:
    num_samples = log_w.shape[0]
    if mode == "reverse":
        weights = torch.exp(log_w - torch.max(log_w))
        weights = weights / torch.sum(weights)
        ess = 1 / torch.sum(weights ** 2)
    elif mode == "forward":
        z_inv = torch.mean(torch.exp(-log_w))
        ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * z_inv)
    else:
        raise ValueError(f"Unknown ESS mode: {mode}")
    return ess


def percentile_summary(values: Iterable[float]) -> Dict[str, float]:
    data = np.array(list(values), dtype=float)
    return {
        "q25": np.percentile(data, 25),
        "q50": np.percentile(data, 50),
        "q75": np.percentile(data, 75),
    }


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ESS of tuned diffusion (ours) versus baseline score models."
    )
    parser.add_argument("--dataset", type=str, default="aldp", help="Dataset to use.")
    parser.add_argument("--net", type=str, default="egnn", help="Network backbone.")
    parser.add_argument("--model-index", type=int, default=0, help="Model checkpoint index.")
    parser.add_argument(
        "--params-index-list",
        type=int,
        nargs="+",
        default=[0],
        help="Indices of tuned parameter checkpoints to include.",
    )
    parser.add_argument(
        "--num-steps-list",
        type=int,
        nargs="+",
        default=[100],
        help="Diffusion step counts to evaluate.",
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        default=None,
        help="Root directory containing importance sampling results (defaults to checkpoints/samples).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the figure (defaults to figures/ess).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_root = (
        Path(args.samples_root).expanduser().resolve()
        if args.samples_root
        else (CHECKPOINTS_DIR / "importance_sampling" / "samples")
    )

    ours_dir = samples_root / args.dataset / "ours"
    baseline_dir = samples_root / args.dataset / "score"

    reverse_ours: Dict[int, Dict[int, float]] = {n: {} for n in args.num_steps_list}
    forward_ours: Dict[int, Dict[int, float]] = {n: {} for n in args.num_steps_list}
    reverse_baseline: Dict[int, Dict[int, float]] = {n: {} for n in args.num_steps_list}
    forward_baseline: Dict[int, Dict[int, float]] = {n: {} for n in args.num_steps_list}

    for num_steps in args.num_steps_list:
        for params_idx in args.params_index_list:
            ours_path = (
                ours_dir
                / f"{args.net}_ours_{args.model_index}model_{num_steps}steps_{params_idx}.pth"
            )
            ours_forward_path = (
                ours_dir
                / f"{args.net}_ours_forward_{args.model_index}model_{num_steps}steps_{params_idx}.pth"
            )
            baseline_path = (
                baseline_dir
                / f"{args.net}_ddpm_{args.model_index}model_{num_steps}steps_{params_idx}.pth"
            )
            baseline_forward_path = (
                baseline_dir
                / f"{args.net}_ddpm_forward_{args.model_index}model_{num_steps}steps_{params_idx}.pth"
            )

            ours_ckpt = load_checkpoint(ours_path, device)
            ours_forward_ckpt = load_checkpoint(ours_forward_path, device)
            baseline_ckpt = load_checkpoint(baseline_path, device)
            baseline_forward_ckpt = load_checkpoint(baseline_forward_path, device)

            reverse_ours_ess = calculate_ess(ours_ckpt["log_weights"], mode="reverse")
            forward_ours_ess = calculate_ess(ours_forward_ckpt["log_weights"], mode="forward")
            reverse_baseline_ess = calculate_ess(baseline_ckpt["log_weights"], mode="reverse")
            forward_baseline_ess = calculate_ess(baseline_forward_ckpt["log_weights"], mode="forward")

            sample_count = ours_ckpt["log_weights"].shape[0]
            reverse_ours[num_steps][params_idx] = (
                reverse_ours_ess.detach().cpu().item() / sample_count * 100
            )
            forward_ours[num_steps][params_idx] = (
                forward_ours_ess.detach().cpu().item() / sample_count * 100
            )
            reverse_baseline[num_steps][params_idx] = (
                reverse_baseline_ess.detach().cpu().item() / sample_count * 100
            )
            forward_baseline[num_steps][params_idx] = (
                forward_baseline_ess.detach().cpu().item() / sample_count * 100
            )

    num_steps_list: List[int] = args.num_steps_list

    ours_forward_stats = [percentile_summary(forward_ours[steps].values()) for steps in num_steps_list]
    ours_reverse_stats = [percentile_summary(reverse_ours[steps].values()) for steps in num_steps_list]
    baseline_forward_stats = [percentile_summary(forward_baseline[steps].values()) for steps in num_steps_list]
    baseline_reverse_stats = [percentile_summary(reverse_baseline[steps].values()) for steps in num_steps_list]

    plt.figure(figsize=(7, 4))
    x_values = num_steps_list

    def as_yerr(stats):
        lower = [s["q50"] - s["q25"] for s in stats]
        upper = [s["q75"] - s["q50"] for s in stats]
        return np.vstack([lower, upper])

    plt.errorbar(
        x_values,
        [s["q50"] for s in ours_forward_stats],
        yerr=as_yerr(ours_forward_stats),
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label="Ours (Forward ESS)",
        linewidth=2,
        capsize=3,
    )
    plt.errorbar(
        x_values,
        [s["q50"] for s in ours_reverse_stats],
        yerr=as_yerr(ours_reverse_stats),
        marker="o",
        linestyle="-",
        color="#76b7b2",
        label="Ours (Reverse ESS)",
        linewidth=2,
        capsize=3,
    )
    plt.errorbar(
        x_values,
        [s["q50"] for s in baseline_forward_stats],
        yerr=as_yerr(baseline_forward_stats),
        marker="s",
        linestyle="--",
        color="#d62728",
        label="DDPM (Forward ESS)",
        linewidth=2,
        capsize=3,
    )
    plt.errorbar(
        x_values,
        [s["q50"] for s in baseline_reverse_stats],
        yerr=as_yerr(baseline_reverse_stats),
        marker="s",
        linestyle="--",
        color="#ff9896",
        label="DDPM (Reverse ESS)",
        linewidth=2,
        capsize=3,
    )

    plt.xlabel("Number of diffusion steps")
    plt.ylabel("ESS (%)")
    plt.title(f"ESS comparison on {args.dataset.upper()} ({args.net})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output is None:
        output_dir = FIGURES_DIR / "ess"
        output_dir.mkdir(parents=True, exist_ok=True)
        steps_str = "_".join(map(str, num_steps_list))
        params_str = "_".join(map(str, args.params_index_list))
        output_path = output_dir / (
            f"ess_plot_{args.dataset}_{args.net}_{steps_str}steps_{params_str}params.pdf"
        )
    else:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
