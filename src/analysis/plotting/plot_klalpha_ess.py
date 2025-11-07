import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401 (kept for backward compatibility)
import torch
import yaml
from tqdm import tqdm

from cov_tuned_diffusion import (
    ScoreNet,
    compute_forward_ess,
    compute_reverse_ess,
    load_dataset,
    load_target_dist,
)
from cov_tuned_diffusion.utils.path_config import (
    FIGURES_DIR,
    get_config_path,
    get_model_checkpoint_path,
    get_params_checkpoint_path,
)


def load_parameters(dataset: str, net: str, params_index: int, num_steps: int, alpha: float) -> Dict:
    """Load tuned parameters for a specific alpha value."""
    params_path = get_params_checkpoint_path(
        dataset,
        net,
        params_index=params_index,
        num_steps=num_steps,
        alpha=alpha,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(params_path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description="Plot ESS as a function of alpha and sample size.")
    parser.add_argument("--dataset", type=str, default="aldp")
    parser.add_argument("--net", type=str, default="egnn")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--params-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000],
        help="Forward/backward sample counts to evaluate.",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Number of Monte Carlo runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store outputs (defaults to figures/alpha_ess).",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Base name for output files (without extension).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (FIGURES_DIR / "alpha_ess")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_name is None:
        args.save_name = f"{args.dataset}_alpha_comparison_ess"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = get_config_path(args.dataset, args.net, "score")
    with open(config_path, "r", encoding="utf-8") as f:
        score_model_config = yaml.safe_load(f)

    score_checkpoint_path = get_model_checkpoint_path(
        args.dataset,
        args.net,
        model_type="score",
        model_index=args.model_index,
    )

    score_model = ScoreNet(
        dataset=args.dataset,
        device=device,
        model_config=score_model_config,
        net=args.net,
    ).to(device)
    score_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint["model_state_dict"])
    score_model.eval()
    score_model.requires_grad_(False)

    true_target_dist = load_target_dist(args.dataset)
    true_data = load_dataset(args.dataset, partition="test")

    results: Dict[str, Dict[float, Dict[int, list]]] = {
        "forward": {alpha: {n: [] for n in args.sample_sizes} for alpha in args.alphas},
        "reverse": {alpha: {n: [] for n in args.sample_sizes} for alpha in args.alphas},
    }

    for alpha in args.alphas:
        print(f"Computing ESS with alpha={alpha}")
        params = load_parameters(
            dataset=args.dataset,
            net=args.net,
            params_index=args.params_index,
            num_steps=args.num_steps,
            alpha=alpha,
        )

        for n_samples in args.sample_sizes:
            print(f"Sample size: {n_samples}")
            for run in range(args.num_runs):
                print(f"  Run {run + 1}/{args.num_runs}")

                indices = torch.randperm(true_data.shape[0])[:n_samples]
                x0 = true_data[indices].to(device)
                log_prob_x0 = true_target_dist.log_prob(x0)

                with torch.no_grad():
                    _, _, log_w_forward = score_model.est_forward_ess(
                        x0,
                        log_prob_x0,
                        args.num_steps,
                        nus=params.get("nus"),
                        time_steps=params.get("time_steps"),
                        progress_bar=True,
                    )

                    *_rest, log_w_backward = score_model.ddpm_sampler(
                        args.num_steps,
                        true_target_dist,
                        num_samples=n_samples,
                        nus=params.get("nus"),
                        time_steps=params.get("time_steps"),
                        progress_bar=True,
                    )

                forward_ess = compute_forward_ess(log_w_forward).item() / n_samples * 100
                reverse_ess = compute_reverse_ess(log_w_backward).item() / n_samples * 100

                results["forward"][alpha][n_samples].append(forward_ess)
                results["reverse"][alpha][n_samples].append(reverse_ess)

                print(f"    Forward ESS: {forward_ess:.2f}% | Reverse ESS: {reverse_ess:.2f}%")

    summary: Dict[str, Dict[float, Dict[int, Dict[str, float]]]] = {
        "forward": {},
        "reverse": {},
    }

    for direction in ["forward", "reverse"]:
        summary[direction] = {}
        for alpha in args.alphas:
            summary[direction][alpha] = {}
            for n_samples in args.sample_sizes:
                values = np.array(results[direction][alpha][n_samples], dtype=float)
                summary[direction][alpha][n_samples] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

    dataset_title = {
        "dw4": "DW-4",
        "lj13": "LJ-13",
        "lj55": "LJ-55",
        "aldp": "Alanine Dipeptide",
    }

    plt.figure(figsize=(7, 4))
    forward_colors = {1.0: "#1f77b4", 2.0: "#d62728"}
    reverse_colors = {1.0: "#76b7b2", 2.0: "#ff9896"}
    markers = {1.0: "o", 2.0: "s"}
    linestyles = {"forward": "-", "reverse": "--"}

    for alpha in args.alphas:
        sample_sizes = list(args.sample_sizes)

        forward_means = [summary["forward"][alpha][n]["mean"] for n in sample_sizes]
        forward_stds = [summary["forward"][alpha][n]["std"] for n in sample_sizes]
        plt.errorbar(
            sample_sizes,
            forward_means,
            yerr=forward_stds,
            fmt=markers[alpha] + linestyles["forward"],
            color=forward_colors.get(alpha, "#1f77b4"),
            ecolor=forward_colors.get(alpha, "#1f77b4"),
            elinewidth=1.5,
            capsize=5,
            markersize=6,
            label=rf"$\alpha={alpha}$ Forward",
        )

        reverse_means = [summary["reverse"][alpha][n]["mean"] for n in sample_sizes]
        reverse_stds = [summary["reverse"][alpha][n]["std"] for n in sample_sizes]
        plt.errorbar(
            sample_sizes,
            reverse_means,
            yerr=reverse_stds,
            fmt=markers[alpha] + linestyles["reverse"],
            color=reverse_colors.get(alpha, "#76b7b2"),
            ecolor=reverse_colors.get(alpha, "#76b7b2"),
            elinewidth=1.5,
            capsize=5,
            markersize=6,
            label=rf"$\alpha={alpha}$ Reverse",
        )

    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("ESS (%)", fontsize=14)
    dataset_name = dataset_title.get(args.dataset, args.dataset.upper())
    plt.title(f"ESS (Mean ± Std) for {dataset_name}", fontsize=16)
    plt.grid(alpha=0.3)
    plt.xscale("log")
    plt.ylim(bottom=0)
    plt.legend(fontsize=11, ncol=len(args.alphas), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    save_base = output_dir / args.save_name
    plt.savefig(save_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(save_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print("Plots saved to:")
    print(f"  - {save_base.with_suffix('.png')}")
    print(f"  - {save_base.with_suffix('.pdf')}")

    json_path = save_base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()
