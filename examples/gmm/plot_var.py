import argparse
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from cov_tuned_diffusion.utils.path_config import FIGURES_DIR, get_gmm2_params_checkpoint_path


def load_params(checkpoint_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load nus and time steps from a parameter checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    nus = checkpoint["nus"].detach().cpu().numpy()
    time_steps = checkpoint.get("time_steps")
    if isinstance(time_steps, torch.Tensor):
        time_steps = time_steps.detach().cpu().numpy()
    return nus, time_steps


def main():
    parser = argparse.ArgumentParser(description="Plot covariance parameter trends for GMM2 experiments.")
    parser.add_argument("--input-dim", type=int, default=5, help="Dimensionality of the GMM.")
    parser.add_argument("--num-steps", type=int, default=40, help="Number of diffusion steps.")
    parser.add_argument("--params-index", type=int, default=1, help="Parameter checkpoint index to load.")
    parser.add_argument(
        "--cov-form",
        type=str,
        default="isotropic",
        choices=["isotropic", "diagonal", "full"],
        help="Covariance form of the tuned parameters.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank for full covariance models (ignored for other forms).",
    )
    parser.add_argument(
        "--baseline-t-min",
        type=float,
        default=2e-3,
        help="Minimum time step for the baseline geometric schedule.",
    )
    parser.add_argument(
        "--baseline-t-max",
        type=float,
        default=80.0,
        help="Maximum time step for the baseline geometric schedule.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the figure (defaults to the figures directory).",
    )
    args = parser.parse_args()

    tuned_path = get_gmm2_params_checkpoint_path(
        input_dim=args.input_dim,
        num_steps=args.num_steps,
        params_index=args.params_index,
        cov_form=args.cov_form,
        tune_time_steps=True,
        rank=args.rank if args.cov_form == "full" else None,
    )
    baseline_path = get_gmm2_params_checkpoint_path(
        input_dim=args.input_dim,
        num_steps=args.num_steps,
        params_index=args.params_index,
        cov_form=args.cov_form,
        tune_time_steps=False,
        rank=args.rank if args.cov_form == "full" else None,
    )

    tuned_nus, tuned_steps = load_params(tuned_path)
    baseline_nus, _ = load_params(baseline_path)
    baseline_steps = np.geomspace(args.baseline_t_min, args.baseline_t_max, args.num_steps)

    if tuned_steps is None:
        tuned_steps = baseline_steps

    plt.figure()
    plt.plot(tuned_steps[1:], tuned_nus, "-o", label="tuned time steps")
    plt.plot(baseline_steps[1:], baseline_nus, "-o", label="baseline (geomspace)")
    plt.xlabel("Time step")
    plt.ylabel(r"$\nu$")
    plt.title(f"{args.input_dim}D GMM2 parameter trend ({args.cov_form})")
    plt.grid(True)
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()

    if args.output is None:
        output_dir = FIGURES_DIR / "gmm2" / "var"
        output_dir.mkdir(parents=True, exist_ok=True)
        rank_suffix = f"_rank{args.rank}" if args.rank is not None else ""
        output_path = output_dir / (
            f"{args.input_dim}D_gmm2_{args.cov_form}_params_{args.num_steps}steps_"
            f"{args.params_index}{rank_suffix}_nus_trend.png"
        )
    else:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
