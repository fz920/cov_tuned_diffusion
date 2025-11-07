
import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Optional profiler (enabled via --measure_flops)
try:
    from torch.profiler import ProfilerActivity, profile, record_function
    _HAS_PROF = True
except Exception:
    _HAS_PROF = False

# Project-specific imports
from cov_tuned_diffusion import (
    ScoreNet,
    compute_forward_ess,
    compute_reverse_ess,
    load_dataset,
    load_target_dist,
)

from cov_tuned_diffusion.utils.path_config import (
    get_config_path,
    get_model_checkpoint_path,
    get_params_checkpoint_path,
    get_sample_path,
    SAMPLES_DIR,
)


# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_covariance_params(args, cov_form: str, idx: int, num_steps: int, diag: bool = False, tune_time_steps: bool = False) -> Dict[str, Any]:
    """
    Load covariance/tuning checkpoint produced by the tuning script(s).

    cov_form: 'isotropic' | 'model' | 'full' | 'diag' | 'ddpm' (for baseline; returns empty)
    Returns a dict-like checkpoint with keys among: 'cov_mat_all', 'time_steps', 'nus', etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if cov_form == "ddpm":
        return {"cov_form": "ddpm"}  # baseline; no tuned covariances

    # Primary structured path
    if cov_form in {"isotropic", "model"}:
        ckpt_path = get_params_checkpoint_path(
            args.dataset, args.net, idx, num_steps, tune_timesteps=tune_time_steps, diag=diag
        )
    elif cov_form in {"full", "diag"}:
        # Allow both old + new naming
        # Try standard path first; if missing, try a few common variants under dataset dir
        base = get_params_checkpoint_path(args.dataset, args.net, idx, num_steps, tune_timesteps=tune_time_steps)
        p = Path(base)
        dataset_dir = p.parent
        candidates = [
            p,  # default
            dataset_dir / f"{p.stem}_{cov_form}.pth",
            dataset_dir / f"{p.stem}_{cov_form}cov_form.pth",
            dataset_dir / f"params_m{idx}_p{args.params_index}_{cov_form}.pth",
        ]
        for cand in candidates:
            if cand.exists():
                ckpt_path = cand
                break
        else:
            # Fall back to generic path with diag flag
            ckpt_path = get_params_checkpoint_path(
                args.dataset, args.net, idx, num_steps, tune_timesteps=tune_time_steps, diag=(cov_form == "diag")
            )
    else:
        raise ValueError(f"Invalid covariance form: {cov_form}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint for cov_form={cov_form} not found: {ckpt_path}")

    return torch.load(ckpt_path, map_location=device)


def _load_model_and_target(args) -> Tuple[ScoreNet, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Score model
    cfg_path = get_config_path(args.dataset, args.net)
    with open(cfg_path, "r") as f:
        score_cfg = yaml.safe_load(f)
    model_ckpt = get_model_checkpoint_path(args.dataset, args.net, "score", args.model_index)
    score_model = ScoreNet(dataset=args.dataset, device=str(device), model_config=score_cfg, net=args.net).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    score_model.load_state_dict(ckpt["model_state_dict"])
    score_model.eval()
    score_model.requires_grad_(False)

    # Target
    true_target = load_target_dist(args.dataset, device=device)

    return score_model, true_target


def _prepare_save_paths(args) -> Tuple[Path, Path, Path]:
    # Resolve base save path
    if args.save_path is None:
        base = Path(SAMPLES_DIR) / args.dataset
    else:
        base = Path(args.save_path)

    forward_dir  = base / "forward"
    backward_dir = base / "backward"
    _ensure_dir(forward_dir)
    _ensure_dir(backward_dir)

    # Named pickle files via project helper
    f_pkl = Path(get_sample_path(args.dataset, "forward", args.net, args.num_steps, args.sample_index, tune_timesteps=args.tune_time_steps))
    b_pkl = Path(get_sample_path(args.dataset, "backward", args.net, args.num_steps, args.sample_index, tune_timesteps=args.tune_time_steps))

    # Log file
    log_dir = base / "logs"
    _ensure_dir(log_dir)
    log_file = log_dir / f"sample_m{args.model_index}_p{args.params_index}_s{args.sample_index}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=str(log_file),
        filemode="a",
    )
    logging.info(f"Logging to {log_file}")
    return f_pkl, b_pkl, log_file


def _maybe_resume(pkl_path: Path) -> Optional[dict]:
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to resume from {pkl_path}: {e}")
    return None


def _get_x0_batch(args, score_model: ScoreNet, true_target) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce a batch x0 and its log_prob under the target.
    Tries dataset loader first; falls back to Gaussian if needed.
    """
    device = next(score_model.parameters()).device
    m, n = score_model._n_particles, score_model._n_dimension

    try:
        data = load_dataset(args.dataset, partition="test", device=device)
        if hasattr(data, "__iter__"):
            batch = next(iter(data))
            x0 = batch.to(device)
        elif hasattr(data, "dataset"):
            x0 = data.dataset[0][0].to(device)  # heuristic
            x0 = x0.unsqueeze(0).repeat(args.num_samples, 1, 1)
        else:
            raise RuntimeError("Dataset object not iterable.")
    except Exception:
        x0 = torch.randn(args.num_samples, m, n, device=device)

    x0 = x0[:args.num_samples]
    log_prob_x0 = true_target.log_prob(x0 / score_model.normalizing_constant)
    return x0, log_prob_x0


def _call_forward_weights(score_model: ScoreNet, sampler: str, x0, log_prob_x0, num_steps, time_steps=None, nus=None, cov_mat_all=None, args=None):
    """
    Compute forward log-weights with either DDPM or DDIM path.
    Returns: log_w (Tensor, shape [M])
    """
    if sampler == "ddpm":
        if cov_mat_all is None:
            # scalar proposal
            if hasattr(score_model, "est_forward_ess"):
                _, _, log_w = score_model.est_forward_ess(x0, log_prob_x0, num_steps, time_steps=time_steps, nus=nus, progress_bar=True)
            else:
                _, _, log_w = score_model.estimate_forward_ess(x0, log_prob_x0, num_steps, time_steps=time_steps, proposal="scalar", nus=nus, progress_bar=True)
        else:
            if hasattr(score_model, "forward_ess_low_rank"):
                _, _, log_w = score_model.forward_ess_low_rank(x0, log_prob_x0, num_steps, time_steps=time_steps, cov_mat_all=cov_mat_all, progress_bar=True)
            else:
                _, _, log_w = score_model.estimate_forward_ess(x0, log_prob_x0, num_steps, time_steps=time_steps, proposal="fullcov", cov_mats=cov_mat_all, progress_bar=True)
        return log_w

    elif sampler == "ddim":
        # DDIM forward weights (deterministic flow)
        if hasattr(score_model, "forward_ess_ddim"):
            kwargs = dict(divergence_mode=args.divergence_mode, num_trace_probes=args.num_trace_probes,
                          probe_kind=args.probe_kind, reuse_probes_across_time=not args.no_reuse_probes)
            ess, log_w = score_model.forward_ess_ddim(x0, log_prob_x0, num_steps, time_steps=time_steps, progress_bar=True, **kwargs)
            return log_w
        else:
            raise RuntimeError("ScoreNet does not implement forward_ess_ddim.")
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def _call_backward_sampling(score_model: ScoreNet, sampler: str, true_target, num_steps, num_samples, time_steps=None, nus=None, cov_mat_all=None, args=None):
    """
    Run the reverse sampler and return (samples, log_w) for backward IS.
    """
    if sampler == "ddpm":
        if cov_mat_all is None:
            if hasattr(score_model, "ddpm_sampler"):
                x, _, _, log_w = score_model.ddpm_sampler(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, nus=nus, progress_bar=True)
            else:
                x, _, _, log_w = score_model.sample_ddpm(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, proposal="scalar", nus=nus, progress_bar=True)
        else:
            if hasattr(score_model, "ddpm_sampler_low_rank"):
                x, _, _, log_w = score_model.ddpm_sampler_low_rank(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, cov_mat_all=cov_mat_all, progress_bar=True)
            else:
                x, _, _, log_w = score_model.sample_ddpm(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, proposal="fullcov", cov_mats=cov_mat_all, progress_bar=True)
        return x, log_w

    elif sampler == "ddim":
        if hasattr(score_model, "ddim_sampler"):
            kwargs = dict(divergence_mode=args.divergence_mode, num_trace_probes=args.num_trace_probes,
                          probe_kind=args.probe_kind, reuse_probes_across_time=not args.no_reuse_probes)
            x, _, _, log_w = score_model.ddim_sampler(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, progress_bar=True, **kwargs)
            return x, log_w
        else:
            # New API name
            if hasattr(score_model, "sample_ddim"):
                x, _, _, log_w = score_model.sample_ddim(num_steps, true_target, num_samples=num_samples, time_steps=time_steps, progress_bar=True)
                return x, log_w
            raise RuntimeError("ScoreNet does not implement DDIM sampler.")
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


# ----------------------------
# CLI + main
# ----------------------------

def make_argparser():
    p = argparse.ArgumentParser(description="Unified sampling driver (DDPM/DDIM).")
    # core
    p.add_argument("--dataset", type=str, default="aldp")
    p.add_argument("--net", type=str, default="egnn")
    p.add_argument("--params_index", type=int, default=0)
    p.add_argument("--model_index", type=int, default=0)
    p.add_argument("--sample_index", type=int, default=0)

    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--sample_num_times", type=int, default=1)

    p.add_argument("--cov_forms", type=str, nargs="+", default=["ddpm"],
                   choices=["ddpm","isotropic","model","full","diag"],
                   help="Covariance form(s) to evaluate. 'ddpm' = baseline (no tuned covariances).")

    p.add_argument("--sampler", type=str, choices=["ddpm","ddim"], default="ddpm")

    p.add_argument("--save_path", type=str, default=None, help="Base path to save samples and weights.")
    p.add_argument("--continue_sampling", action="store_true", help="Append to existing pickles if present.")
    p.add_argument("--overwrite", action="store_true", help="Ignore any existing pickle and start fresh.")
    p.add_argument("--diag", action="store_true", help="Use diagonal covariance for 'model' form (if supported by ckpt naming).")
    p.add_argument("--tune_time_steps", action="store_true", default=False, help="Use tuned time steps from parameter ckpt (if available).")
    p.add_argument("--cpu", action="store_true", help="Force CPU")

    # DDIM divergence controls
    p.add_argument("--divergence_mode", type=str, choices=["exact","hutchinson_vjp","hutchinson_jvp"], default="exact")
    p.add_argument("--num_trace_probes", type=int, default=1)
    p.add_argument("--probe_kind", type=str, choices=["rademacher","gaussian"], default="rademacher")
    p.add_argument("--no_reuse_probes", action="store_true", help="If set, do not reuse Hutchinson probes across time.")

    # profiling
    p.add_argument("--measure_flops", action="store_true", default=False)
    return p


def main():
    args = make_argparser().parse_args()

    # seeds for reproducibility
    torch.manual_seed(args.sample_index)
    np.random.seed(args.sample_index)

    # save paths & logging
    forward_pkl_path, backward_pkl_path, log_file = _prepare_save_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # (optional) resume
    forward_results = {}   # cov_form -> dict(log_weights=[...])
    backward_results = {}  # cov_form -> dict(log_weights=[...], samples=[...])
    if args.continue_sampling and not args.overwrite:
        prev_f = _maybe_resume(forward_pkl_path)
        prev_b = _maybe_resume(backward_pkl_path)
        if prev_f: forward_results.update(prev_f)
        if prev_b: backward_results.update(prev_b)

    # Load model and target
    score_model, true_target_dist = _load_model_and_target(args)

    # get x0 batch and log_prob
    x0, log_prob_x0 = _get_x0_batch(args, score_model, true_target_dist)

    # Profiler context manager
    def maybe_profile():
        if args.measure_flops and _HAS_PROF:
            return profile(
                activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else []),
                record_shapes=False, with_stack=False, profile_memory=True,
            )
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, *exc): return False
        return Dummy()

    # Iterate over requested covariance forms
    for cov_form in args.cov_forms:
        logging.info(f"=== Sampling with cov_form={cov_form} | sampler={args.sampler} ===")

        # Prepare result slots
        if cov_form not in forward_results:
            forward_results[cov_form] = {"log_weights": []}
        if cov_form not in backward_results:
            backward_results[cov_form] = {"log_weights": [], "samples": []}

        total_forward_ess = 0.0
        total_backward_ess = 0.0
        total_samples = 0

        # Load tuning ckpt (if any)
        ckpt = _load_covariance_params(args, cov_form, args.params_index, args.num_steps, diag=args.diag, tune_time_steps=args.tune_time_steps) \
               if cov_form != "ddpm" else {"cov_form":"ddpm"}

        # Extract time_steps / nus / cov_mat_all if available
        time_steps = ckpt.get("time_steps", None)
        nus = ckpt.get("nus", None)
        if isinstance(nus, torch.Tensor): nus = nus.to(device)

        cov_mat_all = ckpt.get("cov_mat_all", None)
        if isinstance(cov_mat_all, torch.Tensor): cov_mat_all = cov_mat_all.to(device)
        if cov_form == "model" and cov_mat_all is None:
            logging.warning("cov_form='model' checkpoint lacks 'cov_mat_all'; falling back to scalar proposal for this run.")

        # Sample multiple batches
        for n in tqdm(range(args.sample_num_times), desc=f"{cov_form}"):
            with maybe_profile():
                # Forward log-weights
                log_w_f = _call_forward_weights(
                    score_model, args.sampler,
                    x0=x0, log_prob_x0=log_prob_x0,
                    num_steps=args.num_steps,
                    time_steps=time_steps, nus=nus, cov_mat_all=cov_mat_all, args=args
                )

                # Reverse sampling & weights
                samples, log_w_b = _call_backward_sampling(
                    score_model, args.sampler, true_target_dist,
                    num_steps=args.num_steps, num_samples=args.num_samples,
                    time_steps=time_steps, nus=nus, cov_mat_all=cov_mat_all, args=args
                )

            # Append to results
            forward_results[cov_form]["log_weights"].append(log_w_f.detach().cpu())
            backward_results[cov_form]["log_weights"].append(log_w_b.detach().cpu())
            backward_results[cov_form]["samples"].append(samples.detach().cpu())

            # ESS per batch (percentages)
            ess_f = compute_forward_ess(log_w_f).item()
            ess_b = compute_reverse_ess(log_w_b).item()
            total_forward_ess += ess_f
            total_backward_ess += ess_b
            total_samples += args.num_samples

            logging.info(f"Batch {n} - {cov_form} - Forward ESS: {100.0*ess_f/args.num_samples:.2f}%")
            logging.info(f"Batch {n} - {cov_form} - Backward ESS: {100.0*ess_b/args.num_samples:.2f}%")

            # Save after each batch
            with open(forward_pkl_path, "wb") as f:
                pickle.dump(forward_results, f)
            with open(backward_pkl_path, "wb") as f:
                pickle.dump(backward_results, f)

        # Final stats
        forward_ess_percent  = 100.0 * total_forward_ess  / max(1, total_samples)
        backward_ess_percent = 100.0 * total_backward_ess / max(1, total_samples)
        logging.info(f"Final {cov_form} - Forward ESS:  {forward_ess_percent:.2f}% ({total_forward_ess:.1f}/{total_samples})")
        logging.info(f"Final {cov_form} - Backward ESS: {backward_ess_percent:.2f}% ({total_backward_ess:.1f}/{total_samples})")

    logging.info(f"Sampling complete. Forward results saved to:  {forward_pkl_path}")
    logging.info(f"Sampling complete. Backward results saved to: {backward_pkl_path}")


if __name__ == "__main__":
    main()
