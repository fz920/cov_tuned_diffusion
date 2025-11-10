"""Path configuration helpers for the variance-tuned diffusion project.

This module provides centralized path configuration for the entire project,
allowing for easy adjustment of paths across all scripts.
"""

import os
from pathlib import Path

# Resolve key directories. Allow overriding the storage root with COV_TUNED_BASE.
# PACKAGE_DIR now points to the installed python package (src/cov_tuned_diffusion)
# so that relative resources like configs resolve correctly even after restructuring.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]
# Store checkpoints/etc. at the parent of the repo (e.g., dissertation/), unless overridden.
DEFAULT_BASE = PROJECT_ROOT.parent
BASE_DIR = Path(os.getenv("COV_TUNED_BASE", DEFAULT_BASE))

CONFIG_DIR = PACKAGE_DIR / "configs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DATASET_DIR = CHECKPOINTS_DIR / "dataset"
MODEL_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "model_checkpoints"
PARAMS_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "params_checkpoints"
PARAMS_CHECKPOINTS_LOW_RANK_DIR = CHECKPOINTS_DIR / "params_checkpoints_low_rank"
PARAMS_CHECKPOINTS_MODEL_DIR = CHECKPOINTS_DIR / "params_checkpoints_model"
SAMPLES_DIR = CHECKPOINTS_DIR / "samples"
SAMPLES_LOW_RANK_DIR = CHECKPOINTS_DIR / "samples_low_rank"
FIGURES_DIR = CHECKPOINTS_DIR / "figures"
ESS_CHECKPOINTS_DIR = BASE_DIR / "ess_checkpoints"
ESS_LOG_DIR = BASE_DIR / "ess_log"

# GMM2-specific directories
GMM2_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "gmm2_checkpoints"
GMM2_MODEL_CHECKPOINTS_DIR = GMM2_CHECKPOINTS_DIR / "model_checkpoints"
GMM2_PARAMS_CHECKPOINTS_DIR = GMM2_CHECKPOINTS_DIR / "params_checkpoints"
GMM2_SAMPLE_CHECKPOINTS_DIR = GMM2_CHECKPOINTS_DIR / "sample_checkpoints"
GMM2_ESS_CHECKPOINTS_DIR = GMM2_CHECKPOINTS_DIR / "ess_checkpoints"

# Create directories if they don't exist
for directory in [
    CHECKPOINTS_DIR,
    DATASET_DIR,
    MODEL_CHECKPOINTS_DIR,
    PARAMS_CHECKPOINTS_DIR,
    PARAMS_CHECKPOINTS_LOW_RANK_DIR,
    PARAMS_CHECKPOINTS_MODEL_DIR,
    SAMPLES_DIR,
    SAMPLES_LOW_RANK_DIR,
    FIGURES_DIR,
    ESS_CHECKPOINTS_DIR,
    ESS_LOG_DIR,
    GMM2_CHECKPOINTS_DIR,
    GMM2_MODEL_CHECKPOINTS_DIR,
    GMM2_PARAMS_CHECKPOINTS_DIR,
    GMM2_SAMPLE_CHECKPOINTS_DIR,
    GMM2_ESS_CHECKPOINTS_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# Helper functions to get specific paths
def get_config_path(dataset, net, type="score"):
    """Get the path to a model configuration file."""
    preferred = CONFIG_DIR / f"{dataset}_{net}_{type}_config.yaml"
    if preferred.exists():
        return preferred

    legacy = CONFIG_DIR / f"{dataset}_{net}_config.yaml"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"Config not found for dataset '{dataset}' and net '{net}'. "
        f"Tried {preferred} and {legacy}."
    )

def get_model_checkpoint_path(dataset, net, model_type="score", model_index=0, use_ot=False):
    """Get the path to a model checkpoint."""
    ot_suffix = "_ot" if use_ot else ""
    model_dir = MODEL_CHECKPOINTS_DIR / dataset / model_type
    os.makedirs(model_dir, exist_ok=True)
    return model_dir / f"{net}_{model_type}_{model_index}{ot_suffix}.pth"

def get_params_checkpoint_path(dataset, net, params_index=0, num_steps=100, **kwargs):
    """Get the path to a parameters checkpoint.
    
    Additional kwargs can include:
    - tune_timesteps: bool
    - alpha: float
    - rho: float
    - low_rank: bool
    - model: bool
    - diag: bool
    """
    if kwargs.get("low_rank", False):
        base_dir = PARAMS_CHECKPOINTS_LOW_RANK_DIR / dataset
        suffix = "low_rank"
    elif kwargs.get("model", False):
        base_dir = PARAMS_CHECKPOINTS_MODEL_DIR / dataset
        suffix = "model"
        if kwargs.get("diag", False):
            suffix += "_diag"
    else:
        base_dir = PARAMS_CHECKPOINTS_DIR / dataset
        suffix = ""
        
    os.makedirs(base_dir, exist_ok=True)
    
    if kwargs.get("tune_timesteps", False):
        return base_dir / f"{net}_score_params_{params_index}_{num_steps}steps_tune_timesteps.pth"
    
    if "alpha" in kwargs and "rho" in kwargs:
        return base_dir / f"{net}_score_params_{params_index}_{num_steps}steps_alpha{kwargs['alpha']}_rho{kwargs['rho']}.pth"
    
    if "alpha" in kwargs:
        return base_dir / f"{net}_score_params_{params_index}_{num_steps}steps_alpha{kwargs['alpha']}.pth"
    
    if suffix:
        return base_dir / f"{net}_params_{params_index}_{num_steps}steps_{suffix}.pth"
    
    return base_dir / f"{net}_score_params_{params_index}_{num_steps}steps.pth"

def get_sample_path(dataset, sample_type="forward", net="egnn", num_steps=100, sample_index=0, **kwargs):
    """Get the path to a sample file."""
    if kwargs.get("low_rank", False):
        base_dir = SAMPLES_LOW_RANK_DIR / dataset / sample_type
    else:
        base_dir = SAMPLES_DIR / dataset / sample_type
    
    os.makedirs(base_dir, exist_ok=True)
    
    suffix = "_tune_timesteps" if kwargs.get("tune_timesteps", False) else ""
    return base_dir / f"{num_steps}steps_{sample_index}sample{suffix}.pkl"

def get_dataset_path(dataset, partition="train"):
    """Get the path to a dataset file."""
    if dataset == "dw4":
        return DATASET_DIR / ("dw4_samples.npy" if partition == "train" else "all_data_DW4-1000.npy")
    if dataset == "lj13":
        return DATASET_DIR / ("all_data_LJ13-1000.npy" if partition == "train" else "train_split_LJ13-1000.npy")
    if dataset == "lj55":
        return DATASET_DIR / ("all_data_LJ55-1000.npy" if partition == "train" else "train_split_LJ55-1000-part1.npy")
    if dataset == "aldp":
        return DATASET_DIR / ("aldp_train.h5" if partition == "train" else "test.h5")
    raise ValueError(f"Unknown dataset: {dataset}")

def get_figure_path(figure_type, dataset, net, **kwargs):
    """Get the path to a figure file."""
    base_dir = FIGURES_DIR / figure_type
    os.makedirs(base_dir, exist_ok=True)
    
    if figure_type == "ess":
        return base_dir / f"ess_plot_{dataset}_{net}.pdf"
    elif figure_type == "hist":
        return base_dir / f"hist_{dataset}_{net}.pdf"
    elif figure_type == "var":
        return base_dir / f"var_{dataset}_{net}.pdf"
    else:
        return base_dir / f"{figure_type}_{dataset}_{net}.pdf"

def get_ess_log_path(dataset, model_index, num_steps, params_index, low_rank=False):
    """Get the path to an ESS log file."""
    base_dir = ESS_LOG_DIR / dataset
    os.makedirs(base_dir, exist_ok=True)
    
    suffix = "low_rank_" if low_rank else ""
    return base_dir / f"forward_ess_{suffix}{model_index}model_{num_steps}steps_{params_index}.txt"


def get_gmm2_model_checkpoint_path(input_dim, n_layers=7, hidden_size=512):
    """Path to a trained GMM2 score model checkpoint."""
    os.makedirs(GMM2_MODEL_CHECKPOINTS_DIR, exist_ok=True)
    filename = f"{input_dim}D_gmm2_score_ckpt_{n_layers}layers_{hidden_size}hidden_size.pth"
    return GMM2_MODEL_CHECKPOINTS_DIR / filename


def get_gmm2_params_checkpoint_path(
    input_dim,
    num_steps,
    params_index,
    cov_form,
    tune_time_steps=False,
    rank=None,
):
    """Path to tuned covariance parameters for the GMM2 experiments."""
    os.makedirs(GMM2_PARAMS_CHECKPOINTS_DIR, exist_ok=True)
    time_suffix = "True" if tune_time_steps else "False"
    filename = (
        f"{input_dim}D_gmm2_score_params_{num_steps}steps_{params_index}_{cov_form}"
        f"_with_time_steps{time_suffix}"
    )
    if cov_form == "full" and rank is not None:
        filename += f"_rank{rank}"
    filename += ".pth"
    return GMM2_PARAMS_CHECKPOINTS_DIR / filename


def get_gmm2_sample_path(
    input_dim,
    num_steps,
    sample_index,
    cov_form,
    direction="backward",
    rank=None,
):
    """Path to stored GMM2 samples (forward/backward)."""
    direction = direction.lower()
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")
    base_dir = GMM2_SAMPLE_CHECKPOINTS_DIR / direction
    os.makedirs(base_dir, exist_ok=True)
    filename = (
        f"{input_dim}D_gmm2_{direction}_{num_steps}steps_{sample_index}sample_{cov_form}"
    )
    if cov_form == "full" and rank is not None:
        filename += f"_rank{rank}"
    filename += ".pkl"
    return base_dir / filename


def get_gmm2_ess_summary_path(input_dim, num_steps, params_indices, rank=None):
    """Path to cached ESS summaries for GMM2 experiments."""
    os.makedirs(GMM2_ESS_CHECKPOINTS_DIR, exist_ok=True)

    def _to_string(value):
        if isinstance(value, (list, tuple)):
            return "_".join(map(str, value))
        return str(value)

    steps_str = _to_string(num_steps)
    params_str = _to_string(params_indices)
    rank_suffix = f"_rank{rank}" if rank is not None else ""
    filename = f"ess_results_{input_dim}D_{steps_str}_{params_str}{rank_suffix}.pth"
    return GMM2_ESS_CHECKPOINTS_DIR / filename
