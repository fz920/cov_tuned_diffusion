#!/usr/bin/env python
"""
Plot nus vs timesteps from model checkpoints.

This script generates visualizations of nu parameter values against timesteps.

Usage:
    python plot_var.py [options]

Options:
    --dataset STR         Dataset name (default: dw4)
    --net STR             Network architecture (default: egnn)
    --num_steps INT       Number of steps (default: 20)
    --params_index INT    Parameters index (default: 0)
    --model_index INT     Model index (default: 0)
    --base_path STR       Base path for checkpoint files
    --output_dir STR      Custom output directory for saving figures

Example:
    python plot_var.py --dataset dw5 --net egnn --num_steps 50 --params_index 1
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse
from pathlib import Path

from cov_tuned_diffusion.utils.path_config import (
    get_config_path,
    get_model_checkpoint_path,
    get_params_checkpoint_path,
    get_sample_path,
    get_figure_path,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
)


def load_checkpoint(checkpoint_path, device):
    """Load parameters from checkpoint file."""
    try:
        params = torch.load(checkpoint_path, map_location=device)
        nus = params['nus'].detach().cpu().numpy()
        time_steps = params['time_steps'].detach().cpu().numpy()
        return nus, time_steps
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise


def plot_nus_vs_timesteps(time_steps, nus, dataset='dw4', net='egnn', 
                          params_index=0, num_steps=20, output_dir=None):
    """Plot nus values against time steps with publication-quality formatting."""
    try:
        # Try to use LaTeX for high-quality text rendering
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'text.usetex': True,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
        })
    except Exception as e:
        print(f"Warning: Could not set LaTeX rendering: {e}")
        # Fallback to standard fonts
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
        })
    
    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    
    # Plot with scientific color scheme
    ax.plot(time_steps[1:], nus, '-o', label='Tuned Variance', alpha=0.8, 
            color='#1f77b4', markerfacecolor='white', markeredgewidth=1.5)
    ax.plot(time_steps[1:], np.ones_like(nus), '--', label='DDPM Variance', 
            color='#d62728', linewidth=2)
    
    # Set labels with appropriate formatting
    try:
        ax.set_xlabel('Time Step $t$', fontsize=14)
        ax.set_ylabel(r'$\eta_n$', fontsize=14)
    except Exception:
        # Fallback for non-LaTeX environment
        ax.set_xlabel('Time Step t', fontsize=14)
        ax.set_ylabel(r'$$\eta_n$$', fontsize=14)
    
    # Configure title
    dataset_title = {
        'dw4': 'DW-4',
        'lj13': 'LJ-13',
        'lj55': 'LJ-55',
        'aldp': 'Alanine Dipeptide'
    }
    ax.set_title(f'VT-DIS Optimal Parameters ({dataset_title[dataset]})', fontsize=16, fontweight='bold')
    
    # Configure axes
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Configure ticks
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
    ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', length=3)
    
    # Add legend with professional formatting
    legend = ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best', fontsize=12)
    legend.get_frame().set_linewidth(1.0)
    
    # Adjust layout for optimal spacing
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    if output_dir is None:
        save_dir = Path(FIGURES_DIR / 'var_figures')
    else:
        save_dir = Path(output_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with high resolution for publication
    try:
        save_path = save_dir / f'{net}_score_params_{params_index}_{num_steps}steps_nus_{dataset}.pdf'
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        
        # Also save as PNG for quick viewing
        png_path = save_dir / f'{net}_score_params_{params_index}_{num_steps}steps_nus_{dataset}.png'
        plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        
        print(f"Figures saved to {save_path} and {png_path}")
    except Exception as e:
        print(f"Error saving figures: {e}")
        # Try with a simpler format if PDF fails
        png_path = save_dir / f'{net}_score_params_{params_index}_{num_steps}steps_nus_{dataset}.png'
        plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"Fallback figure saved to {png_path}")

    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot nus vs timesteps from model checkpoints')
    
    # Add arguments
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of steps (default: 20)')
    parser.add_argument('--params_index', type=int, default=0,
                        help='Parameters index (default: 0)')
    parser.add_argument('--model_index', type=int, default=0,
                        help='Model index (default: 0)')
    parser.add_argument('--base_path', type=str, 
                        default='/rds/user/fz287/hpc-work/dissertation',
                        help='Base path for checkpoint files')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Set up save directory
    if args.save_dir is None:
        args.save_dir = str(FIGURES_DIR / "var_figures")
    
    return args


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define checkpoint paths
    params_checkpoint = f'{args.base_path}/checkpoints/params_checkpoints/{args.dataset}/egnn_score_params_0_{args.num_steps}steps.pth'
    
    # Load parameters
    nus, time_steps = load_checkpoint(params_checkpoint, device)
    
    # Plot results
    plot_nus_vs_timesteps(
        time_steps, 
        nus, 
        dataset=args.dataset,
        net=args.net,
        params_index=args.params_index,
        num_steps=args.num_steps,
        output_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
