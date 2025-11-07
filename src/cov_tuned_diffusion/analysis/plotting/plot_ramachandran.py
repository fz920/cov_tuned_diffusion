import argparse
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import torch
from openmmtools import testsystems

from cov_tuned_diffusion import load_target_dist, load_dataset
from cov_tuned_diffusion.utils.path_config import CHECKPOINTS_DIR, FIGURES_DIR

def detect_chirality_batch(samples: torch.Tensor,
                           center_idx: int,
                           priority_neighs: list[int],
                           eps: float = 1e-5
                          ) -> torch.Tensor:
    """
    Returns +1 for R, -1 for S, and 0 for |triple| < eps (ambiguous).
    """
    A, B, C, D = priority_neighs
    v1 = samples[:, A, :] - samples[:, center_idx, :]
    v2 = samples[:, B, :] - samples[:, center_idx, :]
    v3 = samples[:, C, :] - samples[:, center_idx, :]
    triple = (v1 * torch.cross(v2, v3, dim=1)).sum(dim=1)

    # raw sign
    signs = torch.sign(triple)

    # apply threshold
    ambiguous = torch.abs(triple) < eps
    signs[ambiguous] = 0
    return signs


def postprocess_to_one_chirality(samples: torch.Tensor,
                                 center_idx: int,
                                 priority_neighs: list[int],
                                 keep: str = 'S',
                                 reflect_axis: int = 0,
                                 eps: float = 1e-5
                                ) -> torch.Tensor:
    """
    Same as before, but uses eps to avoid flipping near-planar frames.
    """
    signs = detect_chirality_batch(samples, center_idx, priority_neighs, eps=eps)

    # 1) Turn your batch into an mdtraj.Trajectory so we can compute φ/ψ on the *raw* samples
    topo = mdtraj.Topology.from_openmm(testsystems.AlanineDipeptideVacuum().topology)
    xyz = samples.detach().cpu().numpy()        # (B,22,3)
    traj = mdtraj.Trajectory(xyz, topo)

    phis = mdtraj.compute_phi(traj)[1].reshape(-1)
    psis = mdtraj.compute_psi(traj)[1].reshape(-1)

    # 2) Flatten `signs` to match the (B,) → (B*frames per sample? No, here it's one center per frame)
    signs_np = signs.detach().cpu().numpy()

    # 3) Plot / count each group separately
    for grp, name in [(signs_np>0,'R'),
                    (signs_np<0,'S'),
                    (signs_np==0,'ambiguous')]:
        print(f"{name:>10s} count = {grp.sum():4d}   φ̄ = {phis[grp].mean():6.2f}   ψ̄ = {psis[grp].mean():6.2f}")
        # And you can even do a little 2-D scatter to see where they lie:
        plt.figure(figsize=(4,4))
        plt.scatter(phis[grp], psis[grp], s=2)
        plt.title(name); plt.xlim(-np.pi, np.pi); plt.ylim(-np.pi, np.pi)
        plt.savefig('test.png', format='png')
        plt.show()

    # flip only definite “wrong” enantiomer
    if keep.upper() == 'S':
        mask_flip = signs > 0   # only frames with true R
    else:
        mask_flip = signs < 0   # only frames with true S

    processed = samples.clone()
    processed[mask_flip, :, reflect_axis] *= -1.0
    return processed

def generate_ramachandran_plot(traj, weights=None, title="Ramachandran Plot", cmap='viridis'):
    """
    Generate a Ramachandran plot for the given trajectory.
    
    Parameters:
    -----------
    traj : mdtraj.Trajectory
        The trajectory for which to generate the Ramachandran plot
    weights : torch.Tensor or None
        Optional weights for the samples
    title : str
        Title for the plot
    cmap : str
        Colormap to use for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the Ramachandran plot
    """
    # Compute phi and psi angles
    phi = mdtraj.compute_phi(traj)[1].reshape(-1)
    psi = mdtraj.compute_psi(traj)[1].reshape(-1)
    
    # Filter out NaN values
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    not_nan = np.logical_not(is_nan)
    phi = phi[not_nan]
    psi = psi[not_nan]

    if weights is not None:
        # Adjust weights to match the filtered angles
        weights = weights.cpu().numpy()
        weights = weights[not_nan]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if weights is None:
        h = ax.hist2d(phi, psi, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
                      norm=mpl.colors.LogNorm(vmin=0.0001, vmax=1.0), 
                      density=True, cmap=cmap)
    else:
        h = ax.hist2d(phi, psi, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
                      weights=weights, norm=mpl.colors.LogNorm(vmin=0.0001, vmax=1.0), 
                      density=True, cmap=cmap)
    
    plt.colorbar(h[3], ax=ax)
    ax.set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    ax.set_xticklabels(['-π','-π/2','0','π/2','π'], fontsize=16)
    ax.set_yticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    ax.set_yticklabels(['-π','-π/2','0','π/2','π'], fontsize=16)
    ax.set_xlabel(r'$\phi$', fontsize=24)
    ax.set_ylabel(r'$\psi$', fontsize=24)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser()
    # Sampling settings
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                      help='List of sample indices to combine (uses --sample_index if not provided)')
    parser.add_argument('--cov_form', type=str, default='ddpm',
                      choices=['ddpm', 'isotropic', 'full'],
                      help='Covariance form to use for plotting')

    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--fix_chirality', action='store_true')
    args = parser.parse_args()

    # If sample_indices is not provided, use sample_index
    if args.sample_indices is None:
        args.sample_indices = [args.sample_index]

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define samples base path
    samples_base_path = CHECKPOINTS_DIR / "samples" / args.dataset

    # Get true target distribution and true samples
    # true_target_dist = load_target_dist(args.dataset)
    true_samples = load_dataset(args.dataset, partition='test')
    true_samples = true_samples[:args.num_samples].to(device)

    # # test
    # import mdtraj as md

    # # build an mdtraj.Trajectory just to inspect the topology:
    # topo = md.Topology.from_openmm(testsystems.AlanineDipeptideVacuum().topology)
    # xyz = torch.zeros((1,22,3)).numpy()     # dummy coords
    # traj = md.Trajectory(xyz, topo)

    # print("All atoms:")
    # for atom in traj.topology.atoms:
    #     print(atom.index, atom.residue.name, atom.name, atom.element)

    # # now find your Cα (residue 1 for the alanine) by name & residue index:
    # cas = [a.index for a in traj.topology.atoms
    #     if a.name=='CA' and a.residue.index==1]
    # print("CA indices in residue 1:", cas)

    # # list the bonded neighbors of that CA:
    # ca = cas[0]
    # neighs = []
    # for bond in traj.topology.bonds:
    #     if bond.atom1.index==ca:
    #         neighs.append(bond.atom2.index)
    #     elif bond.atom2.index==ca:
    #         neighs.append(bond.atom1.index)

    # print("Neighbors of CA:", neighs)
    # for n in neighs:
    #     a = traj.topology.atom(n)
    #     print(f"  → {n}: {a.residue.name} {a.name} ({a.element})")

    # breakpoint()

    # # load the alanine dipeptide target to post-process the data 
    # true_target_dist = AldpPotentialCart(
    #     data_path=str(Path(__file__).parent.parent / 'checkpoints/dataset/test.h5'),
    #     temperature=300,
    #     energy_cut=1e8,
    #     energy_max=1e20,
    #     n_threads=64,
    #     shift_dih=False,
    #     transform='internal',
    #     env='implicit'
    # )

    # Load and combine samples from multiple files
    all_samples = []
    all_log_weights = []

    for sample_idx in args.sample_indices:
        backward_pkl_path = samples_base_path / "backward" / f"{args.num_steps}steps_{sample_idx}sample.pkl"
        
        # Load samples and weights
        with open(backward_pkl_path, 'rb') as f:
            backward_results = pickle.load(f)
        
        # Extract samples and weights for the selected covariance form
        # Concatenate all batches of samples and weights from this file
        file_samples = torch.cat(backward_results[args.cov_form]['samples'], dim=0)
        file_log_weights = torch.cat(backward_results[args.cov_form]['log_weights'], dim=0)
        
        all_samples.append(file_samples)
        all_log_weights.append(file_log_weights)
    
    # Concatenate all samples and weights from all files
    gen_samples = torch.cat(all_samples, dim=0).to(device)
    log_w = torch.cat(all_log_weights, dim=0).to(device)

    # Limit to requested number of samples if needed
    if len(gen_samples) > args.num_samples:
        print(f"Limiting combined samples from {len(gen_samples)} to {args.num_samples}")
        gen_samples = gen_samples[:args.num_samples]
        log_w = log_w[:args.num_samples]

    if args.fix_chirality:
        center_idx     = 8                  # CA atom index
        priority_neighs = [6, 14, 10, 9]     # N > C=O > CB > H

        # Convert all "D" frames into "L" by flipping the x-axis:
        processed = postprocess_to_one_chirality(
            gen_samples,
            center_idx=center_idx,
            priority_neighs=priority_neighs,
            keep='R',         # keep S = L-alanine as is
            reflect_axis=0    # flip x
        )
        gen_samples = processed

        # gen_samples = true_target_dist.reflect_d_to_l_cartesian(gen_samples.view(-1, 66).detach().cpu())
        # gen_samples = gen_samples.reshape(-1, 22, 3).cuda()

    # Normalize the importance weights

    w = torch.exp(log_w - log_w.max())
    w = w / w.sum()

    print(f"Reverse ESS (%) for {args.cov_form}: {1 / torch.sum(w ** 2) / len(gen_samples) * 100:.2f}%")
    print(f"Total samples: {len(gen_samples)}")

    # Create figures directory if it doesn't exist
    figures_dir = FIGURES_DIR / 'ramachandran'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to cartesian coordinates if they're not already
    true_cartesian = true_samples
    gen_cartesian = gen_samples
    
    # Create mdtraj trajectories
    aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    
    # Reshape samples for mdtraj (num_samples, n_atoms, 3)
    true_reshaped = true_cartesian.cpu().numpy().reshape(-1, 22, 3)
    gen_reshaped = gen_cartesian.cpu().numpy().reshape(-1, 22, 3)
    
    # Create trajectories
    true_traj = mdtraj.Trajectory(true_reshaped, topology)
    gen_traj = mdtraj.Trajectory(gen_reshaped, topology)

    # Generate a unique suffix for filenames based on sample indices
    if len(args.sample_indices) > 1:
        indices_str = f"indices_{min(args.sample_indices)}-{max(args.sample_indices)}"
    else:
        indices_str = f"index_{args.sample_indices[0]}"
    
    # 1. Plot for true samples
    print("Generating Ramachandran plot for true samples...")
    fig_true = generate_ramachandran_plot(true_traj, title="Ground Truth Samples", cmap='Blues')
    fig_true.savefig(figures_dir / 'true_ramachandran.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_true)
    
    # 2. Plot for generated samples (unweighted)
    print("Generating Ramachandran plot for unweighted samples...")
    fig_gen = generate_ramachandran_plot(gen_traj, title="Generated Samples (Unweighted)", cmap='Greens')
    fig_gen.savefig(figures_dir / f'gen_ramachandran_{args.cov_form}_{args.num_steps}steps_{indices_str}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_gen)
    
    # 3. Plot for reweighted samples
    print("Generating Ramachandran plot for reweighted samples...")
    fig_reweighted = generate_ramachandran_plot(gen_traj, weights=w, title="Generated Samples (Reweighted)", cmap='Reds')
    fig_reweighted.savefig(figures_dir / f'reweighted_ramachandran_{args.cov_form}_{args.num_steps}steps_{indices_str}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig_reweighted)
    
    # Combined plot using subplots for comparison
    print("Generating combined Ramachandran plot...")
    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    
    # Plot true samples
    phi_true = mdtraj.compute_phi(true_traj)[1].reshape(-1)
    psi_true = mdtraj.compute_psi(true_traj)[1].reshape(-1)
    mask_true = ~np.logical_or(np.isnan(phi_true), np.isnan(psi_true))
    phi_true, psi_true = phi_true[mask_true], psi_true[mask_true]
    
    # Plot generated samples
    phi_gen = mdtraj.compute_phi(gen_traj)[1].reshape(-1)
    psi_gen = mdtraj.compute_psi(gen_traj)[1].reshape(-1)
    mask_gen = ~np.logical_or(np.isnan(phi_gen), np.isnan(psi_gen))
    phi_gen, psi_gen = phi_gen[mask_gen], psi_gen[mask_gen]
    w_np = w.cpu().numpy()[mask_gen]
    
    # Use the same normalization for all plots
    norm = mpl.colors.LogNorm(vmin=0.0001, vmax=1.0)
    
    # Ground truth plot
    h1 = axes[0].hist2d(phi_true, psi_true, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
                     norm=norm, density=True)
    axes[0].set_title("Ground Truth", fontsize=24)
    axes[0].set_xlabel(r'$\phi$', fontsize=24)
    axes[0].set_ylabel(r'$\psi$', fontsize=24)
    axes[0].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[0].set_xticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    axes[0].set_yticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[0].set_yticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    
    # Unweighted generated plot
    h2 = axes[1].hist2d(phi_gen, psi_gen, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
                     norm=norm, density=True)
    axes[1].set_title("Generated (Unweighted)", fontsize=24)
    axes[1].set_xlabel(r'$\phi$', fontsize=24)
    axes[1].set_ylabel(r'$\psi$', fontsize=24)
    axes[1].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[1].set_xticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    axes[1].set_yticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[1].set_yticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    
    # Reweighted generated plot
    # Reweight the samples by creating a new dataset with repetitions based on weights
    # First normalize weights to sum to the number of samples we want
    # w_normalized = w_np * (len(phi_gen) / w_np.sum())
    
    # Create reweighted arrays by repeating points according to their weights
    probabilities = w_np / w_np.sum()
    indices = np.random.choice(len(phi_gen), size=len(phi_gen), replace=True, p=probabilities)
    phi_reweighted = phi_gen[indices]
    psi_reweighted = psi_gen[indices]
    
    # Plot the reweighted data without using weights parameter
    h3 = axes[2].hist2d(phi_reweighted, psi_reweighted, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
                     norm=norm, density=True)
    axes[2].set_title("Generated (Reweighted)", fontsize=24)
    axes[2].set_xlabel(r'$\phi$', fontsize=24)
    axes[2].set_ylabel(r'$\psi$', fontsize=24)
    axes[2].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[2].set_xticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    axes[2].set_yticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)))
    axes[2].set_yticklabels(['-π','-π/2','0','π/2','π'], fontsize=18)
    
    # Add a common colorbar
    cbar = fig.colorbar(h3[3], ax=axes, orientation='vertical', pad=0.01, fraction=0.046, shrink=0.8)
    cbar.set_label('Normalized Density (log scale)', fontsize=20)
    
    # plt.tight_layout()
    plt.savefig(figures_dir / f'combined_ramachandran_{args.cov_form}_{args.num_steps}steps_{indices_str}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Ramachandran plots saved to {figures_dir}")

if __name__ == '__main__':
    main() 
