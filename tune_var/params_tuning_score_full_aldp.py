import torch
import argparse
import yaml
from model import ScoreNet, construct_R
from tqdm import tqdm
from training_utils import load_dataset, load_target_dist
import numpy as np
import os
import math
from pathlib import Path

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR, PARAMS_CHECKPOINTS_LOW_RANK_DIR
)

def build_covariance_matrices(dataset, atom_mat, labels, n_dim, n_particle, time_steps, proj_mat, cov_form='diag', gammas=None):
    """
    For each step, compute the covariance matrix.
    """
    if dataset != 'aldp':
        raise NotImplementedError

    device = atom_mat.device
    m = n_particle
    I_n = torch.eye(n_dim, device=device, dtype=atom_mat.dtype)

    cov_matrices = torch.zeros(time_steps.shape[0]-1, n_dim*(m-1), n_dim*(m-1), device=device)

    sigma2_ddpm = torch.zeros(time_steps.shape[0]-1, 1, 1, device=device)

    for n in range(atom_mat.shape[0]):
        if cov_form == 'diag':
            diag = atom_mat[n][labels]
            B = torch.diag(diag)  # construct the diagonal matrix from this
            cov_n = torch.kron(B, I_n)
        elif cov_form == 'full':
            assert gammas is not None
            full_B = atom_mat[n][labels[:, None], labels[None, :]] + gammas[n] ** 2 * torch.eye(m, device=device)
            cov_n = torch.kron(full_B, I_n)
        else:
            raise NotImplementedError

        low_rank_mat = proj_mat.T @ cov_n @ proj_mat
        cov_matrices[n] = 0.5 * (low_rank_mat + low_rank_mat.T)

        tn, tn1 = time_steps[n], time_steps[n+1]
        sigma2_ddpm[n] = tn ** 2 / tn1 ** 2 * (tn1 ** 2 - tn ** 2)

    cov_mat_all = cov_matrices * sigma2_ddpm

    return cov_mat_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--cont_training', action='store_true')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')

    parser.add_argument('--cov_form', type=str, default='diag')

    # Optional: add hyperparameter for forward call (e.g., alpha)
    parser.add_argument('--alpha', type=float, default=2.0)
    args = parser.parse_args()

    # set seed to be params_index
    torch.manual_seed(args.params_index)
    np.random.seed(args.params_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    true_samples = load_dataset(dataset=args.dataset, device=device)

    # Load the pretrained score model
    score_checkpoint_path = get_model_checkpoint_path('{args.dataset}', '{args.net}', 'score', {args.model_index})

    config_path = (
        fstr(Path(__file__).parent.parent / 'consistency_sampling/model/configs/')
        f'{args.dataset}_{args.net}_config.yaml'
    )
    with open(config_path, 'r') as f:
        score_model_config = yaml.safe_load(f)

    score_model = ScoreNet(dataset=args.dataset, device=device, model_config=score_model_config, net=args.net).to(device)
    score_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint['model_state_dict'])
    score_model.eval()
    score_model.requires_grad_(False)

    n_dim, n_particle = score_model._n_dimension, score_model._n_particles

    # Load true target distribution
    true_target_dist = load_target_dist(args.dataset)

    # Prepare checkpoint paths
    param_checkpoint_path = PARAMS_CHECKPOINTS_LOW_RANK_DIR / args.dataset / f"{args.net}_params_{args.params_index}_{args.num_steps}steps_low_rank_{args.cov_form}cov_form.pth"
    log_dir = os.path.dirname(param_checkpoint_path)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(param_checkpoint_path))[0]}_log.txt")

    # construct projection matrix
    proj_mat = construct_R(n_particle, n_dim, device, seed=args.params_index)

    # permutable atoms for alanine dipeptide only
    atom_type_labels = [0, 1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13, 14, 15, 15, 15]
    labels = torch.tensor(atom_type_labels, dtype=torch.long, device=device)
    num_types = int(labels.max().item() + 1)

    # Initialize or load parameters
    if args.cont_training:
        print(f"Checkpoint found at {param_checkpoint_path}. Loading...")
        checkpoints = torch.load(param_checkpoint_path, map_location=device)
        # Load the tuning parameters consistently
        if args.cov_form == 'full':
            atom_mat_unconstrained = checkpoints['atom_mat_unconstrained']
            atom_mat_unconstrained = torch.nn.Parameter(atom_mat_unconstrained)

            gammas = checkpoints['gammas']
            gammas = torch.nn.Parameter(gammas)

        elif args.cov_form == 'diag':
            atom_mat_unconstrained = checkpoints['atom_mat_unconstrained']
            atom_mat_unconstrained = torch.nn.Parameter(atom_mat_unconstrained)

            gammas = None

        start_epoch = checkpoints['epoch']

    else:
        print("Tuning parameters from scratch.")
        start_epoch = 0

        if args.cov_form == 'diag':
            atom_mat_unconstrained = torch.nn.Parameter(torch.ones(args.num_steps-1, num_types, device=device))
            gammas = None
        elif args.cov_form == 'full':
            # atom_mat_unconstrained = torch.nn.Parameter(torch.eye(num_types, device=device).unsqueeze(0).repeat(args.num_steps-1, 1, 1))
            atom_mat_unconstrained = torch.nn.Parameter(torch.zeros(args.num_steps-1, num_types, num_types, device=device))
            gammas = torch.nn.Parameter(torch.ones(args.num_steps-1, device=device))
        else:
            raise NotImplementedError

    with open(log_file_path, 'w') as f:
        f.write(f'Tuning parameters for {args.dataset} dataset using {args.net} model\n')

    parameters_to_tune = [atom_mat_unconstrained]
    if args.cov_form == 'full':
        parameters_to_tune += [gammas]

    optimizer = torch.optim.Adam(parameters_to_tune, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - start_epoch)

    if args.cont_training:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])

    # Training loop with progress bar
    bar = tqdm(range(start_epoch, args.num_epochs), initial=start_epoch, total=args.num_epochs)
    for epoch in bar:
        with torch.no_grad():
            indices = torch.randperm(true_samples.size(0))[:args.num_samples]
            x0 = true_samples[indices].to(device)
            log_prob_0 = true_target_dist.log_prob(x0).to(device)

        optimizer.zero_grad()

        # Build the covariance matrices for each step
        if args.cov_form == 'diag':
            atom_mat = torch.relu(atom_mat_unconstrained)
        elif args.cov_form == 'full':
            atom_mat = torch.bmm(atom_mat_unconstrained, atom_mat_unconstrained.transpose(1, 2))
        else:
            raise NotImplementedError

        time_steps = torch.tensor(np.geomspace(score_model.eps, score_model.T, args.num_steps), device=device, dtype=atom_mat.dtype)
        cov_mat_all = build_covariance_matrices(args.dataset, atom_mat, labels, n_dim, n_particle, time_steps, proj_mat, cov_form=args.cov_form, gammas=gammas)

        # Forward pass of the score model to compute loss and effective sample size (ESS)
        loss, forward_ess, _ = score_model.forward_ess_low_rank(
            x0, log_prob_0, args.num_steps, cov_mat_all=cov_mat_all,
            alpha=args.alpha, time_steps=time_steps
        )
        forward_ess_percentage = forward_ess.item() / args.num_samples * 100

        loss.backward()
        optimizer.step()
        scheduler.step()

        bar.set_description(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Forward ESS (%): {forward_ess_percentage:.2f}')

        # Save checkpoint and log progress periodically
        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'atom_mat_unconstrained': atom_mat_unconstrained.detach(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'cov_mat_all': cov_mat_all,
                'time_steps': time_steps,
                'gammas':gammas
            }, param_checkpoint_path)

        with open(log_file_path, 'a') as f:
            f.write(f'Epoch {epoch+1} - Loss: {loss.item()}, Forward ESS: {forward_ess_percentage:.2f}%\n')

    print('Finished tuning parameters.')

if __name__ == '__main__':
    main()
