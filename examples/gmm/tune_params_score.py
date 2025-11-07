import argparse
import numpy as np

import torch
from tqdm import tqdm

from gmm import create_gmm
from score_model import ScoreNet
from cov_tuned_diffusion.utils.path_config import (
    get_gmm2_model_checkpoint_path,
    get_gmm2_params_checkpoint_path,
)

def initialize_time_steps_params(num_steps, device, T=80.0, eps=0.002):
    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps-2)
    for n in range(2, num_steps):
        mus[n-2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus).to(device)

    return mus

def get_time_steps(num_steps, mus, device, eps=0.002, T=80.0):
    time_steps = torch.zeros(num_steps).to(device)
    time_steps[0], time_steps[-1] = eps, T
    for n in range(num_steps-1, 1, -1):
        tn = time_steps[n]
        tn1 = (tn - eps) * mus[n-2] + eps
        time_steps[n-1] = tn1

    return time_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=10)

    # gradient descent parameters
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_samples', type=int, default=1024)

    parser.add_argument('--num_steps_ours_start', type=int, default=5)
    parser.add_argument('--num_steps_ours', type=int, default=25)
    parser.add_argument('--step_size', type=int, default=5)

    # model parameters
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--hidden_size', type=int, default=512)

    parser.add_argument('--params_index', type=int, default=0)

    parser.add_argument('--cov_form', type=str, default='isotropic', choices=['isotropic', 'diagonal', 'full'])
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--tune_time_steps', action='store_true', default=False)
    args = parser.parse_args()

    if args.cov_form == 'full' and args.rank is None:
        parser.error("--rank is required when cov_form='full'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = args.num_samples

    # Define the GMM
    gmm = create_gmm(args.input_dim, device=device)

    # Load the diffusion model
    score_model = ScoreNet(input_dim=args.input_dim, n_layers=args.n_layers,
                           hidden_size=args.hidden_size, cov_form=args.cov_form).to(device)
    checkpoint_path = get_gmm2_model_checkpoint_path(
        input_dim=args.input_dim,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
    )
    score_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    num_steps_list = range(args.num_steps_ours_start, args.num_steps_ours + 1, args.step_size)

    for num_steps in num_steps_list:
        if args.cov_form == 'isotropic':
            cov_params_unconstrained = torch.ones(num_steps-1, device=device)
            cov_params_unconstrained = torch.nn.Parameter(cov_params_unconstrained)
        elif args.cov_form == 'diagonal':
            cov_params_unconstrained = torch.ones(num_steps-1, args.input_dim, device=device)
            cov_params_unconstrained = torch.nn.Parameter(cov_params_unconstrained)
        elif args.cov_form == 'full':
            assert args.rank >= 1 and args.rank <= args.input_dim, "Rank must be between 1 and input dimension"
            A = torch.randn(num_steps-1, args.input_dim, args.rank, device=device) * 0.01
            lam = torch.ones(num_steps-1, device=device)
            A = torch.nn.Parameter(A)
            lam = torch.nn.Parameter(lam)
            cov_params_unconstrained = {'A': A, 'lam': lam}
        else:
            raise ValueError("Invalid covariance form")

        if args.cov_form == "full":
            params_to_tune = list(cov_params_unconstrained.values())
        else:
            params_to_tune = [cov_params_unconstrained]

        if args.tune_time_steps:
            mus_unconstrained = torch.nn.Parameter(
                initialize_time_steps_params(num_steps, device)
            )
            params_to_tune.append(mus_unconstrained)

        # tune time steps
        optimizer = torch.optim.Adam(params_to_tune, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs,
                                                               eta_min=1e-6)

        bar = tqdm(range(args.num_epochs))
        for epoch in bar:
            optimizer.zero_grad()

            if args.tune_time_steps:
                mus = torch.sigmoid(mus_unconstrained)
                time_steps = get_time_steps(num_steps, mus, device)
            else:
                # time_steps = torch.tensor(np.geomspace(score_model.eps, score_model.T, num_steps), device=device)
                eps, T = score_model.eps, score_model.T
                rho = 1 / 7
                time_steps = (eps ** rho + torch.arange(0, num_steps, device=device) / (num_steps - 1) * (T ** rho - eps ** rho)) ** (1/rho)

            if args.cov_form == 'isotropic':
                cov_params = torch.functional.F.relu(cov_params_unconstrained)
            elif args.cov_form == 'diagonal':
                cov_params = torch.functional.F.relu(cov_params_unconstrained)
            elif args.cov_form == 'full':
                cov_params = cov_params_unconstrained
            else:
                raise ValueError("Invalid covariance form")

            x0 = gmm.sample(num_samples)

            log_prob_x0 = gmm.log_prob(x0)

            loss, forward_ess, _ = score_model.est_forward_ess(
                x0, log_prob_x0, num_steps, time_steps=time_steps,
                alpha=args.alpha, cov_params=cov_params
            )

            forward_ess_percentage = forward_ess.item() / num_samples * 100

            loss.backward()

            optimizer.step()
            scheduler.step()

            bar.set_description(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Forward ESS (%): {forward_ess_percentage:.2f}')

        print("Final Forward ESS: ", forward_ess_percentage)

        filename = get_gmm2_params_checkpoint_path(
            input_dim=args.input_dim,
            num_steps=num_steps,
            params_index=args.params_index,
            cov_form=args.cov_form,
            tune_time_steps=args.tune_time_steps,
            rank=args.rank if args.cov_form == 'full' else None,
        )
        torch.save({
            'cov_params': cov_params,
            'time_steps': time_steps,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': args
        }, filename)


if __name__ == '__main__':
    main()
