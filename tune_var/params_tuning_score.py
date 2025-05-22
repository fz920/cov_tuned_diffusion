import torch
import argparse
from model import ScoreNet
from tqdm import tqdm
import yaml
from training_utils import load_dataset, load_target_dist
import numpy as np
import os

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
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
    # gradient descent parameters
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--cont_training', action='store_true')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=10)

    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    # parser.add_argument('--use_ot', action='store_true')
    parser.add_argument('--tune_time_steps', action='store_true', default=False)

    parser.add_argument('--rho', type=int, default=0)

    # set alpha
    parser.add_argument('--alpha', type=float, default=2.0)
    args = parser.parse_args()

    # set seed to be params_index
    torch.manual_seed(args.params_index)
    np.random.seed(args.params_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    true_samples = load_dataset(dataset=args.dataset, device=device)

    # Load the score model
    # if args.use_ot:
    #     score_checkpoint_path = get_model_checkpoint_path('{args.dataset}', '{args.net}', 'score', {args.model_index}_ot)
    # else:
    score_checkpoint_path = get_model_checkpoint_path('{args.dataset}', '{args.net}', 'score', {args.model_index})

    score_model_config = yaml.safe_load(open(get_config_path('{args.dataset}', '{args.net}', 'config'), 'r'))
    score_model = ScoreNet(dataset=args.dataset, device=device, model_config=score_model_config,
                           net=args.net).to(device)
    score_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint['model_state_dict'])
    score_model.eval()
    score_model.requires_grad_(False)

    # get true target density
    true_target_dist = load_target_dist(args.dataset)

    # Prepare checkpoint paths
    if args.tune_time_steps:
        param_checkpoint_path = get_params_checkpoint_path(
            args.dataset, 
            args.net, 
            args.params_index, 
            args.num_steps, 
            tune_timesteps=True
        )
        log_dir = os.path.dirname(param_checkpoint_path)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(param_checkpoint_path))[0]}_log.txt")
    else:
        if args.rho != 0:
            param_checkpoint_path = get_params_checkpoint_path(
                args.dataset, 
                args.net, 
                args.params_index, 
                args.num_steps, 
                alpha=args.alpha, 
                rho=args.rho
            )
            log_dir = os.path.dirname(param_checkpoint_path)
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(param_checkpoint_path))[0]}_log.txt")
        else:
            param_checkpoint_path = get_params_checkpoint_path(
                args.dataset, 
                args.net, 
                args.params_index, 
                args.num_steps, 
                alpha=args.alpha
            )
            log_dir = os.path.dirname(param_checkpoint_path)
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(param_checkpoint_path))[0]}_log.txt")

    if args.cont_training:
        print(f"Checkpoint found at {param_checkpoint_path}. Loading...")
        checkpoints = torch.load(param_checkpoint_path, map_location=device)
        nus_unconstrained = torch.nn.Parameter(checkpoints['nus_unconstrained'])
        start_epoch = checkpoints['epoch']

    else:
        print("Tuning parameters from scratch.")
        start_epoch = 0
        nus_unconstrained = torch.nn.Parameter(torch.ones(args.num_steps-1).to(device) * 1)
        if args.tune_time_steps:
            mus_unconstrained = initialize_time_steps_params(args.num_steps, device, T=score_model.T, eps=score_model.eps)
            mus_unconstrained = torch.nn.Parameter(mus_unconstrained)

        with open(log_file_path, 'w') as f:
            f.write(f'Tuning parameters for {args.dataset} dataset using {args.net} model\n')

    parameters_to_tune = [nus_unconstrained]
    if args.tune_time_steps:
        parameters_to_tune += [mus_unconstrained]

    optimizer = torch.optim.Adam(parameters_to_tune, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs-start_epoch, eta_min=1e-6
    )

    if args.cont_training:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])

    bar = tqdm(range(start_epoch, args.num_epochs), initial=start_epoch,
            total=args.num_epochs)
    for epoch in bar:
        with torch.no_grad():
            indices = torch.randperm(true_samples.size(0))[:args.num_samples]
            x0 = true_samples[indices, :, :].to(device)
            log_prob_0 = true_target_dist.log_prob(x0).to(device)

        optimizer.zero_grad()

        nus = torch.nn.functional.relu(nus_unconstrained)
        
        # if args.tune_time_steps:
        #     mus = torch.sigmoid(mus_unconstrained)
        #     time_steps = get_time_steps(args.num_steps, mus, device=device, eps=score_model.eps, T=score_model.T)
        # else:
        #     # fixed time steps
        #     time_steps = torch.tensor(np.geomspace(score_model.eps, score_model.T, args.num_steps), device=device)
        if args.rho > 0:
            eps, T = score_model.eps, score_model.T
            time_steps = (eps**(1/args.rho) +
                        torch.linspace(0, 1, args.num_steps, device=device)*(T**(1/args.rho) - eps**(1/args.rho))
                        ) ** args.rho
        else:
            # rho = \infty
            time_steps = torch.tensor(np.geomspace(score_model.eps, score_model.T, args.num_steps), device=device)

        alpha_div, forward_ess, _ = score_model.est_forward_ess(
            x0, log_prob_0, num_steps=args.num_steps, nus=nus, progress_bar=False,
            time_steps=time_steps, alpha=args.alpha, tune_time_steps=args.tune_time_steps
        )

        loss = alpha_div

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        bar.set_description(f'Epoch: {epoch+1}, Loss: {loss.item():.3f}, LR: {current_lr:.6f}, Forward ESS: {forward_ess.item() / args.num_samples * 100:.2f}%')

        if (epoch+1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'time_steps': time_steps,
                'nus_unconstrained': nus_unconstrained,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'nus': nus,
            }, param_checkpoint_path)

            with open(log_file_path, 'a') as f:
                f.write(f'Epoch {epoch+1} - Loss: {loss.item()}, Forward ESS: {forward_ess.item() / args.num_samples * 100:.2f}%\n')

if __name__ == '__main__':
    main()
