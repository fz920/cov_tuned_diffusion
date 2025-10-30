import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from score_model import ScoreNet
from gmm import create_gmm

from utils.path_config import (
    get_gmm2_ess_summary_path,
    get_gmm2_model_checkpoint_path,
    get_gmm2_params_checkpoint_path,
)


def load_covariance_params(args, cov_form, idx, num_steps):
    """
    Load covariance parameters for a specific form
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params_path = get_gmm2_params_checkpoint_path(
        input_dim=args.input_dim,
        num_steps=num_steps,
        params_index=idx,
        cov_form=cov_form,
        tune_time_steps=args.tune_time_steps,
        rank=args.rank if cov_form == "full" else None,
    )
    return torch.load(params_path, map_location=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--params_index_list', nargs='+', type=int, default=[0, 0, 0])
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--tune_time_steps', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.cov_forms = ['isotropic', 'diagonal', 'full', 'ddpm']

    if args.save_path is None:
        default_summary = get_gmm2_ess_summary_path(
            input_dim=args.input_dim,
            num_steps=args.num_steps,
            params_indices=args.params_index_list,
            rank=args.rank,
        ).with_suffix(".txt")
        save_path = default_summary
    else:
        save_path = Path(args.save_path).expanduser().resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path = save_path

    # load the score model
    score_model = ScoreNet(input_dim=args.input_dim, n_layers=7, hidden_size=512).to(device)
    score_ckpt_path = get_gmm2_model_checkpoint_path(
        input_dim=args.input_dim, n_layers=7, hidden_size=512
    )
    score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
    score_model.eval()
    score_model.requires_grad_(False)

    # load the gmm
    gmm = create_gmm(args.input_dim, device)

    results_forward_ess = {
      cov_form: []
      for cov_form in args.cov_forms
    }

    results_reverse_ess = {
      cov_form: []
      for cov_form in args.cov_forms
    }
    
    # Sample from the GMM once
    x0 = gmm.sample(args.num_samples)
    log_prob_x0 = gmm.log_prob(x0)

    for cov_form in tqdm(args.cov_forms):
        for idx in args.params_index_list:
            if cov_form == 'ddpm':
                cov_params = None
                time_steps = None
            else:
                ckpt = load_covariance_params(args, cov_form, idx, args.num_steps)
                cov_params = ckpt['cov_params']
                time_steps = ckpt['time_steps']

            with torch.no_grad():
                _, forward_ess, _ = score_model.est_forward_ess(
                    x0, log_prob_x0, args.num_steps,
                    cov_params=cov_params,
                    time_steps=time_steps,
                    cov_form=cov_form,
                    progress_bar=True
                )
                pct = forward_ess.item() / args.num_samples * 100
                results_forward_ess[cov_form].append(pct)
    
                _, _, reverse_ess = score_model.ddpm_sampler(
                    args.num_steps, args.num_samples,
                    cov_params=cov_params,
                    time_steps=time_steps,
                    cov_form=cov_form,
                    true_gmm=gmm,
                    progress_bar=True
                )
                pct = reverse_ess.item() / args.num_samples * 100
                results_reverse_ess[cov_form].append(pct)

    # compute mean and std
    summary_forward_ess = {}
    for cov_form in results_forward_ess:
        arrs = results_forward_ess[cov_form]
        summary_forward_ess[cov_form] = {
            'mean': np.mean(arrs),
            'std': np.std(arrs)
        }
        
    summary_reverse_ess = {}
    for cov_form in results_reverse_ess:
        arrs = results_reverse_ess[cov_form]
        summary_reverse_ess[cov_form] = {
            'mean': np.mean(arrs),
            'std': np.std(arrs)
        }

    # save all results in the text file
    with open(args.save_path, 'w') as f:
        f.write(f"ESS Results Summary for {args.input_dim}D GMM with {args.num_steps} steps\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Parameters indices: {args.params_index_list}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Forward ESS (%):\n")
        for cov_form in args.cov_forms:
            mean = summary_forward_ess[cov_form]['mean']
            std = summary_forward_ess[cov_form]['std']
            f.write(f"{cov_form.capitalize():10s}: {mean:.4f} ± {std:.4f}\n")
        
        f.write("\nReverse ESS (%):\n")
        for cov_form in args.cov_forms:
            mean = summary_reverse_ess[cov_form]['mean']
            std = summary_reverse_ess[cov_form]['std']
            f.write(f"{cov_form.capitalize():10s}: {mean:.4f} ± {std:.4f}\n")
        
        # # Raw data
        # f.write("\nRaw Data:\n")
        # f.write("Forward ESS (%):\n")
        # for cov_form in args.cov_forms:
        #     f.write(f"{cov_form.capitalize()}: {', '.join([f'{x:.4f}' for x in results_forward_ess[cov_form]])}\n")
        
        # f.write("\nReverse ESS (%):\n")
        # for cov_form in args.cov_forms:
        #     f.write(f"{cov_form.capitalize()}: {', '.join([f'{x:.4f}' for x in results_reverse_ess[cov_form]])}\n")

    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
