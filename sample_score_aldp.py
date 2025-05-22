# Description: Get importance samples and weights from the model."""
import torch
import argparse
import os
import yaml
import numpy as np
import pickle

from tqdm import tqdm

from model import ScoreNet, compute_forward_ess, compute_reverse_ess, CovNet

from training_utils import load_target_dist, load_dataset
import logging

from utils.path_config import (
    get_config_path, 
    get_model_checkpoint_path, 
    get_params_checkpoint_path,
    get_sample_path,
    PARAMS_CHECKPOINTS_LOW_RANK_DIR
)

def load_covariance_params(args, cov_form, idx, num_steps):
    """
    Load covariance parameters for a specific form
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cov_form == 'diag' or cov_form == 'full':
        # Create the path using path configuration
        dataset_dir = PARAMS_CHECKPOINTS_LOW_RANK_DIR / args.dataset
        os.makedirs(dataset_dir, exist_ok=True)
        param_checkpoint_path = dataset_dir / f"{args.net}_params_{idx}_{num_steps}steps_low_rank_{cov_form}cov_form.pth"
    else:
        raise ValueError(f"Invalid covariance form: {cov_form}")

    checkpoint = torch.load(param_checkpoint_path, map_location=device)
    return checkpoint

def main():
    parser = argparse.ArgumentParser()

    # Sampling settings
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--params_index', type=int, default=0)
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--sample_index', type=int, default=0)

    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_steps', type=int, default=100)

    parser.add_argument('--sample_num_times', type=int, default=1)
    parser.add_argument('--cov_forms', type=str, default=['ddpm'], nargs='+', 
                        choices=['ddpm', 'isotropic', 'diag', 'full'],
                        help='Covariance form(s) to use. Can specify multiple options.')
    
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the samples and weights.')
    parser.add_argument('--continue_sampling', action='store_true',
                        help='Continue sampling from the last sample.')

    args = parser.parse_args()

    assert args.dataset == 'aldp'

    # set the seed
    torch.manual_seed(args.sample_index)
    np.random.seed(args.sample_index)

    # Set up save paths
    if args.save_path is None:
        from utils.path_config import SAMPLES_DIR
        args.save_path = SAMPLES_DIR / args.dataset

    # Create save directories
    forward_save_dir = os.path.join(args.save_path, f'forward')
    backward_save_dir = os.path.join(args.save_path, f'backward')

    os.makedirs(forward_save_dir, exist_ok=True)
    os.makedirs(backward_save_dir, exist_ok=True)

    # Set file paths
    forward_pkl_path = os.path.join(forward_save_dir, f'{args.num_steps}steps_{args.sample_index}sample_aldp_{args.cov_forms}.pkl')
    backward_pkl_path = os.path.join(backward_save_dir, f'{args.num_steps}steps_{args.sample_index}sample_aldp_{args.cov_forms}.pkl')

    # Configure logging to file
    log_dir = os.path.join(args.save_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'sampling_{args.dataset}_{args.num_steps}steps_{args.sample_index}sample.log')
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='a'  # Append to the log file if it exists
    )
    logging.info(f"Logging to file: {log_file_path}")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the score model
    config_path = get_config_path(args.dataset, args.net)
    with open(config_path, 'r') as f:
        score_model_config = yaml.safe_load(f)
    
    # Load the pretrained score model
    score_checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", args.model_index)
    score_model = ScoreNet(dataset=args.dataset, device=device, model_config=score_model_config,
                            net=args.net).to(device)
    score_checkpoint = torch.load(score_checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint['model_state_dict'])
    score_model.eval()
    score_model.requires_grad_(False)

    if 'model' in args.cov_forms:
        cov_net_config_path = get_config_path(args.dataset, args.net, type="cov")
        with open(cov_net_config_path, 'r') as f:
            cov_net_config = yaml.safe_load(f)

        cov_model = CovNet(dataset=args.dataset, device=device,
                        model_config=cov_net_config, net=args.net).to(device)
        ckpt = load_covariance_params(args, 'model', args.model_index, args.num_steps)
        cov_model.load_state_dict(ckpt['cov_model'])
        cov_model.eval()
        cov_model.requires_grad_(False)

    # load the true target distribution
    true_target_dist = load_target_dist(args.dataset)

    # load the dataset
    true_data = load_dataset(args.dataset, partition='test')

    # Load previous results if continuing sampling
    if args.continue_sampling:
        if os.path.exists(forward_pkl_path):
            with open(forward_pkl_path, 'rb') as f:
                results_forward = pickle.load(f)
            logging.info(f"Loaded previous forward results from {forward_pkl_path}")
        
        if os.path.exists(backward_pkl_path):
            with open(backward_pkl_path, 'rb') as f:
                results_backward = pickle.load(f)
            logging.info(f"Loaded previous backward results from {backward_pkl_path}")

    else:
        # Initialize results dictionaries with simpler structure
        results_forward = {cov_form: {'log_weights': []} for cov_form in args.cov_forms}
        results_backward = {cov_form: {'log_weights': [], 'samples': []} for cov_form in args.cov_forms}

    # sample from the score model
    for n in tqdm(range(args.sample_num_times), desc='Sampling'):
        # Get initial data samples
        x0 = true_data[np.random.choice(true_data.shape[0], args.num_samples, replace=False)]
        x0 = x0.to(device)
        log_prob_x0 = true_target_dist.log_prob(x0)

        for cov_form in args.cov_forms:
            with torch.no_grad():
                if cov_form == 'ddpm':
                    _, _, log_w = score_model.est_forward_ess(
                        x0, log_prob_x0, args.num_steps,
                        nus=None,
                        time_steps=None,
                        progress_bar=True
                    )
                    
                    samples, _, _, log_w_b = score_model.ddpm_sampler(
                        args.num_steps,
                        true_target_dist,
                        num_samples=args.num_samples,
                        nus=None,
                        time_steps=None,
                        progress_bar=True
                    )

                elif cov_form == 'isotropic':
                    ckpt = load_covariance_params(args, cov_form, args.params_index, args.num_steps, tune_time_steps=args.tune_time_steps)
                    _, _, log_w = score_model.est_forward_ess(
                        x0, log_prob_x0, args.num_steps,
                        nus=ckpt['nus'],
                        time_steps=ckpt['time_steps'] if args.tune_time_steps else None,
                        progress_bar=True,
                        tune_time_steps=args.tune_time_steps
                    )

                    samples, _, _, log_w_b = score_model.ddpm_sampler(
                        args.num_steps,
                        true_target_dist,
                        num_samples=args.num_samples,
                        nus=ckpt['nus'],
                        time_steps=ckpt['time_steps'] if args.tune_time_steps else None,
                        progress_bar=True,
                        tune_time_steps=args.tune_time_steps
                    )

                elif cov_form == 'model':
                    ckpt = load_covariance_params(args, cov_form, args.params_index, args.num_steps)
                    _, _, log_w = score_model.forward_ess_model(
                        x0, log_prob_x0, args.num_steps,
                        cov_model=cov_model,
                        lams=ckpt['lams'],
                        output_scale=ckpt['output_scale'],
                        diag=False,
                        progress_bar=True
                    )

                    samples, _, _, log_w_b = score_model.ddpm_sampler_model(
                        args.num_steps,
                        true_target_dist,
                        num_samples=args.num_samples,
                        cov_model=cov_model,
                        time_steps=ckpt['time_steps'],
                        lams=ckpt['lams'],
                        output_scale=ckpt['output_scale'],
                        diag=False,
                        progress_bar=True,
                    )

                elif cov_form == 'full' or cov_form == 'diag':
                    ckpt = load_covariance_params(args, cov_form, args.params_index, args.num_steps)
                    
                    _, _, log_w = score_model.forward_ess_low_rank(
                        x0, log_prob_x0, args.num_steps,
                        cov_mat_all=ckpt['cov_mat_all'],
                        time_steps=ckpt['time_steps'],
                        alpha=2.0,
                        progress_bar=True
                    )

                    samples, _, _, log_w_b = score_model.ddpm_sampler_low_rank(
                        args.num_steps,
                        true_target_dist,
                        num_samples=args.num_samples,
                        cov_mat_all=ckpt['cov_mat_all'],
                        time_steps=ckpt['time_steps'],
                        progress_bar=True
                    )

                else:
                    raise NotImplementedError

                # Append results to lists
                results_forward[cov_form]['log_weights'].append(log_w)
                results_backward[cov_form]['log_weights'].append(log_w_b)
                results_backward[cov_form]['samples'].append(samples)

                # Calculate ESS for this batch
                ess_forward = compute_forward_ess(log_w)
                ess_backward = compute_reverse_ess(log_w_b)

                ess_forward_percent = ess_forward / args.num_samples * 100
                ess_backward_percent = ess_backward / args.num_samples * 100

                logging.info(f'Batch {n} - {cov_form} - Forward ESS: {ess_forward_percent:.2f}%')
                logging.info(f'Batch {n} - {cov_form} - Backward ESS: {ess_backward_percent:.2f}%')

        # Save results after each batch
        with open(forward_pkl_path, 'wb') as f:
            pickle.dump(results_forward, f)
        
        with open(backward_pkl_path, 'wb') as f:
            pickle.dump(results_backward, f)
        
        logging.info(f"Saved batch {n} results to disk")

    # Calculate and log final ESS values
    for cov_form in args.cov_forms:
        # Concatenate all tensors in the lists
        all_forward_log_w = torch.cat(results_forward[cov_form]['log_weights'], dim=0)
        all_backward_log_w = torch.cat(results_backward[cov_form]['log_weights'], dim=0)

        # Calculate ESS on all data
        total_forward_ess = compute_forward_ess(all_forward_log_w)
        total_backward_ess = compute_reverse_ess(all_backward_log_w)

        total_samples = all_forward_log_w.shape[0]
        forward_ess_percent = total_forward_ess / total_samples * 100
        backward_ess_percent = total_backward_ess / total_samples * 100

        logging.info(f'Final {cov_form} - Forward ESS: {forward_ess_percent:.2f}% ({total_forward_ess:.1f}/{total_samples})')
        logging.info(f'Final {cov_form} - Backward ESS: {backward_ess_percent:.2f}% ({total_backward_ess:.1f}/{total_samples})')

    logging.info(f"Sampling complete. Forward results saved to {forward_pkl_path}")
    logging.info(f"Backward results saved to {backward_pkl_path}")

    logging.info(f"Current total number of forward samples: {all_forward_log_w.shape[0]}")
    logging.info(f"Current total number of backward samples: {all_backward_log_w.shape[0]}")

if __name__ == '__main__':
    main()
