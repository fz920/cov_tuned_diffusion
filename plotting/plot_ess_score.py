import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def calculate_ess(log_w, mode='reverse'):
    num_samples = log_w.shape[0]
    if mode == 'reverse':
        # Calculate the reverse SS
        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)

    elif mode == 'forward':
        # calculate the forward ESS
        Z_inv = torch.mean(torch.exp(-log_w))
        ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * Z_inv)

    return ess

def main():
    # Define the parser
    parser = argparse.ArgumentParser(description="Plot the comparison of ESS proportion between our method and DDPM method.")
    parser.add_argument("--dataset", type=str, default='aldp', help="Dataset to use.")
    parser.add_argument("--net", type=str, default='egnn', help="Network to use.")
    parser.add_argument("--model_index", type=int, default=0, help="Index of the model to use.")
    parser.add_argument("--params_index_list", type=int, nargs="+", help="List of indices of the parameter files to use.")
    parser.add_argument("--num_steps_list", type=int, nargs="+", help="List of number of steps to use for DDIM.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize storage for score_new and score samples
    reverse_ess_score_new_all = {
        num_steps: {} for num_steps in args.num_steps_list
    }

    forward_ess_score_new_all = {
        num_steps: {} for num_steps in args.num_steps_list
    }

    reverse_ess_score_all = {
        num_steps: {} for num_steps in args.num_steps_list
    }

    forward_ess_score_all = {
        num_steps: {} for num_steps in args.num_steps_list
    }

    for num_steps in args.num_steps_list:
        for param_index in args.params_index_list:

            # Paths for loading samples
            score_new_samples_path = (
                fstr(Path(__file__).parent.parent / 'consistency_sampling/importance_sampling/samples/')

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
                f'{args.dataset}/ours/{args.net}_ours_{args.model_index}model_{num_steps}steps_{param_index}.pth'
            )
            score_new_forward_ess_path = (
                fstr(Path(__file__).parent.parent / 'consistency_sampling/importance_sampling/samples/')

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
                f'{args.dataset}/ours/{args.net}_ours_forward_{args.model_index}model_{num_steps}steps_{param_index}.pth'
            )
            score_samples_path = (
                fstr(Path(__file__).parent.parent / 'consistency_sampling/importance_sampling/samples/')

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
                f'{args.dataset}/score/{args.net}_ddpm_{args.model_index}model_{num_steps}steps_{param_index}.pth'
            )
            score_forward_ess_path = (
                fstr(Path(__file__).parent.parent / 'consistency_sampling/importance_sampling/samples/')

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
                f'{args.dataset}/score/{args.net}_ddpm_forward_{args.model_index}model_{num_steps}steps_{param_index}.pth'
            )

            # Load checkpoint files
            score_new_samples_checkpoint = torch.load(score_new_samples_path, map_location=device)
            score_new_forward_checkpoint = torch.load(score_new_forward_ess_path, map_location=device)

            score_samples_checkpoint = torch.load(score_samples_path, map_location=device)
            score_forward_checkpoint = torch.load(score_forward_ess_path, map_location=device)

            # reverse log weights
            score_new_log_weights_total_reverse = score_new_samples_checkpoint['log_weights']
            score_log_weights_total_reverse = score_samples_checkpoint['log_weights']

            # forward log weights
            score_new_log_weights_total_forward = score_new_forward_checkpoint['log_weights']
            score_log_weights_total_forward = score_forward_checkpoint['log_weights']

            # compute ESS
            score_new_reverse_ess_total = calculate_ess(score_new_log_weights_total_reverse, mode='reverse')
            score_reverse_ess_total = calculate_ess(score_log_weights_total_reverse, mode='reverse')

            score_new_forward_ess_total = calculate_ess(score_new_log_weights_total_forward, mode='forward')
            score_forward_ess_total = calculate_ess(score_log_weights_total_forward, mode='forward')

            reverse_ess_score_new_all[num_steps][param_index] = score_new_reverse_ess_total.detach().cpu().numpy() / score_new_log_weights_total_reverse.shape[0] * 100
            forward_ess_score_new_all[num_steps][param_index] = score_new_forward_ess_total.detach().cpu().numpy() / score_new_log_weights_total_forward.shape[0] * 100

            reverse_ess_score_all[num_steps][param_index] = score_reverse_ess_total.detach().cpu().numpy() / score_log_weights_total_reverse.shape[0] * 100
            forward_ess_score_all[num_steps][param_index] = score_forward_ess_total.detach().cpu().numpy() / score_log_weights_total_forward.shape[0] * 100

            print("reverse_ess_score_new_all: ", reverse_ess_score_new_all[num_steps][param_index])
            print("forward_ess_score_new_all: ", forward_ess_score_new_all[num_steps][param_index])

            print("reverse_ess_score_all: ", reverse_ess_score_all[num_steps][param_index])
            print("forward_ess_score_all: ", forward_ess_score_all[num_steps][param_index])


            # Check consistency of data
            assert score_new_log_weights_total_reverse.shape[0] == score_log_weights_total_reverse.shape[0], "Number of log weights do not match."
            assert score_new_log_weights_total_forward.shape[0] == score_log_weights_total_forward.shape[0], "Number of log weights do not match."

            # Debugging information
            print(f"Num steps: {num_steps}, Param index: {param_index}")
            print(f"Current number of score_new reverse log weights: {score_new_log_weights_total_reverse.shape[0]}")
            print(f"Current number of Score reverse log weights: {score_log_weights_total_reverse.shape[0]}")
            print(f"Current number of score_new forward log weights: {score_new_log_weights_total_forward.shape[0]}")
            print(f"Current number of Score forward log weights: {score_log_weights_total_forward.shape[0]}")

    # Prepare data for plotting
    num_steps_list = args.num_steps_list

    score_new_reverse_ess_q75 = [np.percentile(list(reverse_ess_score_new_all[steps].values()), 75) for steps in num_steps_list]
    score_new_reverse_ess_q25 = [np.percentile(list(reverse_ess_score_new_all[steps].values()), 25) for steps in num_steps_list]
    score_new_reverse_ess_q50 = [np.percentile(list(reverse_ess_score_new_all[steps].values()), 50) for steps in num_steps_list]

    score_new_forward_ess_q75 = [np.percentile(list(forward_ess_score_new_all[steps].values()), 75) for steps in num_steps_list]
    score_new_forward_ess_q25 = [np.percentile(list(forward_ess_score_new_all[steps].values()), 25) for steps in num_steps_list]
    score_new_forward_ess_q50 = [np.percentile(list(forward_ess_score_new_all[steps].values()), 50) for steps in num_steps_list]

    score_reverse_ess_q75 = [np.percentile(list(reverse_ess_score_all[steps].values()), 75) for steps in num_steps_list]
    score_reverse_ess_q25 = [np.percentile(list(reverse_ess_score_all[steps].values()), 25) for steps in num_steps_list]
    score_reverse_ess_q50 = [np.percentile(list(reverse_ess_score_all[steps].values()), 50) for steps in num_steps_list]

    score_forward_ess_q75 = [np.percentile(list(forward_ess_score_all[steps].values()), 75) for steps in num_steps_list]
    score_forward_ess_q25 = [np.percentile(list(forward_ess_score_all[steps].values()), 25) for steps in num_steps_list]
    score_forward_ess_q50 = [np.percentile(list(forward_ess_score_all[steps].values()), 50) for steps in num_steps_list]

    score_new_forward_ess_yerr = np.array([
        [m - l for m, l in zip(score_new_forward_ess_q50, score_new_forward_ess_q25)],
        [h - m for h, m in zip(score_new_forward_ess_q75, score_new_forward_ess_q50)]
    ])

    score_new_reverse_ess_yerr = np.array([
        [m - l for m, l in zip(score_new_reverse_ess_q50, score_new_reverse_ess_q25)],
        [h - m for h, m in zip(score_new_reverse_ess_q75, score_new_reverse_ess_q50)]
    ])

    score_forward_ess_yerr = np.array([
        [m - l for m, l in zip(score_forward_ess_q50, score_forward_ess_q25)],
        [h - m for h, m in zip(score_forward_ess_q75, score_forward_ess_q50)]
    ])

    score_reverse_ess_yerr = np.array([
        [m - l for m, l in zip(score_reverse_ess_q50, score_reverse_ess_q25)],
        [h - m for h, m in zip(score_reverse_ess_q75, score_reverse_ess_q50)]
    ])

    # score_new_ess_mean = [np.mean(list(ess_score_new_all[steps].values())) for steps in num_steps_list]
    # score_new_ess_std = [np.std(list(ess_score_new_all[steps].values())) for steps in num_steps_list]

    # score_ess_mean = [np.mean(list(ess_score_all[steps].values())) for steps in num_steps_list]
    # score_ess_std = [np.std(list(ess_score_all[steps].values())) for steps in num_steps_list]

    # Plotting
    plt.figure(figsize=(7, 4))

    # Number of NFEs (assuming scaling)
    num_nfe_list = [1 * steps for steps in num_steps_list]

    # Plot error bars
    plt.errorbar(
        num_nfe_list,
        score_new_forward_ess_q50,
        yerr=score_new_forward_ess_yerr,
        marker='o',
        linestyle='-',
        color='#1f77b4',
        label='Ours (Forward ESS)',
        linewidth=2,
        capsize=3
    )

    plt.errorbar(
        num_nfe_list,
        score_new_reverse_ess_q50,
        yerr=score_new_reverse_ess_yerr,
        marker='o',
        linestyle='-',
        color='#76b7b2',
        label='Ours (Reverse ESS)',
        linewidth=2,
        capsize=3
    )

    plt.errorbar(
        num_nfe_list,
        score_forward_ess_q50,
        yerr=score_forward_ess_yerr,
        marker='o',
        linestyle='-',
        color='#d62728',
        label='DDPM (Forward ESS)',
        linewidth=2,
        capsize=3
    )

    plt.errorbar(
        num_nfe_list,
        score_reverse_ess_q50,
        yerr=score_reverse_ess_yerr,
        marker='o',
        linestyle='-',
        color='#ff9896',
        label='DDPM (Reverse ESS)',
        linewidth=2,
        capsize=3
    )

    plt.xlabel('NFE', fontsize=11)
    plt.ylabel('ESS (%)', fontsize=11)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True)
    plt.tight_layout()

    # Save and show the plot
    save_path = (
        f"/rds/user/fz287/hpc-work/dissertation/consistency_sampling/figures/ess/{args.dataset}/"
        f"{args.dataset}_our_ddpm_compare_{args.model_index}.pdf"
    )
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    # Create the output directory
    figures_dir = FIGURES_DIR / "ess" / args.dataset
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    save_path = figures_dir / f"ess_comparison_{args.dataset}.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    main()