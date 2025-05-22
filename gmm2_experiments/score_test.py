import torch
import numpy as np
from consistency_sampling.gmm2_experiments.score_model import ScoreNet
import matplotlib.pyplot as plt
from gmm import create_gmm

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)

# run the ddpm model on the water molecule dataset for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the save path
input_dim = 1

n_layers = 7
hidden_size = 512

# Define GMM
model_checkpoint_path = fstr(Path(__file__).parent.parent / 'gmm2_experiments/score/checkpoints/{input_dim}D_gmm2_score_ckpt_{n_layers}layers_{hidden_size}hidden_size.pth')
gmm = create_gmm(input_dim, device=device)

score_model = ScoreNet(input_dim=input_dim, n_layers=n_layers, hidden_size=hidden_size).to(device)
score_model.load_state_dict(torch.load(model_checkpoint_path))
score_model.requires_grad_(False)

num_samples = 10000
# with torch.no_grad():
# model_samples, w, ess, _ = score_model.reverse_sde_sampler(num_steps=200000, num_samples=num_samples, true_target=gmm, analytical=True)
with torch.no_grad():
    model_samples, w, ess = score_model.ddpm_sampler(num_steps=100000, num_samples=num_samples, true_gmm=gmm)

    print('ESS (%)', ess.item() / num_samples * 100)

    true_samples = gmm.sample(num_samples)

    # plot the log prob
    model_log_prob = gmm.log_prob(model_samples)
    true_log_prob = gmm.log_prob(true_samples)

model_log_prob = model_log_prob.detach().cpu().numpy()
true_log_prob = true_log_prob.detach().cpu().numpy()

# visualization
plt.figure(figsize=(10, 6))
plt.hist(model_log_prob, bins=50, alpha=0.5, label='Model log_prob', density=True)
plt.hist(true_log_prob, bins=50, alpha=0.5, label='True log_prob', density=True)
plt.title('Histogram of log_prob')
plt.xlabel('log_prob')
plt.ylabel('Density')
plt.legend()
plt.savefig(f"/rds/user/fz287/hpc-work/dissertation/gmm2_experiments/score/figures/log_prob_hist_{input_dim}D.pdf", format='pdf', bbox_inches='tight')
plt.close()

model_samples = model_samples.detach().cpu().numpy()
true_samples = true_samples.detach().cpu().numpy()

# visualize the histograms
plt.figure(figsize=(10, 6))
plt.scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.3, label='Model Samples')
plt.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, label='True Samples')
plt.title("Visualization of Samples")
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.savefig(FIGURES_DIR / 'samples_visualization_{input_dim}D.png', format='png', bbox_inches='tight')
plt.show()
