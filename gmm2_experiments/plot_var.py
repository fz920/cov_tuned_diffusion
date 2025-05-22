import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from consistency_sampling.gmm2_experiments.score_model import ScoreNet

num_steps = 40
input_dim = 5
n_layers = 7
hidden_size = 512
index = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

score_model = ScoreNet(input_dim=input_dim, n_layers=n_layers, hidden_size=hidden_size).to(device)
score_model.load_state_dict(torch.load(fstr(Path(__file__).parent.parent / 'gmm2_experiments/score/checkpoints/{input_dim}D_gmm2_score_ckpt_{n_layers}layers_{hidden_size}hidden_size.pth'), map_location=device))
score_model.eval()
score_model.requires_grad_(False)

# load the parameters for ours to do importance sampling

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)
param_checkpoint_path = f"/rds/user/fz287/hpc-work/dissertation/gmm2_experiments/score/parameters/{input_dim}D_gmm2_score_params_{num_steps}steps_{index}_isotropic_with_time_steps.pth"
params_checkpoint = torch.load(param_checkpoint_path, map_location=device)
nus_tuned = params_checkpoint['nus']
nus_tuned = nus_tuned.detach().cpu().numpy()
time_steps_tuned = params_checkpoint['time_steps']
time_steps_tuned = time_steps_tuned.detach().cpu().numpy()

param_checkpoint_path = f"/rds/user/fz287/hpc-work/dissertation/gmm2_experiments/score/parameters/{input_dim}D_gmm2_score_params_{num_steps}steps_{index}_isotropic_without_time_steps.pth"
params_checkpoint = torch.load(param_checkpoint_path, map_location=device)
nus_without_time_steps = params_checkpoint['nus']
nus_without_time_steps = nus_without_time_steps.detach().cpu().numpy()
time_steps_without_time_steps = np.geomspace(0.002, 80.0, num_steps)

plt.figure()
plt.plot(time_steps_tuned[1:], nus_tuned, '-o', label='tuned time steps')
plt.plot(time_steps_without_time_steps[1:], nus_without_time_steps, '-o', label='without time steps')
plt.xlabel('Time step')
plt.ylabel('Nus')
plt.title(f'{input_dim}D GMM2 score params tuning trend')
plt.grid(True)
plt.xscale('log')
plt.legend()
plt.tight_layout()
# plt.savefig(FIGURES_DIR / 'var_figures/{net}_score_params_{params_index}_{num_steps}steps_nus_trend_rho1_7.png')
plt.savefig(FIGURES_DIR / '{input_dim}D_gmm2_score_params_{num_steps}steps_{index}_nus_trend.png')
plt.close()
