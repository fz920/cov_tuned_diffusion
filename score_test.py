import torch
import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from model import ScoreNet
from scipy.spatial.distance import pdist
from training_utils import load_dataset, load_target_dist
from utils.path_config import get_config_path, get_model_checkpoint_path, FIGURES_DIR

def load_model(args, device):
    """Load the score model from the checkpoint."""
    model_config_path = get_config_path(args.dataset, args.net)
    model_config = yaml.safe_load(open(model_config_path, 'r'))
    model = ScoreNet(dataset=args.dataset, device=device,
                     model_config=model_config, net=args.net).to(device)
    if args.use_ot:
        checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", args.index, use_ot=True)
    else:
        checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", args.index, use_ot=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.requires_grad_(False)
    return model

def calculate_inter_atomic_distances(positions):
    """Calculate pairwise distances for all molecules."""
    distances = []
    for sample in positions:
        pairwise_dist = pdist(sample)
        distances.append(pairwise_dist)
    return np.concatenate(distances)

def visualize_histograms(args, model_energies, true_energies, model_distances, true_distances):
    """Visualize energy and distance histograms."""
    output_dir = FIGURES_DIR / "test" / "score"

    os.makedirs(output_dir, exist_ok=True)

    # Energy histograms
    plt.figure(figsize=(10, 6))
    plt.hist(model_energies, bins=50, alpha=0.5, label='Model Energies', density=True)
    plt.hist(true_energies, bins=50, alpha=0.5, label='True Energies', density=True)
    plt.title('Histogram of Energies')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'energy_histogram_score_{args.dataset}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # Distance histograms
    plt.figure(figsize=(10, 6))
    plt.hist(model_distances, bins=50, alpha=0.5, label='Model Distances', density=True)
    plt.hist(true_distances, bins=50, alpha=0.5, label='True Distances', density=True)
    plt.title('Histogram of Inter-Atomic Distances')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'distance_histogram_score_{args.dataset}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='aldp', help='Dataset to use')
    parser.add_argument('--net', type=str, default='egnn', help='Model architecture to use')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to use')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of sampling steps')
    parser.add_argument('--index', type=int, default=0, help='Index of the model checkpoint')

    parser.add_argument('--use_ot', action='store_true', help='Use OT loss')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the true energy model
    true_target_dist = load_target_dist(args.dataset)

    # Load the score model
    score_model = load_model(args, device)

    # Run the DDPM sampler
    model_samples, w, ess, _ = score_model.ddpm_sampler(
        num_steps=args.num_steps, num_samples=args.num_samples,
        true_target=true_target_dist, progress_bar=True
    )

    print('ESS (%):', ess.item() / args.num_samples * 100)

    # Compute energies
    # model_samples = torch.tensor(model_samples, device=device)
    model_energies = true_target_dist.energy(model_samples)

    true_samples = load_dataset(args.dataset, device=device)
    true_samples = true_samples[:args.num_samples]
    true_energies = true_target_dist.energy(true_samples)

    # test for biasness
    test_phi = lambda x: torch.exp(x.norm(dim=(1,2)))
    true_value = torch.mean(test_phi(true_samples))
    model_value = torch.mean(test_phi(model_samples))

    # reweight samples
    reweighted_values = torch.sum(test_phi(model_samples) * w)
    print('True value:', true_value.item())
    print('Model value:', model_value.item())
    print('Reweighted value:', reweighted_values.item())

    if isinstance(model_energies, torch.Tensor):
        model_energies = model_energies.detach().cpu().numpy()
    if isinstance(true_energies, torch.Tensor):
        true_energies = true_energies.detach().cpu().numpy()

    # Calculate distances
    model_samples_np = model_samples.detach().cpu().numpy()
    true_samples_np = true_samples.detach().cpu().numpy()
    model_distances = calculate_inter_atomic_distances(model_samples_np)
    true_distances = calculate_inter_atomic_distances(true_samples_np)

    # Visualize results
    visualize_histograms(args, model_energies, true_energies, model_distances, true_distances)

if __name__ == '__main__':
    main()
