import torch
import argparse
import os
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist

from torch.profiler import profile, ProfilerActivity, record_function
import time

# Import from local modules
from cov_tuned_diffusion import (
    ScoreNet,
    compute_forward_ess,
    compute_reverse_ess,
    load_target_dist,
    load_dataset,
)
from cov_tuned_diffusion.utils.path_config import get_config_path, get_model_checkpoint_path, FIGURES_DIR

def calculate_inter_atomic_distances(positions):
    """Calculate pairwise distances for all molecules."""
    distances = []
    for sample in positions:
        pairwise_dist = pdist(sample)
        distances.append(pairwise_dist)
    return np.concatenate(distances)

def visualize_histograms(args, model_energies, true_energies, model_distances, true_distances):
    """Visualize energy and distance histograms."""
    output_dir = FIGURES_DIR / "ode_test"
    os.makedirs(output_dir, exist_ok=True)

    # Energy histograms
    plt.figure(figsize=(10, 6))
    plt.hist(model_energies, bins=50, alpha=0.5, label='ODE Samples Energies', density=True)
    plt.hist(true_energies, bins=50, alpha=0.5, label='True Energies', density=True)
    plt.title('Histogram of Energies')
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{args.dataset}_energy_histogram_ode_{args.sampler}_{args.num_steps}steps.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Distance histograms
    plt.figure(figsize=(10, 6))
    plt.hist(model_distances, bins=50, alpha=0.5, label='ODE Samples Distances', density=True)
    plt.hist(true_distances, bins=50, alpha=0.5, label='True Distances', density=True)
    plt.title('Histogram of Inter-Atomic Distances')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{args.dataset}_distance_histogram_ode_{args.sampler}_{args.num_steps}steps.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def load_model(args, device):
    """Load the score model from the checkpoint."""
    model_config_path = get_config_path(args.dataset, args.net)
    model_config = yaml.safe_load(open(model_config_path, 'r'))
    model = ScoreNet(dataset=args.dataset, device=device,
                     model_config=model_config, net=args.net).to(device)

    checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", 0)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.requires_grad_(False)
    return model

def log_output(message, log_file=None):
    """Print output to console and save to log file if provided."""
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")

def euler_sampler(score_model, true_target_dist, num_samples, num_steps,
                  batch_size=100, progress_bar=False):
    """
    Generate samples using the Euler sampler
    """
    device = score_model.device
    
    # Initialize samples from prior
    xt = torch.randn(num_samples, score_model._n_particles, score_model._n_dimension,
                     device=device)
    
    eps, T = score_model.eps, score_model.T
    time_steps = torch.linspace(T, eps, num_steps).to(device)
    dt = (time_steps[0] - time_steps[-1]) / (num_steps - 1)
    
    range_iter = range(num_steps - 1)
    if progress_bar:
        range_iter = tqdm(range_iter, desc="Euler sampler")
    
    for i in range_iter:
        t = time_steps[i] * torch.ones(num_samples, device=device)
        
        # Process in batches
        all_score = []
        for j in range(0, num_samples, batch_size):
            batch_xt = xt[j:j+batch_size]
            batch_t = t[j:j+batch_size]
            score = score_model.score_fn(batch_xt, batch_t)
            all_score.append(score)
        
        score = torch.cat(all_score, dim=0)
        
        # Euler step
        xt = xt + dt * (-0.5 * xt + (time_steps[i] ** 2) * score)
    
    # Final evaluation
    log_probs = []
    with torch.no_grad():
        for j in range(0, num_samples, batch_size):
            batch_xt = xt[j:j+batch_size]
            log_prob = true_target_dist.log_prob(batch_xt)
            log_probs.append(log_prob)
    
    log_probs = torch.cat(log_probs, dim=0)
    
    return xt, log_probs

def main():
    parser = argparse.ArgumentParser(description="Test DDIM sampling (ODE) with GFLOPs measurement")
    
    # Dataset and model settings
    parser.add_argument('--dataset', type=str, default='lj13', choices=['lj13', 'aldp', 'dw4', 'lj55'], 
                        help='Dataset to use for sampling')
    parser.add_argument('--net', type=str, default='egnn', help='Model architecture')
    parser.add_argument('--model_index', type=int, default=0, help='Index of the model checkpoint')
    parser.add_argument('--sample_index', type=int, default=0, help='Index for the sample batch')
    
    # Sampling settings
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps for DDIM sampling')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--measure_flops', action='store_true', default=False, help='Measure FLOPs during sampling')
    parser.add_argument('--continue_sampling', action='store_true', help='Continue sampling from the last sample')

    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddim', 'ddpm', 'euler'])
    
    args = parser.parse_args()
    
    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(args.save_dir, f"{args.dataset}_{args.sampler}_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write(f"Results for {args.dataset} dataset using {args.sampler} sampling\n")
        f.write(f"Sampling parameters: num_samples={args.num_samples}, num_steps={args.num_steps}, batch_size={args.batch_size}\n")
        f.write("-" * 80 + "\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_output(f"Using device: {device}", log_file)
    
    # Create directories for forward weights
    forward_save_dir = os.path.join(args.save_dir, 'forward')
    os.makedirs(forward_save_dir, exist_ok=True)

    # Set file path for forward weights
    forward_pkl_path = os.path.join(forward_save_dir, f'{args.num_steps}steps_{args.sample_index}sample.pkl')
    
    # Load previous results if continuing sampling
    all_log_w_fwd = []
    if args.continue_sampling and os.path.exists(forward_pkl_path):
        with open(forward_pkl_path, 'rb') as f:
            previous_results = pickle.load(f)
        if 'log_weights' in previous_results and previous_results['log_weights']:
            gpu_log_weights = [w.to(device) for w in previous_results['log_weights']]
            all_log_w_fwd.extend(gpu_log_weights)
        log_output(f"Loaded previous forward results from {forward_pkl_path}", log_file)

    # Initialize model
    model = load_model(args, device=device)
    
    # Load the true target distribution
    true_target = load_target_dist(args.dataset)

    if args.sampler == 'ddim':
        reverse_sampler = model.ddim_sampler
    elif args.sampler == 'ddpm':
        reverse_sampler = model.ddpm_sampler
    elif args.sampler == 'euler':
        reverse_sampler = euler_sampler
    else:
        raise NotImplementedError
    
    # Run DDIM sampler with FLOPs measurement
    if args.measure_flops:
        log_output(f"Running DDIM sampler with {args.num_steps} steps and {args.num_samples} samples", log_file)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True) as prof, \
             record_function("ddim_sampling"):
            samples, weights, ess, _ = reverse_sampler(
                num_steps=args.num_steps,
                true_target=true_target,
                num_samples=args.num_samples,
                progress_bar=True
            )

        # Calculate and print FLOPs
        sampling_flops = sum(e.flops for e in prof.key_averages())
        log_output(f'Total FLOPs: {sampling_flops/1e9:.2f}G', log_file)
        log_output(f'FLOPs per sample: {sampling_flops/args.num_samples/1e9:.4f}G', log_file)
    else:
        # Run without profiling
        # Handle large sample sizes by batching
        batch_size = args.batch_size
        num_batches = (args.num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        all_samples = []
        all_log_w = []
        
        for i in range(num_batches):
            log_output(f"Processing batch {i+1}/{num_batches}", log_file)
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, args.num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_samples, batch_weights, batch_ess, batch_log_w = reverse_sampler(
                num_steps=args.num_steps,
                true_target=true_target,
                num_samples=current_batch_size,
                progress_bar=True
            )
            
            all_samples.append(batch_samples)
            all_log_w.append(batch_log_w)
        
        # Combine results
        samples = torch.cat(all_samples, dim=0)
        log_w_batch = torch.cat(all_log_w, dim=0)
        
        # Compute weights and ESS from log_w_batch
        rev_ess = compute_reverse_ess(log_w_batch)

        weights = log_w_batch - torch.max(log_w_batch)
        weights /= weights.sum()

        log_output(f"Reverse ESS (%): {rev_ess.item() / args.num_samples * 100:.2f}%", log_file)

    # Load true samples for comparison
    true_samples = load_dataset(args.dataset, device=device, partition='test')
    true_samples = true_samples[torch.randperm(len(true_samples))[:args.num_samples]]

    # compute forward ESS
    x0 = true_samples.clone()
    log_prob_x0 = true_target.log_prob(x0)
    
    # Handle large sample sizes by batching
    batch_size = args.batch_size
    num_batches = (len(x0) + batch_size - 1) // batch_size  # Ceiling division
    
    if not hasattr(all_log_w_fwd, 'append'):
        all_log_w_fwd = []
    
    for i in range(num_batches):
        log_output(f"Processing forward batch {i+1}/{num_batches}", log_file)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x0))
        
        batch_x0 = x0[start_idx:end_idx]
        batch_log_prob_x0 = log_prob_x0[start_idx:end_idx]
        
        _, batch_log_w = model.forward_ess_ddim(batch_x0, batch_log_prob_x0, args.num_steps,
                                            progress_bar=True)
        
        all_log_w_fwd.append(batch_log_w)
    
    # Combine results
    log_w_fwd = torch.cat(all_log_w_fwd, dim=0)
    
    # Compute weights and ESS from log_w_fwd
    fwd_ess = compute_forward_ess(log_w_fwd)
    
    log_output(f"Sampling completed", log_file)
    log_output(f"Total samples: {len(log_w_fwd)}", log_file)
    log_output(f"Forward ESS (%): {fwd_ess.item() / args.num_samples * 100:.2f}%", log_file)

    # Save forward weights to pickle file
    # Move all tensors to CPU before saving
    gpu_log_weights = [w for w in all_log_w_fwd]
    results_forward = {'log_weights': gpu_log_weights}
    with open(forward_pkl_path, 'wb') as f:
        pickle.dump(results_forward, f)
    log_output(f"Forward weights saved to {forward_pkl_path}", log_file)

    # Compute energies
    model_energies = true_target.energy(samples)
    true_energies = true_target.energy(true_samples)
    
    # Calculate bias test (similar to score_test.py)
    test_phi = lambda x: torch.exp(x.norm(dim=(1,2)))
    true_value = torch.mean(test_phi(true_samples))
    model_value = torch.mean(test_phi(samples))
    
    # Reweight samples
    reweighted_values = torch.sum(test_phi(samples) * weights)
    log_output('True value:', log_file)
    log_output(f'{true_value.item()}', log_file)
    log_output('Model value:', log_file)
    log_output(f'{model_value.item()}', log_file)
    log_output('Reweighted value:', log_file)
    log_output(f'{reweighted_values.item()}', log_file)
    
    # Convert to numpy for visualization
    if isinstance(model_energies, torch.Tensor):
        model_energies = model_energies.detach().cpu().numpy()
    if isinstance(true_energies, torch.Tensor):
        true_energies = true_energies.detach().cpu().numpy()
    
    # Resample model samples based on importance weights
    indices = torch.multinomial(weights, num_samples=args.num_samples, replacement=True)
    resampled_samples = samples[indices]
    samples = resampled_samples

    # Calculate distances
    model_samples_np = samples.detach().cpu().numpy()
    true_samples_np = true_samples.detach().cpu().numpy()
    model_distances = calculate_inter_atomic_distances(model_samples_np)
    true_distances = calculate_inter_atomic_distances(true_samples_np)
    
    # Visualize results
    visualize_histograms(args, model_energies, true_energies, model_distances, true_distances)
    
    # Save samples and metrics
    save_path = os.path.join(args.save_dir, f"{args.dataset}_ddim_samples.pt")
    torch.save({
        'samples': samples.cpu(),
        'weights': weights.cpu(),
        # 'rev_ess': rev_ess.cpu(),
        'flops': sampling_flops if args.measure_flops else None,
        'model_energies': model_energies,
        'true_energies': true_energies,
        'model_distances': model_distances,
        'true_distances': true_distances
    }, save_path)
    
    log_output(f"Results saved to {save_path}", log_file)
    log_output(f"Log file saved to {log_file}", log_file)

if __name__ == "__main__":
    main()
