import torch
import argparse
import yaml
import os
import time
import numpy as np
from tqdm import tqdm

from model import ScoreNet
from training_utils import load_target_dist, load_dataset
from utils.path_config import get_config_path, get_model_checkpoint_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model config
    model_config_path = get_config_path(args.dataset, args.net)
    model_config = yaml.safe_load(open(model_config_path, 'r'))
    
    # Load model checkpoint
    checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", 0)
    
    # Create model
    score_model = ScoreNet(dataset=args.dataset, device=device, model_config=model_config,
                           net=args.net).to(device)
    score_checkpoint = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(score_checkpoint['model_state_dict'])
    score_model.eval()
    
    # Count parameters
    total_params = count_parameters(score_model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Load target distribution
    true_target_dist = load_target_dist(args.dataset)
    
    # Measure forward pass time (score function evaluation)
    num_batches = args.num_samples // args.batch_size
    
    # Initialize samples
    print(f"Measuring inference time for {args.num_samples} samples in batches of {args.batch_size}")
    xt = torch.randn(args.num_samples, score_model._n_particles, score_model._n_dimension,
                    device=device)
    t = torch.ones(args.num_samples, device=device) * score_model.T
    
    # Warmup
    for _ in range(5):
        for i in range(0, args.num_samples, args.batch_size):
            batch_xt = xt[i:i+args.batch_size]
            batch_t = t[i:i+args.batch_size]
            _ = score_model.score_fn(batch_xt, batch_t)
    
    # Measure score function time
    start_time = time.time()
    for i in range(0, args.num_samples, args.batch_size):
        batch_xt = xt[i:i+args.batch_size]
        batch_t = t[i:i+args.batch_size]
        _ = score_model.score_fn(batch_xt, batch_t)
    score_time = time.time() - start_time
    
    # Measure DDPM sampling time
    torch.cuda.empty_cache()
    start_time = time.time()
    samples, _, _, _ = score_model.ddpm_sampler(
        args.num_steps, true_target_dist, num_samples=args.batch_size,
        progress_bar=False
    )
    sampling_time = time.time() - start_time
    
    # Calculate operations per step
    ops_per_step = total_params * 2  # Forward pass operations (approximately)
    total_ops = ops_per_step * args.num_steps * args.num_samples
    gflops = total_ops / 10**9
    
    print(f"\nPerformance Metrics:")
    print(f"Score function evaluation time for {args.num_samples} samples: {score_time:.4f} seconds")
    print(f"Average time per batch: {score_time / num_batches:.4f} seconds")
    print(f"DDPM sampling time for {args.batch_size} samples with {args.num_steps} steps: {sampling_time:.4f} seconds")
    print(f"Estimated time for {args.num_samples} samples: {sampling_time * (args.num_samples/args.batch_size):.4f} seconds")
    print(f"Estimated GFLOPs for sampling {args.num_samples} samples: {gflops:.4f}")

if __name__ == '__main__':
    main() 