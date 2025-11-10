import torch
import argparse
import yaml
import os
from tqdm import tqdm

import numpy as np

from cov_tuned_diffusion import ScoreNet, load_dataset
from cov_tuned_diffusion.utils.path_config import get_config_path, get_model_checkpoint_path

def main():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--train_seed', type=int, default=1)
    parser.add_argument('--train_num_samples', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--mu', type=float, default=0.9, help='Exponential moving average decay')
    parser.add_argument('--net', type=str, default='egnn')
    parser.add_argument('--dataset', type=str, default='aldp')
    parser.add_argument('--cont_training', action='store_true', help='Continue training from a checkpoint')
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    # Set seed and device
    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import data
    training_data = load_dataset(args.dataset, device=device)
    total_num_samples = training_data.shape[0]

    # load model configuaration and initialize models
    model_config_path = get_config_path(args.dataset, args.net)
    model_config = yaml.safe_load(open(model_config_path, 'r'))
    score_model = ScoreNet(dataset=args.dataset, device=device, model_config=model_config,
                           net=args.net).to(device)
    score_model_online = ScoreNet(dataset=args.dataset, device=device, model_config=model_config,
                                  net=args.net).to(device)
    score_model_online.load_state_dict(score_model.state_dict())

    # Get checkpoint path and log file path
    checkpoint_path = get_model_checkpoint_path(args.dataset, args.net, "score", args.index)
    
    # Create log file
    log_dir = os.path.dirname(checkpoint_path)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(checkpoint_path))[0]}.txt")

    start_epoch = 0
    T_max = args.num_epochs
    optimizer = torch.optim.AdamW(score_model_online.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    if args.cont_training:
        print(f"Checkpoint found at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        score_model.load_state_dict(checkpoint['model_state_dict'])
        score_model_online.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("Starting from scratch.")
        with open(log_file, 'w') as f:
            f.write(f'Training log for score model {args.index} with {args.net} on {args.dataset} dataset\n')
            f.write(f'Model configuration: {model_config}\n')

    bar = tqdm(range(start_epoch, args.num_epochs), initial=start_epoch, total=args.num_epochs, desc="Training Progress")

    loss_hist = []
    for e in bar:
        # get iid samples
        optimizer.zero_grad()

        x0 = training_data[np.random.choice(total_num_samples, args.train_num_samples, replace=False), :, :]

        loss = score_model_online.compute_loss(x0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())

        bar.set_description('Average Loss: {:5f}'.format(loss.item()))

        # EMA update
        for param, param_online in zip(score_model.parameters(), score_model_online.parameters()):
            param.data = args.mu * param.data + (1 - args.mu) * param_online.data

        if (e+1) % args.save_freq == 0 or e == args.num_epochs - 1:
            torch.save({
                'epoch': e,
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            with open(log_file, 'a') as f:
                f.write(f'Epoch {e+1} - Loss: {loss.item()}\n')

if __name__ == '__main__':
    main()
