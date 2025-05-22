import torch
import argparse
from score_model import ScoreNet
from tqdm import tqdm

from gmm import create_gmm

from utils.path_config import (
    get_config_path, get_model_checkpoint_path, get_params_checkpoint_path,
    get_sample_path, get_figure_path, FIGURES_DIR, CHECKPOINTS_DIR
)

def main():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--train_seed', type=int, default=1)
    parser.add_argument('--train_num_samples', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50000)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--mu', type=float, default=0.9)

    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=256)
    args = parser.parse_args()

    # Set seed and device
    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the GMM
    gmm = create_gmm(args.input_dim, device=device)

    # initialize models
    score_model = ScoreNet(input_dim=args.input_dim, n_layers=args.n_layers,
                           hidden_size=args.hidden_size).to(device)
    score_model_online = ScoreNet(input_dim=args.input_dim, n_layers=args.n_layers,
                           hidden_size=args.hidden_size).to(device)
    score_model_online.load_state_dict(score_model.state_dict())

    T_max = args.num_epochs
    optimizer = torch.optim.AdamW(score_model_online.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    bar = tqdm(range(args.num_epochs))
    loss_hist = []
    for e in bar:
        # get iid samples
        optimizer.zero_grad()
        avg_loss = 0
        x0 = gmm.sample(args.train_num_samples)

        loss = score_model_online.compute_loss(x0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_loss += loss.item()
        loss_hist.append(avg_loss)

        bar.set_description('Average Loss: {:5f}'.format(avg_loss))

        # EMA update
        for param, param_online in zip(score_model.parameters(), score_model_online.parameters()):
            param.data = args.mu * param.data + (1 - args.mu) * param_online.data

        if (e+1) % args.save_freq == 0 or e == args.num_epochs - 1:
            torch.save(score_model.state_dict(), fstr(Path(__file__).parent.parent / 'gmm2_experiments/score/checkpoints/{args.input_dim}D_gmm2_score_ckpt_{args.n_layers}layers_{args.hidden_size}hidden_size.pth'))

if __name__ == '__main__':
    main()
