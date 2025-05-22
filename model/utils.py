import torch
import torch.nn as nn
import math
import numpy as np

class TimeEmbedding_Diffusion(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

def assert_mean_zero(xt, tol: float = 1e-3):
    # xt shape: (batch, n_particles, n_dim)
    mean    = xt.mean(dim=(1,2))             # → (batch,)
    max_abs = mean.abs().max()               # → scalar tensor

    # pure‐tensor comparison
    if not torch.all(max_abs < tol):
        # you can still .item() here if you really need a Python float,
        # but only in the *error* branch which we rarely hit
        raise RuntimeError(f"New mean is not zero: {max_abs.item():.3e}")


def remove_mean(x):
    assert len(x.size()) == 3, 'Input tensor should have shape [batch, n_atoms, n_dimensions]'
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

def center_gravity_zero_gaussian_log_likelihood(mu, sigma2, x):
    assert len(x.size()) == 3
    assert mu.shape == x.shape
    assert len(sigma2.size()) == 0, 'Sigma should be a scalar'
    B, N, D = x.size()
    assert_mean_zero(x)
    assert_mean_zero(mu)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch((x - mu).pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * torch.log(2 * torch.pi * sigma2)
    log_px = -0.5 * (1 / sigma2) * r2 + log_normalizing_constant

    return log_px

def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = result.scatter_add(0, segment_ids, data)
    return result

def create_edges(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    return [torch.LongTensor(rows), torch.LongTensor(cols)]

def cast_edges2batch(edges_dict, edges, n_batch, n_nodes):
    if n_batch not in edges_dict:
        edges_dict = {}
        rows, cols = edges
        rows_total, cols_total = [], []
        for i in range(n_batch):
            rows_total.append(rows + i * n_nodes)
            cols_total.append(cols + i * n_nodes)
        rows_total = torch.cat(rows_total)
        cols_total = torch.cat(cols_total)

        edges_dict[n_batch] = [rows_total, cols_total]
    return edges_dict[n_batch]

def center_of_gravity_gaussian_log_likelihood_full_cov(mu, Sigma, x, dataset='dw4'):
    """
    Computes the log-likelihood of a zero center-of-gravity Gaussian distribution with 
    full covariance matrix Sigma for data points x.

    The points x (and the mean mu) are assumed to lie in the subspace of ℝ^(M×n) defined by
        sum_{i=1}^{M} x_i = 0,
    so that the effective dimensionality is (M-1)*n.
    
    The density is given by:
    
         N(x | μ, Σ) = [ (2π)^{(M-1)n} det(Σ) ]^(-1/2)
                         exp{ -1/2 (x-μ)^T Σ^(-1) (x-μ) }
    
    Parameters:
      mu    : Tensor of shape (B, M, n), the mean (which must also satisfy center-of-gravity zero).
      Sigma : Tensor of shape ((M-1)*n, (M-1)*n), the full covariance matrix (must be positive definite).
      x     : Tensor of shape (B, M, n), the data (which must be center-of-gravity zero).
    
    Returns:
      log_px : Tensor of shape (B,) containing the log-likelihood of each batch element.
    """
    if dataset == 'dw4':
        projection_matrix = R_dw4
    elif dataset == 'lj13':
        projection_matrix = R_lj13
    elif dataset == 'lj55':
        projection_matrix = R_lj55
    elif dataset == 'aldp':
        projection_matrix = R_aldp

    assert_mean_zero(mu)
    assert_mean_zero(x)

    # Construct the orthonormal basis R for the subspace.
    # Expect R to have shape: (M * n, (M-1) * n)
    # projection_matrix = construct_R(_n_particles, _n_dimension, device=mu.device, dtype=torch.float64)

    B, M, n = x.shape  # M: number of points, n: dimensionality of each point

    mu_proj = mu.reshape(B, -1) @ projection_matrix
    x_proj = x.reshape(B, -1) @ projection_matrix
    Sigma_proj = Sigma

    mvn = torch.distributions.MultivariateNormal(mu_proj, covariance_matrix=Sigma_proj)
    log_px = mvn.log_prob(x_proj)

    return log_px

def sample_center_of_gravity_gaussian_full_cov(mu, Sigma_proj, dataset='dw4'):
    """
    Sample from a zero center-of-gravity Gaussian distribution with full covariance matrix Sigma_proj.
    
    The points are assumed to lie in the subspace of ℝ^(M×n) defined by
        sum_{i=1}^{M} x_i = 0,
    so that the effective dimensionality is (M-1)*n.
    
    Parameters:
      mu         : Tensor of shape (B, M, n), the mean (which must also satisfy center-of-gravity zero).
      Sigma_proj : Tensor of shape ((M-1)*n, (M-1)*n) or (B, (M-1)*n, (M-1)*n), 
                   the full covariance matrix in the reduced coordinates.
      dataset    : Identifier for the dataset, which is used to determine _n_particles and _n_dimension.

    Returns:
      x          : Tensor of shape (B, M, n) containing the generated samples.
    """
    if dataset == 'dw4':
        _n_particles = 4
        _n_dimension = 2
        projection_matrix = R_dw4
    elif dataset == 'lj13':
        _n_particles = 13
        _n_dimension = 3
        projection_matrix = R_lj13
    elif dataset == 'lj55':
        _n_particles = 55
        _n_dimension = 3
        projection_matrix = R_lj55
    elif dataset == 'aldp':
        _n_particles = 22
        _n_dimension = 3
        projection_matrix = R_aldp

    # Ensure the mean is centered.
    assert_mean_zero(mu)

    B = mu.size(0)  # batch size

    # Flatten mu (from shape (B, M, n)) to shape (B, M*n)
    mu_flat = mu.reshape(B, -1)

    # Express mu in the reduced coordinates (R provides a coordinate system for the subspace):
    # mu_proj will have shape (B, (M-1)*n)
    mu_proj = mu_flat @ projection_matrix

    # Check if Sigma_proj is batched
    is_batched = len(Sigma_proj.shape) == 3
    
    if is_batched:
        # For batched covariance matrices (B, (M-1)*n, (M-1)*n)
        assert Sigma_proj.shape[0] == B, "Batch size of Sigma_proj must match batch size of mu"
        assert Sigma_proj.shape[1] == Sigma_proj.shape[2], "Covariance matrix must be square"
        assert Sigma_proj.shape[1] == mu_proj.shape[1], "Covariance dimensions must match mean dimensions"
        
        # Compute the Cholesky factor L for each matrix in the batch
        # L will have shape (B, (M-1)*n, (M-1)*n)
        L = torch.linalg.cholesky(Sigma_proj)
        
        # Generate standard normal samples in the reduced coordinate space (shape: (B, (M-1)*n))
        z = torch.randn_like(mu_proj)
        
        # For batched operations: need to do batch matrix multiplication
        # bmm expects 3D tensors, so reshape z to (B, 1, (M-1)*n)
        z_reshaped = z.unsqueeze(1)
        # L.transpose(-2, -1) gives batch transpose of the last two dimensions
        x_proj = mu_proj + torch.bmm(z_reshaped, L.transpose(-2, -1)).squeeze(1)
    else:
        # Original non-batched version
        # Compute the Cholesky factor L of Sigma_proj: L has shape ((M-1)*n, (M-1)*n)
        L = torch.linalg.cholesky(Sigma_proj)

        # Generate standard normal samples in the reduced coordinate space (shape: (B, (M-1)*n))
        z = torch.randn_like(mu_proj)

        # Compute the sample: note the multiplication order—z @ L.T gives shape (B, (M-1)*n)
        x_proj = mu_proj + z @ L.T

    # Map the reduced coordinate sample back to the original ambient space.
    # x_proj has shape (B, (M-1)*n) and projection_matrix.T has shape ((M-1)*n, M*n),
    # so the product has shape (B, M*n)
    x_flat = x_proj @ projection_matrix.T

    # Reshape back to (B, M, n)
    x = x_flat.reshape(B, _n_particles, _n_dimension)
    
    x = remove_mean(x)  # for safety

    return x

def construct_VN(n_particles, device=None, dtype=torch.float32):
    """
    Constructs an orthonormal basis for the subspace of R^N 
    defined by sum(v)=0. The returned matrix has shape (n_particles, n_particles-1)
    and satisfies V_N^T V_N = I_{n_particles-1} and
    V_N V_N^T = I_{n_particles} - (1/n_particles) 1_N 1_N^T.
    """
    if device is None:
        device = torch.device('cpu')
    
    I_N = torch.eye(n_particles, device=device, dtype=dtype)
    ones = torch.ones((n_particles, 1), device=device, dtype=dtype)
    
    # Create basis vectors: e_i - (1/n_particles)*ones for i = 1,...,n_particles-1
    basis_list = []
    for i in range(1, n_particles):
        e_i = I_N[:, i:i+1]  # shape: (n_particles, 1)
        v_i = e_i - ones / n_particles
        basis_list.append(v_i)
    
    A = torch.cat(basis_list, dim=1)  # shape: (n_particles, n_particles-1)
    
    # Orthonormalize using QR decomposition.
    Q, _ = torch.linalg.qr(A, mode='reduced')  # Q has shape: (n_particles, n_particles-1)
    return Q

def construct_R(n_particles, n_dim, device=None, dtype=torch.float32, seed=0):
    """
    Constructs an orthonormal basis for the zero center-of-mass subspace in R^(n_particles x n_dim)
    (where n_particles is the number of points and n_dim is the spatial dimension).
    
    Returns R of shape (n_particles*n_dim, (n_particles-1)*n_dim) such that
      R^T R = I and 
      R R^T = kron(I_{n_particles} - (1/n_particles)1_N1_N^T, I_{n_dim}),
    which is the correct projector when flattening a tensor of shape (B, n_particles, n_dim)
    in row–major order.
    """
    if device is None:
        device = torch.device('cpu')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Construct the orthonormal basis for the N-space (points dimension)
    # Use high precision for calculations
    V_N = construct_VN(n_particles, device=device, dtype=torch.float64).contiguous()  # shape: (n_particles, n_particles-1)
    I_dim = torch.eye(n_dim, device=device, dtype=torch.float64).contiguous()  # shape: (n_dim, n_dim)
    
    # IMPORTANT: Use kron(V_N, I_dim) to be consistent with row-major flattening.
    R = torch.kron(V_N, I_dim)  # shape: (n_particles*n_dim, (n_particles-1)*n_dim)
    
    # Convert to the specified dtype before returning
    return R.to(dtype=dtype)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R_dw4 = construct_R(4, 2, device=device)
R_lj13 = construct_R(13, 3, device=device)
R_lj55 = construct_R(55, 3, device=device)
R_aldp = construct_R(22, 3, device=device)


def mvn_log_density(mean, cov, samples):
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    log_density = mvn.log_prob(samples)
    return log_density


def compute_forward_ess(log_w):
    # Ensure num_samples is a float tensor in order to compute logarithms correctly.
    num_samples = log_w.shape[0]
    n = torch.tensor(num_samples, dtype=log_w.dtype, device=log_w.device)

    # Compute logsumexp values in a numerically stable manner:
    lse_w = torch.logsumexp(log_w, dim=0)        # log(sum(exp(log_w)))
    lse_neg_w = torch.logsumexp(-log_w, dim=0)     # log(sum(exp(-log_w)))

    # Calculate log of forward_ess using the relation:
    # log(forward_ess) = 3*log(n) - (logsumexp(log_w) + logsumexp(-log_w))
    log_forward_ess = 3 * torch.log(n) - (lse_w + lse_neg_w)

    # Return forward ESS
    forward_ess = torch.exp(log_forward_ess)
    return forward_ess

def compute_reverse_ess(log_w):
    w = torch.exp(log_w - torch.max(log_w))
    w = w / torch.sum(w)

    reverse_ess = 1 / torch.sum(w ** 2)
    return reverse_ess

# test sigma2_mat is invariant to rotation and permutation
def test_invariance(x, sigma2_mat, dataset, projection_matrix=None):
    """
    Test if sigma2_mat is invariant to rotation and permutation of the molecule.
    
    Args:
        x: Input tensor of shape [batch, n_particles, n_dimensions]
        sigma2_mat: Covariance matrix to test
        dataset: Dataset name to get the right projection matrix
        projection_matrix: Optional pre-computed projection matrix
    
    Returns:
        max_diff_rot: Maximum difference after rotation
        max_diff_perm: Maximum difference after permutation
    """
    if projection_matrix is None:
        if dataset == 'dw4':
            projection_matrix = R_dw4.to(x.device)
        elif dataset == 'lj13':
            projection_matrix = R_lj13.to(x.device)
        elif dataset == 'lj55':
            projection_matrix = R_lj55.to(x.device)
        elif dataset == 'aldp':
            projection_matrix = R_aldp.to(x.device)
    
    # Original projection
    x_flat = x.reshape(x.size(0), -1)
    x_proj = x_flat @ projection_matrix
    
    # 1. Test rotation invariance
    # Create random rotation matrix (3D)
    batch_size = x.size(0)
    n_dim = x.size(2)
    n_particles = x.size(1)
    
    if n_dim == 3:
        # Generate random rotation matrix for 3D
        theta = torch.rand(batch_size) * 2 * torch.pi  # Random angle
        phi = torch.rand(batch_size) * 2 * torch.pi    # Random angle
        psi = torch.rand(batch_size) * 2 * torch.pi    # Random angle
        
        # Rotation matrices around x, y, z axes
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        
        # Create batch of rotation matrices
        rot_matrices = []
        for i in range(batch_size):
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(theta[i]), -torch.sin(theta[i])],
                [0, torch.sin(theta[i]), torch.cos(theta[i])]
            ], device=x.device)
            
            Ry = torch.tensor([
                [torch.cos(phi[i]), 0, torch.sin(phi[i])],
                [0, 1, 0],
                [-torch.sin(phi[i]), 0, torch.cos(phi[i])]
            ], device=x.device)
            
            Rz = torch.tensor([
                [torch.cos(psi[i]), -torch.sin(psi[i]), 0],
                [torch.sin(psi[i]), torch.cos(psi[i]), 0],
                [0, 0, 1]
            ], device=x.device)
            
            # Complete rotation matrix
            R = Rz @ Ry @ Rx
            rot_matrices.append(R)
            
    elif n_dim == 2:
        # For 2D, just use a simple rotation matrix
        theta = torch.rand(batch_size) * 2 * torch.pi
        rot_matrices = []
        for i in range(batch_size):
            R = torch.tensor([
                [torch.cos(theta[i]), -torch.sin(theta[i])],
                [torch.sin(theta[i]), torch.cos(theta[i])]
            ], device=x.device)
            rot_matrices.append(R)
    else:
        raise ValueError(f"Unsupported dimension: {n_dim}")
    
    # Apply rotation to each particle in each batch
    x_rotated = torch.zeros_like(x)
    for b in range(batch_size):
        R = rot_matrices[b]
        for p in range(x.size(1)):
            x_rotated[b, p] = x[b, p] @ R
    
    # Ensure center of gravity is zero
    x_rotated = remove_mean(x_rotated)
    
    # Project rotated coordinates
    x_rotated_flat = x_rotated.reshape(x.size(0), -1)
    x_rotated_proj = x_rotated_flat @ projection_matrix
    
    # Check if covariance matrix gives same log-likelihood
    mvn_orig = torch.distributions.MultivariateNormal(
        x_proj, covariance_matrix=sigma2_mat)
    mvn_rot = torch.distributions.MultivariateNormal(
        x_rotated_proj, covariance_matrix=sigma2_mat)
    
    log_prob_orig = mvn_orig.log_prob(x_proj)
    log_prob_rot = mvn_rot.log_prob(x_rotated_proj)
    
    max_diff_rot = torch.max(torch.abs(log_prob_orig - log_prob_rot))
    
    # 2. Test permutation invariance
    # Create random permutation for each batch
    x_permuted = torch.zeros_like(x)
    for b in range(batch_size):
        perm_idx = torch.randperm(x.size(1))
        x_permuted[b] = x[b, perm_idx]
    
    # Ensure center of gravity is zero
    x_permuted = remove_mean(x_permuted)
    
    # Project permuted coordinates
    x_permuted_flat = x_permuted.reshape(x.size(0), -1)
    x_permuted_proj = x_permuted_flat @ projection_matrix
    
    # Check log-likelihood after permutation
    mvn_perm = torch.distributions.MultivariateNormal(
        x_permuted_proj, covariance_matrix=sigma2_mat)
    
    log_prob_perm = mvn_perm.log_prob(x_permuted_proj)
    
    max_diff_perm = torch.max(torch.abs(log_prob_orig - log_prob_perm))
    
    # 3. Check the equivariance property: projection(L_full) * covariance * projection(L_full)^T = covariance
    # For rotation
    # Create full-dimensional rotation matrix (block diagonal)
    L_full_rot = torch.zeros(batch_size, n_particles * n_dim, n_particles * n_dim, device=x.device)
    
    for b in range(batch_size):
        R = rot_matrices[b]
        # Create block diagonal matrix with R repeated for each particle
        for p in range(n_particles):
            start_idx = p * n_dim
            end_idx = start_idx + n_dim
            L_full_rot[b, start_idx:end_idx, start_idx:end_idx] = R
    
    # Take just the first batch element for testing
    L_full_b = L_full_rot[0]
    
    # Compute projection(L_full)
    proj_L_full = projection_matrix.T @ L_full_b @ projection_matrix
    
    # Check equivariance: proj_L_full * sigma2_mat * proj_L_full.T = sigma2_mat
    transformed_cov = proj_L_full @ sigma2_mat @ proj_L_full.T
    cov_diff_rot = torch.norm(transformed_cov - sigma2_mat) / torch.norm(sigma2_mat)
    
    # For permutation (similar approach)
    # Create full permutation matrix for one batch example
    perm_idx = torch.randperm(n_particles)
    L_full_perm = torch.zeros(n_particles * n_dim, n_particles * n_dim, device=x.device)
    
    for i in range(n_particles):
        for j in range(n_dim):
            src_idx = i * n_dim + j
            tgt_idx = perm_idx[i] * n_dim + j
            L_full_perm[tgt_idx, src_idx] = 1.0
    
    # Compute projection(L_full) for permutation
    proj_L_full_perm = projection_matrix.T @ L_full_perm @ projection_matrix
    
    # Check equivariance
    transformed_cov_perm = proj_L_full_perm @ sigma2_mat @ proj_L_full_perm.T
    cov_diff_perm = torch.norm(transformed_cov_perm - sigma2_mat) / torch.norm(sigma2_mat)
    
    print(f"Rotation equivariance check: {cov_diff_rot.item():.6e}")
    print(f"Permutation equivariance check: {cov_diff_perm.item():.6e}")
    
    return max_diff_rot, max_diff_perm


if __name__ == '__main__':
    # Example usage:
    m = 5  # number of points (change as needed)
    N = 2
    device = torch.device('cpu')
    R = construct_R(m, N, device=device)
    print("R shape:", R.shape)
    # Verify orthonormality:
    print("R^T R:\n", R.T @ R)  # should be close to the identity matrix

    I_m = torch.eye(m, device=device)
    ones_m = torch.ones(m, m, device=device) / m
    Q1 = I_m - ones_m  # projector for one coordinate
    Q_theory = torch.kron(Q1, torch.eye(N, device=device))

    diff = Q_theory - R @ R.T
    print("Max difference:", torch.max(torch.abs(diff)))

    x = torch.randn(100, m, N, device=device)
    x_proj = remove_mean(x)

    diff = x_proj.reshape(100, -1) - x_proj.reshape(100, -1) @ (R @ R.T)
    print("Max difference:", torch.max(torch.abs(diff)))

    diff2 = x_proj.reshape(100, -1) - x_proj.reshape(100, -1) @ Q_theory
    print("Max difference:", torch.max(torch.abs(diff2)))