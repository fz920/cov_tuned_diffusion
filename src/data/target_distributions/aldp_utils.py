import os
import torch
import numpy as np
import torch.nn as nn

import boltzgen as bg
import mdtraj
import matplotlib as mpl
from matplotlib import pyplot as plt
from openmmtools.testsystems import AlanineDipeptideVacuum
from openmmtools import testsystems
from simtk import openmm as mm
from simtk import unit
from openmm import app
import tempfile

import abc
from typing import Optional, Dict

class TargetDistribution(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn = None,
                            batch_size: Optional[int] = None) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.
            batch_size: If performance metrics are aggregated over many points that require network
                forward passes, batch_size ensures that the forward passes don't overload GPU
                memory by doing all the points together.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError


    def sample(self, shape):
        raise NotImplementedError


class AldpBoltzmann(nn.Module, TargetDistribution):
    def __init__(self, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='internal',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
                 env='vacuum'):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """
        super(AldpBoltzmann, self).__init__()

        # Define molecule parameters
        ndim = 66
        if transform == 'mixed':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [6, 8, 9, 10, 14]
        # elif transform == 'internal':
        elif transform == 'internal' or transform == 'cartesian':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [8, 6, 14]

        # System setup
        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            sim = app.Simulation(system.topology, system.system,
                                mm.LangevinIntegrator(temperature * unit.kelvin,
                                                      1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                                platform=mm.Platform.getPlatformByName('Reference'))
            sim.context.setPositions(system.positions)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + '/aldp.pt'
            torch.save(torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path)

            del (sim)

        if data_path[-2:] == 'h5':
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == 'pt':
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError('Loading data or this format is not implemented.')

        # Set distribution
        mode = "mixed" if transform == 'mixed' else "internal"
        self.coordinate_transform = bg.flows.CoordinateTransform(transform_data,
                                        ndim, z_matrix, cart_indices, mode=mode,
                                        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
                                        shift_dih_params=shift_dih_params,
                                        default_std=default_std)

        if n_threads > 1:
            if transform == 'cartesian':
                self.p = bg.distributions.BoltzmannParallel(system, temperature, 
                                energy_cut=energy_cut, energy_max=energy_max, n_threads=n_threads)
            else:
                self.p = bg.distributions.TransformedBoltzmannParallel(system,
                                temperature, energy_cut=energy_cut, energy_max=energy_max,
                                transform=self.coordinate_transform, n_threads=n_threads)
        else:
            if transform == 'cartesian':
                self.p = bg.distributions.Boltzmann(system, temperature, 
                    energy_cut=energy_cut, energy_max=energy_max)
            else:
                self.p = bg.distributions.TransformedBoltzmann(sim.context,
                                temperature, energy_cut=energy_cut, energy_max=energy_max,
                                transform=self.coordinate_transform)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}

def project_samples_to_constraints(samples, transform, batch_size=1000):
    """
    Project samples to adhere to constraints
    """
    n_batches = int(np.ceil(len(samples) / batch_size))
    for i in range(n_batches):
        if i == n_batches - 1:
            end = len(samples)
        else:
            end = (i + 1) * batch_size
        z = samples[(i * batch_size):end, :]
        x, _ = transform(z.double())
        z, _ = transform.inverse(x)
        samples[(i * batch_size):end, :] = z
    return samples

def evaluate_aldp(z_sample, z_test, log_prob, transform,
                  iter, metric_dir=None, plot_dir=None,
                  batch_size=1000):
    """
    Evaluate model of the Boltzmann distribution of the
    Alanine Dipeptide
    :param z_sample: Samples from the model
    :param z_test: Test data
    :param log_prob: Function to evaluate the log
    probability
    :param transform: Coordinate transformation
    :param iter: Current iteration count used for
    labeling of the generated files
    :param metric_dir: Directory where to store
    evaluation metrics
    :param plot_dir: Directory where to store plots
    :param batch_size: Batch size when processing
    the data
    """
    # Get mode of transform
    if isinstance(transform.transform, bg.mixed.MixedTransform):
        transform_mode = 'mixed'
    elif isinstance(transform.transform, bg.internal.CompleteInternalCoordinateTransform):
        transform_mode = 'internal'
    else:
        raise NotImplementedError('The evaluation is not implemented '
                                  'for this transform.')
    # Determine likelihood of test data and transform it
    # z_d_np = z_test.cpu().data.numpy()
    # x_d_np = np.zeros((0, 66))
    x_d_np = []
    # log_p_sum = 0
    n_batches = int(np.ceil(len(z_test) / batch_size))
    # for i in tqdm(range(n_batches)):
    for i in range(n_batches):
        if i == n_batches - 1:
            end = len(z_test)
        else:
            end = (i + 1) * batch_size
        z = z_test[(i * batch_size):end, :]
        x, _ = transform(z.double())
        # x_d_np = np.concatenate((x_d_np, x.cpu().data.numpy()))
        x_d_np.append(x.cpu().data.numpy())
    x_d_np = np.concatenate(x_d_np, axis=0)


    #     log_p = log_prob(z)
    #     log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
    # log_p_avg = log_p_sum.cpu().data.numpy() / len(z_test)

    # Transform samples
    # z_np = np.zeros((0, 60))
    # x_np = np.zeros((0, 66))
    z_np = []
    x_np = []
    n_batches = int(np.ceil(len(z_sample) / batch_size))
    # for i in tqdm(range(n_batches)):
    for i in range(n_batches):
        if i == n_batches - 1:
            end = len(z_sample)
        else:
            end = (i + 1) * batch_size
        z = z_sample[(i * batch_size):end, :]
        x, _ = transform(z.double())
        # x_np = np.concatenate((x_np, x.cpu().data.numpy()))
        x_np.append(x.cpu().data.numpy())
        z, _ = transform.inverse(x)
        # z_np = np.concatenate((z_np, z.cpu().data.numpy()))
        z_np.append(z.cpu().data.numpy())
    x_np = np.concatenate(x_np, axis=0)
    z_np = np.concatenate(z_np, axis=0)


    # Estimate density of marginals
    nbins = 200
    hist_range = [-5, 5]
    ndims = z_np.shape[1]

    hists_test = np.zeros((nbins, ndims))
    hists_gen = np.zeros((nbins, ndims))

    # for i in tqdm(range(ndims)):
    for i in range(ndims):
        if z_test[:, i].cpu().data.numpy().max() > 5 or z_test[:, i].cpu().data.numpy().min() < -5:
            # print out the quantiles and NaN proportion
            data = z_test[:, i].cpu().data.numpy()
            nan_count = np.isnan(data).sum()
            nan_proportion = nan_count / len(data)
            print("In plots:")
            print(f"Data not within [-5, 5]. Quantiles of z_test[:, {i}]: {np.quantile(data, [0.01, 0.1, 0.9, 0.99])}")
            print(f"NaN count: {nan_count}, proportion: {nan_proportion:.4f}")
            print()
        htest, _ = np.histogram(z_test[:, i].cpu().data.numpy(), nbins, range=hist_range, density=True);
        hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True);
        hists_test[:, i] = htest
        hists_gen[:, i] = hgen

    # # Compute KLD of marginals
    eps = 1e-10
    kld_unscaled = np.sum(hists_test * np.log((hists_test + eps) / (hists_gen + eps)), axis=0)
    kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins

    # Split KLD into groups
    ncarts = transform.transform.len_cart_inds
    permute_inv = transform.transform.permute_inv.cpu().data.numpy()
    bond_ind = transform.transform.ic_transform.bond_indices.cpu().data.numpy()
    angle_ind = transform.transform.ic_transform.angle_indices.cpu().data.numpy()
    dih_ind = transform.transform.ic_transform.dih_indices.cpu().data.numpy()

    kld_cart = kld[:(3 * ncarts - 6)]
    kld_ = np.concatenate([kld[:(3 * ncarts - 6)], np.zeros(6), kld[(3 * ncarts - 6):]])
    kld_ = kld_[permute_inv]
    kld_bond = kld_[bond_ind]
    kld_angle = kld_[angle_ind]
    kld_dih = kld_[dih_ind]
    if transform_mode == 'internal':
        kld_bond = np.concatenate((kld_cart[:2], kld_bond))
        kld_angle = np.concatenate((kld_cart[2:], kld_angle))

    # Compute Ramachandran plot angles
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    test_traj = mdtraj.Trajectory(x_d_np.reshape(-1, 22, 3), topology)
    sampled_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
    psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
    not_nan = np.logical_not(is_nan)
    psi_d = psi_d[not_nan]
    phi_d = phi_d[not_nan]
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    not_nan = np.logical_not(is_nan)
    psi = psi[not_nan]
    phi = phi[not_nan]

    # Compute KLD of phi and psi
    htest_phi, _ = np.histogram(phi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True);
    kld_phi = np.sum(htest_phi * np.log((htest_phi + eps) / (hgen_phi + eps))) \
              * 2 * np.pi / nbins
    htest_psi, _ = np.histogram(psi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True);
    kld_psi = np.sum(htest_psi * np.log((htest_psi + eps) / (hgen_psi + eps))) \
              * 2 * np.pi / nbins

    # Compute KLD of Ramachandran plot angles
    nbins_ram = 64
    eps_ram = 1e-10
    hist_ram_test = np.histogram2d(phi_d, psi_d, nbins_ram,
                                   range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                   density=True)[0]
    hist_ram_gen = np.histogram2d(phi, psi, nbins_ram,
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                  density=True)[0]
    kld_ram = np.sum(hist_ram_test * np.log((hist_ram_test + eps_ram)
                                            / (hist_ram_gen + eps_ram))) \
              * (2 * np.pi / nbins_ram) ** 2

    # Save metrics
    if metric_dir is not None:
        # Calculate and save KLD stats of marginals
        kld = (kld_bond, kld_angle, kld_dih)
        kld_labels = ['bond', 'angle', 'dih']
        if transform_mode == 'mixed':
            kld = (kld_cart,) + kld
            kld_labels = ['cart'] + kld_labels
        kld_ = np.concatenate(kld)
        kld_append = np.array([[iter + 1, np.median(kld_), np.mean(kld_)]])
        kld_path = os.path.join(metric_dir, 'kld.csv')
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=',',
                   header='it,kld_median,kld_mean', comments='')
        for kld_label, kld_ in zip(kld_labels, kld):
            kld_append = np.concatenate([np.array([iter + 1, np.median(kld_), np.mean(kld_)]), kld_])
            kld_append = kld_append[None, :]
            kld_path = os.path.join(metric_dir, 'kld_' + kld_label + '.csv')
            if os.path.exists(kld_path):
                kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
                if len(kld_hist.shape) == 1:
                    kld_hist = kld_hist[None, :]
                kld_hist = np.concatenate([kld_hist, kld_append])
            else:
                kld_hist = kld_append
            header = 'it,kld_median,kld_mean'
            for kld_ind in range(len(kld_)):
                header += ',kld' + str(kld_ind)
            np.savetxt(kld_path, kld_hist, delimiter=',',
                       header=header, comments='')

        # Save KLD of Ramachandran and log_p
        kld_path = os.path.join(metric_dir, 'kld_ram.csv')
        kld_append = np.array([[iter + 1, kld_phi, kld_psi, kld_ram]])
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=',',
                   header='it,kld_phi,kld_psi,kld_ram', comments='')

        # # Save log probability
        # log_p_append = np.array([[iter + 1, log_p_avg]])
        # log_p_path = os.path.join(metric_dir, 'log_p_test.csv')
        # if os.path.exists(log_p_path):
        #     log_p_hist = np.loadtxt(log_p_path, skiprows=1, delimiter=',')
        #     if len(log_p_hist.shape) == 1:
        #         log_p_hist = log_p_hist[None, :]
        #     log_p_hist = np.concatenate([log_p_hist, log_p_append])
        # else:
        #     log_p_hist = log_p_append
        # np.savetxt(log_p_path, log_p_hist, delimiter=',',
        #            header='it,log_p', comments='')

    # Create plots
    if plot_dir is not None:
        # Histograms of the groups
        hists_test_cart = hists_test[:, :(3 * ncarts - 6)]
        hists_test_ = np.concatenate([hists_test[:, :(3 * ncarts - 6)],
                                      np.zeros((nbins, 6)),
                                      hists_test[:, (3 * ncarts - 6):]], axis=1)
        hists_test_ = hists_test_[:, permute_inv]
        hists_test_bond = hists_test_[:, bond_ind]
        hists_test_angle = hists_test_[:, angle_ind]
        hists_test_dih = hists_test_[:, dih_ind]

        hists_gen_cart = hists_gen[:, :(3 * ncarts - 6)]
        hists_gen_ = np.concatenate([hists_gen[:, :(3 * ncarts - 6)],
                                     np.zeros((nbins, 6)),
                                     hists_gen[:, (3 * ncarts - 6):]], axis=1)
        hists_gen_ = hists_gen_[:, permute_inv]
        hists_gen_bond = hists_gen_[:, bond_ind]
        hists_gen_angle = hists_gen_[:, angle_ind]
        hists_gen_dih = hists_gen_[:, dih_ind]

        if transform_mode == 'internal':
            hists_test_bond = np.concatenate((hists_test_cart[:, :2],
                                              hists_test_bond), 1)
            hists_gen_bond = np.concatenate((hists_gen_cart[:, :2],
                                             hists_gen_bond), 1)
            hists_test_angle = np.concatenate((hists_test_cart[:, 2:],
                                               hists_test_angle), 1)
            hists_gen_angle = np.concatenate((hists_gen_cart[:, 2:],
                                              hists_gen_angle), 1)

        label = ['bond', 'angle', 'dih']
        hists_test_list = [hists_test_bond, hists_test_angle,
                           hists_test_dih]
        hists_gen_list = [hists_gen_bond, hists_gen_angle,
                          hists_gen_dih]
        if transform_mode == 'mixed':
            label = ['cart'] + label
            hists_test_list = [hists_test_cart] + hists_test_list
            hists_gen_list = [hists_gen_cart] + hists_gen_list
        x = np.linspace(*hist_range, nbins)
        for i in range(len(label)):
            if transform_mode == 'mixed':
                ncol = 3
                if i == 0:
                    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
                else:
                    fig, ax = plt.subplots(6, 3, figsize=(10, 20))
                    ax[5, 2].set_axis_off()
            elif transform_mode == 'internal':
                ncol = 4
                if i == 0:
                    fig, ax = plt.subplots(6, 4, figsize=(15, 24))
                    for j in range(1, 4):
                        ax[5, j].set_axis_off()
                elif i == 2:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
                    ax[4, 3].set_axis_off()
                else:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
            for j in range(hists_test_list[i].shape[1]):
                ax[j // ncol, j % ncol].plot(x, hists_test_list[i][:, j])
                ax[j // ncol, j % ncol].plot(x, hists_gen_list[i][:, j])
            plt.savefig(os.path.join(plot_dir, 'marginals_%s_%07i.png' % (label[i], iter + 1)))
            plt.close()

        # Plot phi and psi
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        x = np.linspace(-np.pi, np.pi, nbins)
        ax[0].plot(x, htest_phi, linewidth=3)
        ax[0].plot(x, hgen_phi, linewidth=3)
        # np.save("gibbs_phi.npy", hgen_phi)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[0].set_xlabel(r'$\phi$', fontsize=24)
        ax[0].set_ylabel(r'$p(\phi)$', fontsize=24)
        ax[0].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])
        # ax[0].set_yscale("log", base=10)
        # ax[0].set_ylim(1e-5, 1.0)
        ax[1].plot(x, htest_psi, linewidth=3)
        ax[1].plot(x, hgen_psi, linewidth=3)
        # np.save("gibbs_psi.npy", hgen_psi)
        ax[1].tick_params(axis='both', labelsize=20)
        ax[1].set_xlabel(r'$\psi$', fontsize=24)
        ax[1].set_ylabel(r'$p(\psi)$', fontsize=24)
        ax[1].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])
        plt.savefig(os.path.join(plot_dir, 'phi_psi_%07i.png' % (iter + 1)))
        plt.close()

        # Ramachandran plot
        plt.figure(figsize=(12, 10))
        plt.hist2d(phi, psi, bins=100, norm=mpl.colors.LogNorm(vmin=0.0001, vmax=1.0),
                   range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True)
        # plt.colorbar(ticks=mpl.ticker.LogLocator(subs=range(10)))
        plt.xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'], fontsize=30)
        plt.yticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'], fontsize=30)
        plt.xlabel(r'$\phi$', fontsize=50)
        plt.ylabel(r'$\psi$', fontsize=50)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'ramachandran_%07i.png' % (iter + 1)))
        plt.close()


def filter_chirality(x, ind=[17, 26], mean_diff=-0.043, threshold=0.8):
    """
    Filters batch for the L-form
    :param x: Input batch
    :param ind: Indices to be used for determining the chirality
    :param mean_diff: Mean of the difference of the coordinates
    :param threshold: Threshold to be used for splitting
    :return: Returns indices of batch, where L-form is present
    """
    diff_ = torch.column_stack((x[:, ind[0]] - x[:, ind[1]],
                                x[:, ind[0]] - x[:, ind[1]] + 2 * np.pi,
                                x[:, ind[0]] - x[:, ind[1]] - 2 * np.pi))
    min_diff_ind = torch.min(torch.abs(diff_), 1).indices
    diff = diff_[torch.arange(x.shape[0]), min_diff_ind]
    ind = torch.abs(diff - mean_diff) < threshold
    return ind
