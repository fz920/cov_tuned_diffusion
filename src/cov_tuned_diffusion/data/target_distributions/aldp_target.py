from typing import Optional, Tuple
import torch
import numpy as np
import openmm
from openmmtools import testsystems
import multiprocessing as mp
import mdtraj

R = 8.31447e-3

def openmm_energy(x: torch.Tensor, openmm_context, temperature: float = 800.0):
    """Compute the energy of a single configuration using OpenMM.

    Args:
        x: A configuration of shape (n_atoms, 3).
        openmm_context: An OpenMM context.
        temperature: The temperature in Kelvin.

    Returns:
        The energy of the configuration.
    """
    kBT = R * temperature
    x_np = x.detach().cpu().numpy()

    # Handle nans and infinities
    if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
        return torch.tensor(np.nan), torch.tensor(np.nan)
    else:
        openmm_context.setPositions(x_np)
        state = openmm_context.getState(getForces=True, getEnergy=True)

        # Get energy
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole) / kBT

        # Get forces
        force = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer) / kBT

        energy_tensor = torch.tensor(energy, dtype=x.dtype)
        force_tensor = torch.tensor(force, dtype=x.dtype)

        return energy_tensor, force_tensor

def openmm_multi_proc_init(env, temp, plat):
    """
    Method to initialize temperature and openmm context for workers
    of multiprocessing pool
    """
    global temperature_g, openmm_context_g
    temperature_g = temp
    # System setup
    if env == 'vacuum':
        system = testsystems.AlanineDipeptideVacuum(constraints=None)
    elif env == 'implicit':
        system = testsystems.AlanineDipeptideImplicit(constraints=None)
    else:
        raise NotImplementedError('This environment is not implemented.')
    sim = openmm.app.Simulation(system.topology, system.system,
                                openmm.LangevinIntegrator(temp * openmm.unit.kelvin,
                                                          1.0 / openmm.unit.picosecond,
                                                          1.0 * openmm.unit.femtosecond),
                                platform=openmm.Platform.getPlatformByName(plat))
    openmm_context_g = sim.context

def openmm_energy_multi_proc(x: torch.Tensor):
    """Compute the energy of a single configuration using OpenMM using global
    temperature and context.

    Args:
        x: A configuration of shape (n_atoms, 3).

    Returns:
        The energy of the configuration.
    """
    return openmm_energy(x, openmm_context_g, temperature_g)

def openmm_energy_batched(x: torch.Tensor, openmm_context, temperature: float = 800.0):
    """Compute the energy of a batch of configurations using OpenMM.

    Args:
        x: A batch of configurations of shape (n_batch, n_atoms, 3).
        openmm_context: An OpenMM context.
        temperature: The temperature in Kelvin.

    Returns:
        The energy of each configuration in the batch.
    """
    energies = []
    forces = []
    for i in range(x.shape[0]):
        energy, force = openmm_energy(x[i, ...], openmm_context, temperature)
        energies.append(energy)
        forces.append(force)
    return torch.stack(energies), torch.stack(forces)

def openmm_energy_multi_proc_batched(x: torch.Tensor, pool):
    x_list = [x[i, ...] for i in range(x.shape[0])]
    out = pool.map(openmm_energy_multi_proc, x_list)
    energies_out, forces_out = zip(*out)
    energies = torch.stack(energies_out)
    forces = torch.stack(forces_out)
    return energies, forces

def get_log_prob_fn(temperature: float = 800, environment: str = 'implicit', platform: str = 'Reference',
                    scale: Optional[float] = None):
    """Get a function that computes the energy of a batch of configurations.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.
        platform (str, optional): The compute platform that OpenMM shall use. Can be 'Reference', 'CUDA', 'OpenCL',
        and 'CPU'. Defaults to 'Reference'.
        scale (Optional[float], optional): A scaling factor applied to the input batch. Defaults to None.

    Returns:
        A function that computes the energy of a batch of configurations.
    """
    # System setup
    if environment == 'vacuum':
        system = testsystems.AlanineDipeptideVacuum(constraints=None)
    elif environment == 'implicit':
        system = testsystems.AlanineDipeptideImplicit(constraints=None)
    else:
        raise NotImplementedError('This environment is not implemented.')
    assert platform in ['Reference', 'CUDA', 'OpenCL', 'CPU']
    sim = openmm.app.Simulation(system.topology, system.system,
                                openmm.LangevinIntegrator(temperature * openmm.unit.kelvin,
                                                          1. / openmm.unit.picosecond,
                                                          1. * openmm.unit.femtosecond),
                                openmm.Platform.getPlatformByName(platform))

    def log_prob_and_grad(x: torch.Tensor):
        if scale is not None:
            x = x * scale
        if len(x.shape) == 2:
            energy, force = openmm_energy_batched(x[None, ...], sim.context, temperature=temperature)
            return -energy[0, ...], force[0, ...]
        elif len(x.shape) == 3:
            energies, forces = openmm_energy_batched(x, sim.context, temperature=temperature)
            return -energies, forces
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    return log_prob_and_grad

def get_multi_proc_log_prob_fn(temperature: float = 800, environment: str = 'implicit', platform: str = 'Reference',
                               n_threads: Optional[int] = None):
    """Get a function that computes the energy of a batch of configurations via multiprocessing.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.
        n_threads (Optional[int], optional): Number of threads for multiprocessing. If None, the number of CPUs is
        taken. Defaults to None.

    Returns:
        A function that computes the energy of a batch of configurations.
    """
    assert environment in ['implicit', 'vacuum'], 'Environment must be either implicit or vacuum'
    assert platform in ['Reference', 'CUDA', 'OpenCL', 'CPU'], 'Platform must be either Reference, CUDA, OpenCL, or CPU'
    if n_threads is None:
        n_threads = mp.cpu_count()
    # Initialize multiprocessing pool
    # pool = mp.Pool(n_threads, initializer=openmm_multi_proc_init, initargs=(environment, temperature, platform))

    # Define function
    def log_prob_and_grad(x: torch.Tensor):
        with mp.Pool(n_threads, initializer=openmm_multi_proc_init, initargs=(environment, temperature, platform)) as pool:
            if len(x.shape) == 2:
                energy, force = openmm_energy_multi_proc_batched(x[None, ...], pool)
                return -energy[0, ...], force[0, ...]
            elif len(x.shape) == 3:
                energies, forces = openmm_energy_multi_proc_batched(x, pool)
                return -energies, forces
            else:
                raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    return log_prob_and_grad


def load_aldp(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    train_n_points=None,
    val_n_points=None,
    test_n_points=None,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Load the ALDP dataset from given paths.

    Args:
        train_path (Optional[str]): Path to the training dataset.
        val_path (Optional[str]): Path to the validation dataset.
        test_path (Optional[str]): Path to the test dataset.
        train_n_points (Optional[int]): Number of points to load from the training dataset.
        val_n_points (Optional[int]): Number of points to load from the validation dataset.
        test_n_points (Optional[int]): Number of points to load from the test dataset.

    Returns:
        Tuple[Optional[dict], Optional[dict], Optional[dict]]: Loaded training, validation, and test datasets.
    """
    paths = [train_path, val_path, test_path]
    n_points = [train_n_points, val_n_points, test_n_points]
    datasets = [None, None, None]

    for i in range(3):
        if paths[i] is not None:
            traj = mdtraj.load(paths[i])
            features = torch.arange(traj.n_atoms, dtype=torch.int)[:, None]
            positions = torch.tensor(traj.xyz, dtype=torch.float32)
            if n_points[i] is not None:
                positions = positions[:n_points[i]]
            datasets[i] = {
                'positions': positions,
                'features': features.repeat(positions.shape[0], 1, 1),
            }

    return tuple(datasets)


class AldpEnergy:
    def __init__(self, temperature: float = 500.0, environment: str = 'implicit', platform: str = 'Reference',
                 scale: Optional[float] = None, n_threads: Optional[int] = 8):
        self.log_prob_mp_fn = get_multi_proc_log_prob_fn(temperature=temperature, n_threads=n_threads,
                                                         environment=environment, platform=platform)

    def log_prob(self, x: torch.Tensor):
        log_prob, _ = self.log_prob_mp_fn(x.detach().cpu())
        log_prob = log_prob.to(x.device)
        return log_prob

    def energy(self, x: torch.Tensor):
        log_prob, _ = self.log_prob_mp_fn(x.detach().cpu())
        energy = -log_prob.to(x.device)
        return energy
