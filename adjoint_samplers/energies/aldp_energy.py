"""
adjoint_samplers/energies/aldp_energy.py

Alanine dipeptide energy via OpenMM.
22 atoms x 3D = 66 dimensions.

The energy is returned in units of kBT (dimensionless) so that
p(x) ∝ exp(-E(x)) is the Boltzmann distribution at 300K.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


class AlanineDipeptideEnergy(BaseEnergy):
    """Alanine dipeptide potential energy via OpenMM.

    Coordinates are in nanometers, flattened to (B, 66).
    Energy is returned in units of kBT at the specified temperature.

    Args:
        dim: must be 66
        temperature: in Kelvin (default 300)
        platform: OpenMM platform ('CPU' or 'CUDA')
        device: torch device for output tensors
        max_energy: clamp energy to this value (in kBT) to avoid NaN
    """

    def __init__(self, dim=66, temperature=300.0, platform='CPU',
                 device='cpu', max_energy=1000.0):
        super().__init__("aldp", dim)
        assert dim == 66, "Alanine dipeptide has 22 atoms x 3 = 66 dimensions"
        assert HAS_OPENMM, "OpenMM not installed. Run: conda install -c conda-forge openmm"

        self.temperature = temperature
        self.device = device
        self.n_particles = 22
        self.n_spatial_dim = 3
        self.max_energy = max_energy

        # kBT in kJ/mol
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.kBT = (kB * temperature * unit.kelvin).value_in_unit(
            unit.kilojoule_per_mole
        )

        # Build OpenMM system
        self._build_system(platform)

    def _build_system(self, platform_name):
        """Build OpenMM system for alanine dipeptide in vacuum."""
        import os
        import tempfile

        pdb_string = self._get_aldp_pdb()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            pdb_path = f.name

        pdb = app.PDBFile(pdb_path)
        os.unlink(pdb_path)

        # Force field
        forcefield = app.ForceField('amber14-all.xml')

        # Create system (vacuum, no cutoff, no constraints)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )

        # Create integrator (not used for dynamics, just needed for Context)
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)

        # Platform
        platform = openmm.Platform.getPlatformByName(platform_name)
        self.context = openmm.Context(system, integrator, platform)

        # Store initial positions for reference
        self.init_positions = pdb.positions

        # Store initial positions as numpy array (nm, flattened)
        init_pos_np = np.array(
            pdb.positions.value_in_unit(unit.nanometers)
        ).flatten()
        self.init_positions_flat = init_pos_np  # (66,)

        # Store topology for dihedral computation
        self.topology = pdb.topology

    def _get_aldp_pdb(self):
        """Return a standard alanine dipeptide PDB string.

        ACE-ALA-NME (22 atoms) in vacuum.
        Downloads from OpenMM's test data on GitHub.
        Falls back to local file if download fails.
        """
        import os

        # Try local file first
        local_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'alanine-dipeptide.pdb'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'alanine-dipeptide-implicit.pdb'),
        ]
        for path in local_paths:
            path = os.path.abspath(path)
            if os.path.exists(path):
                with open(path) as f:
                    return f.read()

        # Download from GitHub
        try:
            import urllib.request
            url = (
                "https://raw.githubusercontent.com/openmm/openmm/master/"
                "wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb"
            )
            response = urllib.request.urlopen(url, timeout=30)
            pdb_text = response.read().decode('utf-8')

            # Save locally for future use
            save_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..',
                             'data', 'alanine-dipeptide.pdb')
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(pdb_text)

            return pdb_text
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot find alanine dipeptide PDB ({e}). Download manually from "
                "https://raw.githubusercontent.com/openmm/openmm/master/"
                "wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb "
                "and place it at data/alanine-dipeptide.pdb"
            )

    @torch.no_grad()
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x) / kBT for a batch of configurations.

        Args:
            x: (B, 66) tensor -- flattened Cartesian coordinates in nanometers

        Returns:
            E: (B,) tensor -- dimensionless energy E/kBT
        """
        B = x.shape[0]
        assert x.shape[1] == 66

        x_np = x.detach().cpu().numpy()
        energies = np.zeros(B)

        for i in range(B):
            coords = x_np[i].reshape(self.n_particles, 3)
            self.context.setPositions(coords * unit.nanometers)
            state = self.context.getState(getEnergy=True)
            E_kJ = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            energies[i] = E_kJ / self.kBT

        # Clamp to avoid NaN/Inf from unphysical configurations
        energies = np.clip(energies, -self.max_energy, self.max_energy)

        return torch.tensor(energies, dtype=x.dtype, device=x.device)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score s(x) = -grad E(x)/kBT using OpenMM forces.

        Args:
            x: (B, 66) tensor

        Returns:
            score: (B, 66) tensor -- negative gradient of dimensionless energy
        """
        B = x.shape[0]
        x_np = x.detach().cpu().numpy()
        scores = np.zeros((B, 66))

        for i in range(B):
            coords = x_np[i].reshape(self.n_particles, 3)
            self.context.setPositions(coords * unit.nanometers)
            state = self.context.getState(getForces=True)
            # Forces are -dE/dx in kJ/(mol*nm)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometers
            )
            # score = -grad(E/kBT) = forces / kBT
            scores[i] = forces.flatten() / self.kBT

        return torch.tensor(scores, dtype=x.dtype, device=x.device)

    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        """Override base class: use OpenMM forces directly (no autograd).

        Returns grad E/kBT = -score.
        """
        return -self.score(x)

    def compute_dihedrals(self, x: torch.Tensor):
        """Compute Ramachandran angles (phi, psi) for visualization.

        Args:
            x: (B, 66) tensor

        Returns:
            phi: (B,) tensor -- phi dihedral in radians
            psi: (B,) tensor -- psi dihedral in radians
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError(
                "mdtraj required for dihedral computation: pip install mdtraj"
            )

        B = x.shape[0]
        coords = x.detach().cpu().numpy().reshape(B, self.n_particles, 3)

        # Create mdtraj trajectory
        top = md.Topology.from_openmm(self.topology)
        traj = md.Trajectory(coords, top)

        # Compute phi and psi
        _, phi_angles = md.compute_phi(traj)
        _, psi_angles = md.compute_psi(traj)

        # Alanine dipeptide has 1 phi and 1 psi
        phi = torch.tensor(phi_angles[:, 0], dtype=x.dtype)
        psi = torch.tensor(psi_angles[:, 0], dtype=x.dtype)

        return phi, psi

    def get_init_positions(self):
        """Return minimized PDB coordinates as (66,) numpy array in nm."""
        return self.init_positions_flat.copy()
