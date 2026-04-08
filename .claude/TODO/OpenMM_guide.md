Yes, this is manageable. Alanine dipeptide is OpenMM’s hello-world example — there are hundreds of tutorials for it. The energy evaluation is: pass 22-atom Cartesian coordinates to OpenMM, get potential energy back. Here’s the prompt:

I need you to add alanine dipeptide as a benchmark for our SDR-ASBS experiments. This requires OpenMM for energy evaluation. The plan:

## Step 1: Create a Separate Conda Environment

Create a new conda environment specifically for alanine dipeptide experiments. Our main adjoint_samplers code is pure PyTorch — OpenMM is only needed for energy evaluation.

```bash
conda create -n asbs_aldp python=3.10 -y
conda activate asbs_aldp
conda install -c conda-forge openmm -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mdtraj parmed numpy scipy matplotlib hydra-core omegaconf pot
pip install -e .  # install adjoint_samplers in editable mode


Verify OpenMM works:

import openmm
print(openmm.__version__)


Step 2: Implement Alanine Dipeptide Energy
Create adjoint_samplers/energies/aldp_energy.py. This wraps OpenMM to provide energy and gradient evaluations compatible with our BaseEnergy interface.
Alanine dipeptide specs:
	∙	22 atoms, 3D → d = 66 dimensions
	∙	Use the AMBER14 force field (standard for this system)
	∙	Vacuum (no solvent, implicit or explicit) for simplicity — this is what most sampling papers use
	∙	Temperature: 300K (kBT ≈ 2.494 kJ/mol)
	∙	The energy should return E(x) / kBT (dimensionless, matching our Boltzmann convention p ∝ exp(-E))

"""
adjoint_samplers/energies/aldp_energy.py

Alanine dipeptide energy via OpenMM.
22 atoms × 3D = 66 dimensions.

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
    """
    def __init__(self, dim=66, temperature=300.0, platform='CPU', device='cpu'):
        super().__init__("aldp", dim)
        assert dim == 66, "Alanine dipeptide has 22 atoms × 3 = 66 dimensions"
        assert HAS_OPENMM, "OpenMM not installed. Run: conda install -c conda-forge openmm"

        self.temperature = temperature
        self.device = device
        self.n_atoms = 22
        self.n_spatial_dim = 3

        # kBT in kJ/mol
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.kBT = (kB * temperature * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)

        # Build OpenMM system
        self._build_system(platform)

    def _build_system(self, platform_name):
        """Build OpenMM system for alanine dipeptide in vacuum."""
        # Load alanine dipeptide topology from OpenMM's built-in data
        # Use PDBFile with a standard alanine dipeptide PDB
        import os
        import tempfile

        # Standard alanine dipeptide PDB (ACE-ALA-NME in vacuum)
        # Write a minimal PDB file
        pdb_string = self._get_aldp_pdb()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            pdb_path = f.name

        pdb = app.PDBFile(pdb_path)
        os.unlink(pdb_path)

        # Force field
        forcefield = app.ForceField('amber14-all.xml')

        # Create system (vacuum, no cutoff)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,  # No constraints — we need full flexibility
        )

        # Create integrator (not used for dynamics, just needed for Context)
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)

        # Platform
        platform = openmm.Platform.getPlatformByName(platform_name)
        self.context = openmm.Context(system, integrator, platform)

        # Store initial positions for reference
        self.init_positions = pdb.positions

        # Store topology for dihedral computation
        self.topology = pdb.topology

    def _get_aldp_pdb(self):
        """Return a standard alanine dipeptide PDB string.

        ACE-ALA-NME (22 atoms) in vacuum.
        You can also download from:
        https://raw.githubusercontent.com/openmm/openmm/master/wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb
        """
        # Try to download or use a built-in test file
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/openmm/openmm/master/wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb"
            response = urllib.request.urlopen(url)
            return response.read().decode('utf-8')
        except Exception:
            # Fallback: use openmm test data if available
            import openmm.app as app
            import os
            openmm_dir = os.path.dirname(os.path.abspath(app.__file__))
            test_pdb = os.path.join(openmm_dir, '..', '..', 'tests', 'systems',
                                     'alanine-dipeptide-implicit.pdb')
            if os.path.exists(test_pdb):
                with open(test_pdb) as f:
                    return f.read()
            raise FileNotFoundError(
                "Cannot find alanine dipeptide PDB. Download manually from "
                "https://raw.githubusercontent.com/openmm/openmm/master/"
                "wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb "
                "and place it at data/alanine-dipeptide.pdb"
            )

    @torch.no_grad()
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x) / kBT for a batch of configurations.

        Args:
            x: (B, 66) tensor — flattened Cartesian coordinates in nanometers

        Returns:
            E: (B,) tensor — dimensionless energy E/kBT
        """
        B = x.shape[0]
        assert x.shape[1] == 66

        x_np = x.detach().cpu().numpy()
        energies = np.zeros(B)

        for i in range(B):
            coords = x_np[i].reshape(self.n_atoms, 3)
            self.context.setPositions(coords * unit.nanometers)
            state = self.context.getState(getEnergy=True)
            E_kJ = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energies[i] = E_kJ / self.kBT

        return torch.tensor(energies, dtype=x.dtype, device=x.device)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score s(x) = -∇E(x)/kBT using OpenMM forces.

        Args:
            x: (B, 66) tensor

        Returns:
            score: (B, 66) tensor — negative gradient of dimensionless energy
        """
        B = x.shape[0]
        x_np = x.detach().cpu().numpy()
        scores = np.zeros((B, 66))

        for i in range(B):
            coords = x_np[i].reshape(self.n_atoms, 3)
            self.context.setPositions(coords * unit.nanometers)
            state = self.context.getState(getForces=True)
            # Forces are -dE/dx in kJ/(mol·nm)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometers
            )
            # score = -∇(E/kBT) = forces / kBT
            scores[i] = forces.flatten() / self.kBT

        return torch.tensor(scores, dtype=x.dtype, device=x.device)

    def compute_dihedrals(self, x: torch.Tensor) -> tuple:
        """Compute Ramachandran angles (φ, ψ) for visualization.

        Args:
            x: (B, 66) tensor

        Returns:
            phi: (B,) tensor — φ dihedral in radians
            psi: (B,) tensor — ψ dihedral in radians
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("mdtraj required for dihedral computation: pip install mdtraj")

        B = x.shape[0]
        coords = x.detach().cpu().numpy().reshape(B, self.n_atoms, 3)

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


Step 3: Generate Reference Samples
Run a long MD simulation at 300K to get reference equilibrium samples. Save as .npy file.
Create scripts/generate_aldp_reference.py:

"""
Generate reference samples for alanine dipeptide via OpenMM Langevin dynamics.

Run a long simulation (10ns), save snapshots every 1ps → 10,000 frames.
Then subsample to get uncorrelated reference samples.
"""

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

# Build system (same as AlanineDipeptideEnergy._build_system)
# Download PDB
import urllib.request
url = "https://raw.githubusercontent.com/openmm/openmm/master/wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb"
urllib.request.urlretrieve(url, "data/alanine-dipeptide.pdb")

pdb = app.PDBFile("data/alanine-dipeptide.pdb")
forcefield = app.ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)

# Langevin integrator at 300K
temperature = 300 * unit.kelvin
friction = 1.0 / unit.picoseconds
timestep = 1.0 * unit.femtoseconds
integrator = openmm.LangevinIntegrator(temperature, friction, timestep)

platform = openmm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Minimize energy first
simulation.minimizeEnergy()

# Equilibration: 100ps
print("Equilibrating...")
simulation.step(100_000)  # 100ps at 1fs timestep

# Production: 10ns, save every 1ps
print("Production run...")
n_frames = 10_000
save_interval = 1_000  # every 1ps
all_positions = []

for i in range(n_frames):
    simulation.step(save_interval)
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    all_positions.append(pos.flatten())  # (66,)

    if (i + 1) % 1000 == 0:
        print(f"  Frame {i+1}/{n_frames}")

all_positions = np.array(all_positions, dtype=np.float32)  # (10000, 66)
print(f"Collected {len(all_positions)} frames, shape {all_positions.shape}")

# Save
np.save("data/test_split_ALDP-10000.npy", all_positions)
print("Saved to data/test_split_ALDP-10000.npy")

# Also compute and save dihedrals for visualization
try:
    import mdtraj as md
    top = md.Topology.from_openmm(pdb.topology)
    traj = md.Trajectory(all_positions.reshape(-1, 22, 3), top)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    np.save("data/aldp_phi.npy", phi[:, 0])
    np.save("data/aldp_psi.npy", psi[:, 0])
    print("Saved Ramachandran angles")
except ImportError:
    print("mdtraj not available, skipping dihedral computation")


Run it:

conda activate asbs_aldp
python scripts/generate_aldp_reference.py


This should take 5-10 minutes on CPU.
Step 4: Create Config Files
configs/problem/aldp.yaml:

# @package _global_

dim: 66
n_particles: 22
spatial_dim: 3

energy:
  _target_: adjoint_samplers.energies.aldp_energy.AlanineDipeptideEnergy
  dim: ${dim}
  temperature: 300.0
  platform: CPU

evaluator:
  _target_: adjoint_samplers.components.evaluator.SyntheticEenergyEvaluator
  ref_samples_path: data/test_split_ALDP-10000.npy


configs/experiment/aldp_asbs.yaml:

# @package _global_

defaults:
  - /problem: aldp
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: term_cost
  - /matcher@adjoint_matcher: adjoint_ve

exp_name: aldp_asbs
nfe: 200
sigma_max: 0.1
sigma_min: 0.001
rescale_t: null
num_epochs: 5000
max_grad_E_norm: 100

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 500
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-4
    weight_decay: 0

use_wandb: false
eval_freq: 200


IMPORTANT NOTE ON SIGMA_MAX: Alanine dipeptide coordinates are in NANOMETERS. Typical bond lengths are ~0.15 nm, and the molecule is ~1 nm across. sigma_max=0.1 nm is a reasonable starting point. Do NOT use sigma_max=2 like the LJ benchmarks — that would blow the molecule apart. You may need to tune this. Start with 0.1, and if training doesn’t converge, try 0.05 or 0.2.
IMPORTANT NOTE ON ARCHITECTURE: Start with FourierMLP (non-graph). If results are poor, consider using EGNN with n_particles=22, spatial_dim=3 — but this requires the graph SDE and graph-aware configs. Try FourierMLP first since it’s simpler.
configs/experiment/aldp_ksd_asbs.yaml: Same as aldp_asbs but change matcher to ksd_adjoint_ve and add ksd_lambda: 0.1. Start with small lambda — the energy scale for molecular systems in kBT units is very different from LJ.
Step 5: Train

conda activate asbs_aldp

# Baseline
python train.py experiment=aldp_asbs seed=0 use_wandb=false

# SDR-ASBS
python train.py experiment=aldp_ksd_asbs seed=0 use_wandb=false


Training will be SLOW because OpenMM energy evaluations are called in a Python loop (one at a time per sample). Expect ~10x slower per epoch than LJ13. For 5000 epochs with batch size 500, that’s 500 OpenMM calls per buffer refresh. Consider reducing resample_size to 200 if it’s too slow.
Step 6: Evaluate with Ramachandran Plot
The key evaluation for alanine dipeptide is the Ramachandran plot — scatter plot of (φ, ψ) dihedral angles. This is the standard visualization in every enhanced sampling paper.
Create scripts/eval_aldp.py:

"""
Evaluate alanine dipeptide: compute Ramachandran plot (φ, ψ) for
baseline and SDR-ASBS, compare mode coverage in dihedral space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load models (same pattern as other eval scripts)
# Generate 2000 samples from each
# Compute phi, psi via energy.compute_dihedrals(samples)

# Figure: 3-panel Ramachandran plot
# Panel 1: Reference (from MD) — should show C7eq, C7ax, C5, αR, αL basins
# Panel 2: Baseline ASBS — likely misses minor basins (C7ax, αL)
# Panel 3: SDR-ASBS — should cover more basins

# Metrics:
# - Count samples in each Ramachandran basin
# - Energy W2, KSD², dist_W2 (same as other benchmarks)
# - Dihedral coverage: number of occupied basins out of 5-6 known states


The Ramachandran plot should show:
	∙	Reference: 5-6 populated basins (C7eq dominant, C7ax, C5, αR, αL)
	∙	Baseline ASBS: 1-3 basins (typically C7eq + maybe αR)
	∙	SDR-ASBS: 3-5 basins (hopefully more than baseline)
This is the most recognized plot in computational chemistry — any reviewer will immediately understand it.
Step 7: Important Caveats
	1.	OpenMM energy evaluation is on CPU and sequential (Python loop). This is the bottleneck. For production, you could batch via OpenMM’s CUDA platform, but that requires GPU memory management alongside PyTorch.
	2.	The coordinates must be physically reasonable — atoms cannot overlap (infinite energy) or be too far apart (bond breaking). The source distribution gauss = N(0, I) in 66D will produce GARBAGE initial coordinates. You need a better source:
	∙	Option A: Use reference samples as source (load from .npy, add small noise)
	∙	Option B: Use a harmonic prior centered at the minimized structure
	∙	Option C: Initialize from the PDB coordinates + noise
The ASBS paper likely handles this with a carefully chosen source. Check if starting from N(0, 0.01²I) centered at the minimized PDB coordinates works. If the source is too far from physical configurations, OpenMM will return NaN energies.
	3.	The force field (AMBER14) has no constraints in our setup. In standard MD, bond lengths are constrained (SHAKE/LINCS). Without constraints, the energy landscape includes unphysical high-frequency bond vibrations. This is fine for sampling — it just means the energy barriers include bond-stretch contributions.
	4.	If OpenMM returns NaN or infinite energies during training, it means particles have reached unphysical configurations (atoms overlapping). Add a gradient clipping or energy clipping: E = min(E, max_energy) with max_energy = 1000 (in kBT units).
	5.	The coordinate units matter: OpenMM uses nanometers. If the SDE noise scale (sigma_max) is in nanometers, 0.1 nm is about one bond length — a reasonable perturbation. 1.0 nm would displace atoms by the full molecular diameter.