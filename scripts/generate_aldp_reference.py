"""
Generate reference samples for alanine dipeptide via OpenMM Langevin dynamics.

Run a long simulation (10ns), save snapshots every 1ps -> 10,000 frames.
Then subsample to get uncorrelated reference samples.
"""

import numpy as np
import os
import sys

import openmm
import openmm.app as app
import openmm.unit as unit

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
data_dir = os.path.abspath(data_dir)
os.makedirs(data_dir, exist_ok=True)

# Download PDB if not present
pdb_path = os.path.join(data_dir, "alanine-dipeptide.pdb")
if not os.path.exists(pdb_path):
    import urllib.request
    url = ("https://raw.githubusercontent.com/openmm/openmm/master/"
           "wrappers/python/tests/systems/alanine-dipeptide-implicit.pdb")
    print(f"Downloading PDB from {url}...")
    urllib.request.urlretrieve(url, pdb_path)
    print(f"Saved to {pdb_path}")

pdb = app.PDBFile(pdb_path)
forcefield = app.ForceField('amber14-all.xml')
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=None,
)

# Langevin integrator at 300K
temperature = 300 * unit.kelvin
friction = 1.0 / unit.picoseconds
timestep = 1.0 * unit.femtoseconds
integrator = openmm.LangevinIntegrator(temperature, friction, timestep)

platform = openmm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Minimize energy first
print("Minimizing energy...")
simulation.minimizeEnergy()
state = simulation.context.getState(getEnergy=True)
E_min = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
print(f"Minimized energy: {E_min:.2f} kJ/mol")

# Save minimized positions
min_pos = simulation.context.getState(getPositions=True).getPositions(
    asNumpy=True
).value_in_unit(unit.nanometers)
np.save(os.path.join(data_dir, "aldp_minimized.npy"),
        min_pos.flatten().astype(np.float32))
print("Saved minimized coordinates")

# Equilibration: 100ps
print("Equilibrating (100ps)...")
simulation.step(100_000)  # 100ps at 1fs timestep
print("Equilibration done.")

# Production: 10ns, save every 1ps
print("Production run (10ns)...")
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
        sys.stdout.flush()

all_positions = np.array(all_positions, dtype=np.float32)  # (10000, 66)
print(f"Collected {len(all_positions)} frames, shape {all_positions.shape}")

# Save
save_path = os.path.join(data_dir, "test_split_ALDP-10000.npy")
np.save(save_path, all_positions)
print(f"Saved to {save_path}")

# Also compute and save dihedrals for visualization
try:
    import mdtraj as md
    top = md.Topology.from_openmm(pdb.topology)
    traj = md.Trajectory(all_positions.reshape(-1, 22, 3), top)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    np.save(os.path.join(data_dir, "aldp_phi.npy"), phi[:, 0])
    np.save(os.path.join(data_dir, "aldp_psi.npy"), psi[:, 0])
    print("Saved Ramachandran angles (phi, psi)")

    # Print basin statistics
    phi_deg = np.degrees(phi[:, 0])
    psi_deg = np.degrees(psi[:, 0])
    print(f"\nRamachandran stats:")
    print(f"  phi range: [{phi_deg.min():.1f}, {phi_deg.max():.1f}] deg")
    print(f"  psi range: [{psi_deg.min():.1f}, {psi_deg.max():.1f}] deg")

    # Count samples in major basins
    # C7eq: phi ~ -80, psi ~ 80
    c7eq = np.sum((phi_deg > -120) & (phi_deg < -40) & (psi_deg > 40) & (psi_deg < 120))
    # C7ax: phi ~ 75, psi ~ -65
    c7ax = np.sum((phi_deg > 40) & (phi_deg < 120) & (psi_deg > -110) & (psi_deg < -20))
    # alphaR: phi ~ -80, psi ~ -40
    alphaR = np.sum((phi_deg > -120) & (phi_deg < -40) & (psi_deg > -80) & (psi_deg < 0))
    # alphaL: phi ~ 60, psi ~ 40
    alphaL = np.sum((phi_deg > 20) & (phi_deg < 100) & (psi_deg > 0) & (psi_deg < 80))

    print(f"  C7eq: {c7eq} ({100*c7eq/len(phi_deg):.1f}%)")
    print(f"  C7ax: {c7ax} ({100*c7ax/len(phi_deg):.1f}%)")
    print(f"  alphaR: {alphaR} ({100*alphaR/len(phi_deg):.1f}%)")
    print(f"  alphaL: {alphaL} ({100*alphaL/len(phi_deg):.1f}%)")

except ImportError:
    print("mdtraj not available, skipping dihedral computation")

print("\nDone!")
