# Benchmark, Evaluation & Visualization Guide for SDR-ASBS

## 1. Benchmarks

### 1.1 Alanine Dipeptide (ALDP)

**What it is:** 22 atoms, 66 dimensions. The standard molecular benchmark for sampling methods. The Ramachandran dihedral angles (φ, ψ) define well-known metastable basins: C7eq (dominant), C7ax, αR, αL, and others. Mode concentration manifests as missing the minority basins (C7ax, αL).

**Force field:** The ASBS paper (Table 3) uses the same setup as prior work (PIS, DDS). This is the classical force field via OpenMM, typically Amber ff99SB or CHARMM22* in vacuum. The `wt_asbs` codebase uses OpenMM with their own config. For consistency with the ASBS Table 3 baselines, **use exactly the same energy class as in the original `adjoint_samplers` repo** — check `adjoint_samplers/energies/` for an ALDP energy that calls OpenMM.

**Reference data:** WT-ASBS hosts reference WTMetaD simulation data on HuggingFace (`facebook/wt-asbs`). You need to apply for access. However, for the ASBS-style metrics (KL on marginals, W2 on joint), the standard approach is to generate a long reference trajectory (e.g., 1M steps of replica exchange or long WTMetaD) and bin it into histograms. The original ASBS paper likely uses the same reference as PIS/DDS — a long MD reference trajectory.

**What you need installed:** OpenMM (`conda install -c conda-forge openmm`). The ASBS codebase should already have the ALDP energy wrapper.

### 1.2 Alanine Tetrapeptide (ALTP)

**What it is:** Larger peptide, ~8 backbone dihedrals, higher-dimensional CV space. Harder than ALDP because there are more modes and the energy barriers are higher. Less standardized benchmarks exist — fewer papers report on it.

**Caution:** ALTP is less commonly benchmarked. The ASBS paper doesn’t include it. WT-ASBS may include it (they mention “peptide conformational sampling benchmarks” in plural). If you include ALTP, you’d be providing a new benchmark result, which is good, but you won’t have ASBS baseline numbers to compare against. **Recommendation:** Include ALTP only if you can generate your own ASBS baseline alongside SDR-ASBS. Otherwise, stick to ALDP where you have Table 3 to compare against.

-----

## 2. Evaluation Metrics

### 2.1 Metrics from ASBS Table 3 (keep these)

|Metric                      |What it measures             |How to compute                                                                            |
|----------------------------|-----------------------------|------------------------------------------------------------------------------------------|
|D_KL(φ)                     |KL divergence of φ marginal  |Bin φ into histogram (e.g., 100 bins over [-π, π]), compute KL against reference histogram|
|D_KL(ψ)                     |KL divergence of ψ marginal  |Same for ψ                                                                                |
|D_KL(γ1), D_KL(γ2), D_KL(γ3)|KL for other torsion angles  |Same binning for the three non-Ramachandran torsions                                      |
|W2(φ, ψ)                    |Wasserstein-2 on joint (φ, ψ)|scipy.stats.wasserstein_distance or POT library on the 2D joint distribution              |

These are your primary **distributional quality** metrics. Report them in the same format as ASBS Table 3 so reviewers can directly compare.

### 2.2 Metrics from WT-ASBS (add these)

|Metric                                     |What it measures                                                             |How to compute                                                           |
|-------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
|Free energy difference ΔF(C7eq → C7ax)     |Accuracy of relative basin populations                                       |ΔF = -kT ln(P(C7ax)/P(C7eq)), compare to reference                       |
|Free energy difference ΔF(C7eq → αR)       |Same for another basin pair                                                  |Same computation                                                         |
|Free energy surface (FES) RMSE             |Global accuracy of the 2D free energy landscape                              |F(φ,ψ) = -kT ln p(φ,ψ), RMSE against reference FES on a grid             |
|Number of basins discovered                |Mode coverage                                                                |Count basins with population > threshold (e.g., > 1% of reference weight)|
|KL on reweighted samples (WT-ASBS specific)|For WT-ASBS, samples need reweighting by the bias; for you this doesn’t apply|Skip this — it’s specific to their biasing mechanism                     |

**How to compute ΔF:**

1. Generate N samples from your model
1. Compute (φ, ψ) for each sample
1. Assign each sample to the nearest basin (define basin regions in Ramachandran space — standard definitions exist in the literature)
1. Count: P(basin) = N_basin / N
1. ΔF = -kT ln(P(basin_A) / P(basin_B))
1. Compare to reference ΔF from long MD or WTMetaD

**Basin definitions for ALDP (approximate, in radians):**

- C7eq: φ ∈ [-2.5, -0.5], ψ ∈ [0.3, 2.0]
- C7ax: φ ∈ [0.5, 2.0], ψ ∈ [-2.0, -0.3]
- αR: φ ∈ [-2.5, -0.5], ψ ∈ [-2.0, -0.3]
- αL: φ ∈ [0.3, 1.5], ψ ∈ [0.0, 1.5]

(Verify these against your reference data — exact boundaries depend on force field.)

### 2.3 Mode discovery over training (new, your addition)

|Metric                    |What it measures                         |How to compute                                                                 |
|--------------------------|-----------------------------------------|-------------------------------------------------------------------------------|
|Modes discovered vs. epoch|How quickly mode coverage improves       |After each epoch (or every K epochs), generate samples, count discovered basins|
|α_k vs. epoch per mode    |Track policy mode weights during training|Count fraction of samples in each basin after each evaluation                  |

This directly visualizes the mode concentration phenomenon and whether SDR fixes it. It connects your theory (Theorem 1) to the experiments.

**Implementation:**

```
for epoch in evaluation_epochs:
    load checkpoint
    generate N=10000 samples
    compute (φ, ψ)
    for each basin:
        α_k = count_in_basin / N
    record {epoch, α_1, ..., α_K, num_discovered}
```

Define “discovered” as α_k > some threshold (e.g., 0.5% of reference weight w_k, or absolute threshold like 0.1%).

### 2.4 Metric summary table for the paper

Your paper should have **two tables for ALDP:**

**Table A (ASBS-style, distributional quality):**
Same format as ASBS Table 3. Methods: PIS, DDS, AS, ASBS, SDR-ASBS. Copy PIS/DDS/AS/ASBS numbers from the ASBS paper. Add your SDR-ASBS numbers. Add WT-ASBS numbers if they report in this format (they likely don’t — cite their paper instead).

**Table B (mode-specific, new):**

|Method   |ΔF(C7eq→C7ax)|ΔF(C7eq→αR)|ΔF(C7eq→αL)|Basins found|FES RMSE|
|---------|-------------|-----------|-----------|------------|--------|
|Reference|…            |…          |…          |K           |0       |
|ASBS     |…            |…          |…          |…           |…       |
|SDR-ASBS |…            |…          |…          |…           |…       |
|WT-ASBS* |…            |…          |…          |…           |…       |

*Cited from their paper, not reproduced.

-----

## 3. Visualization Strategy

### 3.1 Ramachandran plots (ALDP, essential)

**What:** 2D scatter/density plot of (φ, ψ) for generated samples, compared to reference.

**Layout:** A row of panels: [Reference] [ASBS] [SDR-ASBS]. Use the same colormap and axis range for all panels. KDE smoothing or 2D histogram with ~100×100 bins.

**Implementation:**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, samples, title in zip(axes, [ref, asbs, sdr], ["Reference", "ASBS", "SDR-ASBS"]):
    phi, psi = compute_dihedrals(samples)
    ax.hexbin(phi, psi, gridsize=80, cmap='viridis', mincnt=1)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("φ")
    ax.set_ylabel("ψ")
    ax.set_title(title)
    ax.set_aspect('equal')
```

**Key visual signal:** ASBS panel missing C7ax and/or αL basins; SDR-ASBS panel showing them.

### 3.2 Free energy surfaces (ALDP, important)

**What:** F(φ, ψ) = -kT ln p(φ, ψ) plotted as a contour map.

**Layout:** Same row of panels as above. Use contour levels in kT units (e.g., 0 to 8 kT). Clip high-energy regions to a max value.

**Implementation:**

```python
# Bin into 2D histogram
H, xedges, yedges = np.histogram2d(phi, psi, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
H = H / H.sum()  # normalize
F = -kT * np.log(H + 1e-10)  # free energy, avoid log(0)
F = F - F.min()  # shift minimum to zero

ax.contourf(xcenters, ycenters, F.T, levels=np.linspace(0, 8*kT, 20), cmap='RdYlBu_r')
```

### 3.3 Mode weight evolution over training (new, your key figure)

**What:** Line plot of α_k(epoch) for each Ramachandran basin, compared to reference weights w_k (horizontal dashed lines).

**Layout:** Two panels side by side: [ASBS] [SDR-ASBS]. Each panel has K colored lines (one per basin) and K dashed horizontal lines (target weights). x-axis: epoch. y-axis: mode fraction α_k.

**Key visual signal:** In the ASBS panel, minority mode lines flatline near zero or decrease over training (mode concentration). In the SDR-ASBS panel, they recover toward the target lines.

**This figure directly illustrates Theorem 1 and its resolution.** It’s the most important new visualization in your paper.

### 3.4 DARW weight distribution (supplementary)

**What:** Histogram of the importance weights ŵ_i during training at various epochs (early, mid, late).

**Purpose:** Shows that weights are near-uniform early (modes not yet concentrated), spread out mid-training (DARW is active), and stabilize late training.

-----

## 4. Reference Data

### 4.1 For ALDP

**Option A (recommended): Generate your own.** Run long WTMetaD or replica exchange MD using OpenMM + PLUMED. Standard recipe:

- Force field: match the ASBS paper exactly (check their config)
- CVs: φ, ψ dihedrals
- WTMetaD: bias factor 10, Gaussian height 1.2 kJ/mol, width 0.35 rad, deposit every 1 ps
- Run length: 50–100 ns (converges within ~20 ns for ALDP in vacuum)
- Extract: binned (φ, ψ) histogram, basin populations, ΔF values

**Option B: Use existing reference data.**

- WT-ASBS provides reference data on HuggingFace (`facebook/wt-asbs`). Requires applying for access.
- The original ASBS paper’s reference is likely from the `adjoint_samplers` repo. Check if there’s a reference dataset included.
- Many papers use the Amber03 / ff99SB reference FES for ALDP in vacuum, which is well-characterized in the literature.

-----

## 5. What Your Paper Should Present (Recommendation)

### Main paper figures:

1. **Ramachandran plots** (ALDP): Reference vs. ASBS vs. SDR-ASBS (3 panels)
1. **Mode weight evolution** (ALDP): α_k over training epochs, ASBS vs. SDR-ASBS (2 panels) — this is your star figure
1. **Free energy surfaces** (ALDP): contour plots, Reference vs. ASBS vs. SDR-ASBS

### Main paper tables:

1. **ASBS-style table** (Table 3 format): KL marginals + W2 joint for ALDP, comparing against PIS/DDS/AS/ASBS
1. **Mode-specific table**: ΔF between basins, number of basins discovered, FES RMSE

### Appendix:

- DARW weight distribution over training
- β ablation (β ∈ {0, 0.3, 0.5, 0.7, 1.0})
- λ ablation for KSD (if you keep KSD as a component)
- Per-seed results for all metrics
-----

## 6. Open Questions / Things to Verify

1. **Force field match:** Confirm your ALDP energy function uses the same force field as the original ASBS Table 3. If it doesn’t match, you can’t copy their baseline numbers and must rerun all baselines yourself.
1. **ALTP feasibility:** Do you have an ALTP energy class? If not, OpenMM can build it, but you need to define the system (PDB file, force field assignment). This is nontrivial setup work. Decide if the payoff is worth it.
1. **WT-ASBS numbers:** Their paper likely reports FES plots and ΔF values rather than the KL/W2 metrics from ASBS Table 3. You may not be able to directly compare in a single table. Plan to cite qualitatively: “WT-ASBS recovers all basins using domain-specific CVs; SDR-ASBS achieves similar mode coverage without requiring CV specification.”
1. **Computational cost:** WT-ASBS reports wall-clock time and energy evaluations. You should report training cost (epochs, wall-clock) for SDR-ASBS vs. ASBS to show the overhead is small.