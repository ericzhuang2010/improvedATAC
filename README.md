# improvedATAC

**ATAC-native probabilistic deconvolution with learned peak selection.**

improvedATAC estimates cell-type composition from spatial ATAC-seq data using a
Bayesian generative model built on [Pyro](https://pyro.ai/). Unlike
RNA-based deconvolution tools, it operates directly on chromatin accessibility
counts, learns which peaks are most informative for deconvolution, and
incorporates peak-level scores as structured priors.

## Generative model

Each observed fragment count $y_{s,p}$ at spot $s$ and peak $p$ is modelled as:

$$y_{s,p} \sim \text{Poisson}(\mu_{s,p})$$

$$\mu_{s,p} = d_s \cdot b_p \cdot \sum_c a_{s,c}\, w_p\, \theta_{c,p}$$

| Symbol | Meaning |
|--------|---------|
| $d_s$ | Spot-level sequencing depth (Gamma prior centred on observed library size) |
| $b_p$ | Peak-specific technical bias — GC content, mappability, Tn5 insertion (Gamma prior centred at 1) |
| $a_{s,c}$ | Abundance of cell type $c$ in spot $s$ (Gamma prior controlled by `N_cells_per_location`) |
| $w_p$ | Learned peak informativeness weight (Gamma prior whose mean is set by pre-computed peak scores) |
| $\theta_{c,p}$ | Reference accessibility profile (fixed from scATAC reference) |

Alternative observation likelihoods — Negative Binomial (`"nb"`) and
Zero-Inflated Poisson (`"zip"`) — are also available.

## Features

- **ATAC-native**: works directly on fragment / insertion counts without
  converting to gene activity scores.
- **Learned peak weights**: the model learns which peaks carry the most
  deconvolution signal, informed by composite peak scores (specificity ×
  reproducibility × spatial variability).
- **Flexible likelihoods**: Poisson (default), Negative Binomial, or
  Zero-Inflated Poisson.
- **Peak scoring and filtering**: optional automated selection of the most
  informative peaks before training.
- **Scanpy-compatible**: inputs and outputs are
  [AnnData](https://anndata.readthedocs.io/) objects; results are stored in
  `adata.obsm` for downstream analysis.
- **One-call pipeline**: `deconvolve()` runs the full workflow — reference
  signatures → peak scoring → SVI training → posterior export — in a single
  function call.

## Installation

```bash
pip install -e .
```

With development dependencies (pytest, pre-commit, ruff):

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Pyro ≥ 1.8. See
`pyproject.toml` for the full dependency list.

## Quick start

```python
import anndata as ad
import improvedatac as ia

adata_spatial = ad.read_h5ad("spatial_atac.h5ad")   # spots × peaks
adata_ref     = ad.read_h5ad("reference_scatac.h5ad")  # cells × peaks

results = ia.tl.deconvolve(
    adata_spatial,
    adata_ref,
    cluster_key="cell_type",   # column in adata_ref.obs
    max_epochs=30_000,
    use_gpu=True,
)

# Cell-type proportions per spot (spots × cell types, rows sum to 1)
proportions = results["proportions"]

# Also stored directly on the spatial object
adata_spatial.obsm["proportions"]
```

## Pipeline overview

`ia.tl.deconvolve()` orchestrates four steps:

1. **Reference signatures** — compute accessibility profiles
   $\theta_{c,p}$ from the scATAC reference by averaging within each cell
   type.
2. **Peak scoring** *(optional)* — rank peaks by a composite of cell-type
   specificity, within-cluster reproducibility, and spatial variability;
   optionally filter to the top $k$.
3. **Model training** — fit the Pyro SVI model with mini-batch stochastic
   variational inference and early stopping.
4. **Posterior export** — draw posterior samples for cell-type abundances and
   peak weights; normalise to proportions; save CSVs and annotate the AnnData
   object.

## API reference

### `improvedatac.tl` — tools

| Function | Description |
|----------|-------------|
| `deconvolve(adata_spatial, adata_ref, cluster_key, ...)` | End-to-end deconvolution pipeline |
| `rmse(true, predicted)` | Root mean squared error |
| `jsd(true, predicted)` | Mean Jensen–Shannon divergence |
| `pearson_per_celltype(true, predicted)` | Per-cell-type Pearson correlation |
| `spearman_per_celltype(true, predicted)` | Per-cell-type Spearman correlation |

### `improvedatac.pp` — preprocessing

| Function | Description |
|----------|-------------|
| `compute_peak_scores(adata_ref, cluster_key, ...)` | Composite peak informativeness scores |

### `improvedatac.model` — model components

| Class / Function | Description |
|-----------------|-------------|
| `ATACDeconvModel` | High-level model: training, posterior export, proportions, save/load |
| `ATACDeconvModule` | Pyro generative module implementing the probabilistic model |
| `compute_reference_signatures(adata_ref, ...)` | Reference profile estimation from scATAC |
| `compute_normalized_signatures(adata_ref, ...)` | Normalized reference profiles |

### Key `ATACDeconvModel` methods

| Method | Description |
|--------|-------------|
| `train(max_epochs, lr, batch_size, ...)` | Run SVI with early stopping |
| `export_posterior(num_samples)` | Sample posterior means, quantiles, and std for abundances and peak weights |
| `get_proportions(num_samples)` | Normalised cell-type proportions (rows sum to 1) |
| `plot_history(skip)` | ELBO convergence plot |
| `plot_peak_weights(top_n)` | Bar plot of top learned peak weights |
| `save(path)` / `load(path)` | Persist and restore model parameters |

## Step-by-step usage

For finer control you can call each stage individually:

```python
import improvedatac as ia

# 1. Reference signatures
cell_state_df = ia.model.compute_reference_signatures(
    adata_ref, cluster_key="cell_type"
)

# 2. Peak scoring
peak_scores = ia.pp.compute_peak_scores(
    adata_ref, cluster_key="cell_type", adata_spatial=adata_spatial
)

# 3. Build and train model
model = ia.model.ATACDeconvModel(
    adata_spatial, cell_state_df,
    peak_scores=peak_scores,
    likelihood="poisson",
)
model.train(max_epochs=30_000, lr=0.005)

# 4. Extract results
results = model.export_posterior(num_samples=1000)
proportions = model.get_proportions()

# Diagnostics
model.plot_history(skip=100)
model.plot_peak_weights(top_n=30)

# Save / load
model.save("model.pt")
model.load("model.pt")
```

## Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `N_cells_per_location` | 8.0 | Expected cells per spot; increase for dense tissues |
| `detection_alpha` | 20.0 | Higher values shrink spot depth toward observed library size |
| `peak_weight_concentration` | 5.0 | Higher values shrink peak weights toward prior scores |
| `peak_bias_concentration` | 10.0 | Higher values keep peak bias closer to 1 |
| `abundance_mean_var_ratio` | 5.0 | Mean/variance ratio for abundance prior |
| `likelihood` | `"poisson"` | `"poisson"`, `"nb"`, or `"zip"` |
| `use_peak_scores` | `True` | Incorporate peak informativeness into the prior |
| `n_top_peaks` | `None` | Filter to top-k most informative peaks |
| `shrinkage` | 0.0 | Shrinkage toward global mean in reference signatures |
| `max_epochs` | 30000 | Maximum SVI iterations (early stopping is automatic) |

## Project structure

```
src/improvedatac/
├── __init__.py              # Package entry point
├── model/
│   ├── _reference.py        # Reference signature estimation
│   ├── _spatial_module.py   # Pyro generative module
│   └── _spatial_model.py    # Training, posterior, diagnostics
├── pp/
│   └── peak_scoring.py      # Peak informativeness scoring
└── tl/
    ├── deconvolve.py        # End-to-end pipeline
    └── metrics.py           # Evaluation metrics (RMSE, JSD, correlations)
```

## License

MIT
