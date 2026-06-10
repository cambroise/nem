# Changelog

All notable changes to **pynem** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Optional `viz` extra: matplotlib is no longer a hard dependency. Install
  `pynem[viz]` for plotting; `import pynem` works without matplotlib (lighter on
  headless servers). Plotting functions raise a clear error if it is missing.
- `CHANGELOG.md` and `CITATION.cff`.

### Changed
- Continuous integration (GitHub Actions): lint (ruff) + a test matrix on Python
  3.9–3.13; ruff configuration in `pyproject.toml`.

## [0.5.0] — 2026-06-10

### Added
- **MAG-aware completeness** (mOTUpan-style): per-genome `completeness` for the
  Bernoulli model. An absence in an incomplete genome is forgiven
  (`P(x=1) = mu_kj * gamma_j`), so incomplete metagenome-assembled genomes no
  longer collapse the persistent class. `NEM(completeness=...)` and
  `partition_pangenome(completeness=...)`.
- `completeness="auto"`: self-estimate completeness from the data (no CheckM) by
  iterative re-estimation from the inferred persistent set (mOTUpan Eq. 6).

## [0.4.0] — 2026-06-05

### Added
- **Weighted NEM**: per-variable `feature_weights` (`NEM`) and per-genome
  `genome_weights` (`partition_pangenome`) to down-weight redundant features.
- `pynem.genome_weights(...)`: derive the weights automatically by grouping the
  genomes with Jaccard distance + UPGMA (`pynem.weights`), choosing the number of
  groups by a hand-rolled silhouette (no scikit-learn dependency).

## [0.3.1] — 2026-06-05

### Added
- Input validation in `fit` (graph required, contiguous `0..N-1` node labels,
  `1 <= K <= N`) with clear errors.

### Changed
- Split the overloaded `EPSILON` into named floors (`PROB_FLOOR`, `VAR_FLOOR`,
  `ZERO_DISP_TOL`, `DIV_GUARD`); values unchanged.
- Cache `log(p_k f_k)` once per iteration (shared between the E-step and the
  criteria).

## [0.3.0]

### Added
- `pynem.partition_pangenome`: a PPanGGOLiN-faithful pipeline reproducing the
  embedded NEM C core element-wise (validated against `run_partitioning`).
- k-means++ recovery of collapsed (empty) classes.

### Changed
- Performance: sparse CSR adjacency (`spatial.py`, contexts = `A @ C`),
  vectorised densities, and an optional Numba JIT of the sequential E-step
  (`pynem[fast]`) with a pure-Python fallback.

## [0.2.0]

- PPanGGOLiN vendored as a git submodule for cross-validation.

## [0.1.0]

- Initial standalone `pynem` package: the NEM algorithm (Normal / Laplace /
  Bernoulli families, `nem`/`ncem`/`gem`, dispersion and proportion models),
  with a scikit-learn-style API, I/O for `.str`/`.dat`/`.nei`, and visualization.

[Unreleased]: https://github.com/cambroise/nem/compare/pynem-v0.5.0...HEAD
[0.5.0]: https://github.com/cambroise/nem/releases/tag/pynem-v0.5.0
[0.4.0]: https://github.com/cambroise/nem/releases/tag/pynem-v0.4.0
[0.3.1]: https://github.com/cambroise/nem/releases/tag/pynem-v0.3.1
[0.3.0]: https://github.com/cambroise/nem/releases/tag/pynem-v0.3.0
[0.2.0]: https://github.com/cambroise/nem/releases/tag/pynem-v0.2.0
[0.1.0]: https://github.com/cambroise/nem/releases/tag/pynem-v0.1.0
