# Invariant–Relational DCT–Schur Watermarking

This repository contains **one proposed method**—Invariant–Relational DCT–Schur—and **16 published baseline implementations** used only for controlled comparison. DCT–QR and Spatial DetQR are not proposal modules in this repository.

## Core research idea

For each 8×8 DCT block, the virtual upper-triangular Schur form is written as

\[
T=D+N,
\]

where the unchanged diagonal \(D\) represents local spectral identity and the strict-upper component \(N\) contains writable relational coordinates. The embedding is the minimum-Frobenius projection

\[
n^\star=n+H^{\mathsf T}(t-Hn),
\]

so the virtual Schur spectrum, trace, and determinant remain invariant. During extraction, the preserved diagonal-derived scale acts as an identity witness that reduces confidence in locally damaged coupling evidence.

The application layer supports binary watermarks, arbitrary bytes, text, JSON, signed provenance records, rendered document pages, and video-frame chains.

## Correct research separation

```text
dct_schur/    only proposed method and its application core
baselines/    published comparison implementations
benchmarking/ common adapters, method selection, runner, outputs
pipelines/    user-facing image/data/provenance/video/benchmark workflows
evaluation/   attacks and metrics shared by every method
```

The proposal never imports baseline implementation code. Baselines never import the DCT–Schur engine. They meet only through `benchmarking.MethodAdapter`.

## Included comparison methods

- 7 strict blind baselines;
- 5 semi-blind baselines;
- 2 key-assisted blind baselines;
- 2 non-blind reference methods;
- DCT DEW at its disclosed 256-bit native payload;
- all other methods at the common 4,096-bit payload.

Run:

```bash
python -m dct_schur list-methods
```

See [`docs/BASELINES.md`](docs/BASELINES.md) for IDs, tiers, fidelity disclosures, and payload sizes.

## Validated proposal result

| Metric | Value |
|---|---:|
| Mean PSNR across 13 hosts | **50.379869 dB** |
| Minimum PSNR across 13 hosts | **50.312240 dB** |
| Clean NC on every host | **1.000000** |
| Spectrum/trace/determinant preservation | Verified |
| Repository tests | **13 passed** |
| Baseline clean smoke tests | **16/16 passed** |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Main commands

```bash
# Show the one proposal and all baselines
python -m dct_schur list-methods

# DCT–Schur binary-image pipeline
python -m dct_schur image-embed \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --output results/schur_marked.png \
  --key results/schur_key.json

# Run one baseline independently
python -m dct_schur baseline-embed \
  --method dct_dm_qim_chen2001_blind \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --output results/dm_qim_marked.png \
  --key results/dm_qim_key.pkl

# Compare every method on one host and one watermark
python -m dct_schur benchmark \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --methods all \
  --suite sanity \
  --output-json results/all_methods_sanity.json \
  --output-summary-csv results/all_methods_sanity_summary.csv \
  --output-attack-csv results/all_methods_sanity_attacks.csv

# Full host-folder experiment
python -m dct_schur benchmark \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods all \
  --suite extended \
  --output-json results/full_benchmark.json \
  --output-summary-csv results/full_benchmark_summary.csv \
  --output-attack-csv results/full_benchmark_attacks.csv


# Resumable full comparison
python -m dct_schur benchmark-checkpointed \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods all \
  --suite extended \
  --output-directory results/checkpointed_full
```

Selectors include `dct_schur`, `baselines`, `blind`, `semi_blind`, `key_assisted`, `non_blind`, `common_4096`, and comma-separated method IDs.

## Verify

```bash
pytest -q
python -m dct_schur audit --root .
```

## Scientific limitation

The proposal remains limited by transformations that destroy 8×8 block correspondence, especially uncompensated crop, rotation, perspective, deformation, and print–scan. The next core contribution should be an orthogonal Schur synchronization subspace, not a larger QIM step.
