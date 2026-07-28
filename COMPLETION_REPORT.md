# Corrected completion report

## Correct interpretation

The repository now has:

- one proposed method: **Invariant–Relational DCT–Schur**;
- no DCT–QR proposal implementation;
- no Spatial DetQR proposal implementation;
- all 16 published baseline implementations restored;
- no `src/` directory;
- modular proposal, baseline, benchmarking, pipeline, and evaluation packages.

## Baseline restoration

Restored categories:

- strict blind: DM-QIM, STDM-QIM, ISS, DEW, Hessenberg quantization, IWT-SVD-QIM, QWT-QSVD blind;
- semi-blind: CA-QIM, DWT-HD-SVD, Roy DWT-SVD, DWT-WHT-SVD, QWT-QSVD semi-blind;
- key-assisted blind: Guo DWT-QR-FA and Gaata DWT-Hessenberg-FWA;
- non-blind: Cox spread spectrum and DWT entropy alpha blending.

The canonical registry contains 16 unique baseline IDs. All 16 complete clean embed/extract smoke tests with NC at least 0.99.

## Common benchmark pipeline

A new `benchmarking/` package provides:

1. method specifications and tier selectors;
2. isolated proposal/baseline adapters;
3. per-trial timing and metadata collection;
4. common attacks and metrics;
5. method-specific payload disclosure;
6. JSON, summary CSV, and attack-level CSV outputs.

## Proposal validation retained

Across all 13 supplied hosts:

- mean PSNR: `50.379869 dB`;
- minimum PSNR: `50.312240 dB`;
- clean NC: `1.000000` on every host;
- spectrum, trace, and determinant preserved;
- mean coset projection-energy ratio: `0.586771`.

## Validation status

- repository tests: `13 passed`;
- baseline clean smoke: `16/16 passed`;
- proposal plus baseline common-pipeline test: passed;
- no `src/` directory;
- exactly one registered proposal;
- all 16 baselines registered;
- no folder above 100 files.
