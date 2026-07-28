# Research roadmap

## Stage 1 — completed in this repository

- Single DCT–Schur method only.
- Minimum-Frobenius strict-upper projection.
- Payload-wise coset optimization.
- Spectrum, trace, and determinant preservation.
- Diagonal-derived identity witness.
- Arbitrary data, signed provenance, document pages, and video-frame chains.
- Exact semantic validation through ECC, CRC, and signatures.

## Stage 2 — next mathematical contribution

Split the six-dimensional strict-upper space into payload and synchronization components:

\[
\mathbb R^6=\mathcal H_{\mathrm{payload}}\oplus\mathcal H_{\mathrm{sync}}.
\]

The synchronization component should estimate translation, scale, and limited rotation without modifying the diagonal. The transformation search should maximize a joint score:

\[
S(\theta)=S_{\mathrm{sync}}(\theta)
+\lambda S_{\mathrm{witness}}(\theta)
+\mu S_{\mathrm{code}}(\theta).
\]

`S_code` can use Viterbi path quality and CRC validity, providing an objective stopping criterion.

## Stage 3 — deployment validation

- Multiple watermarks, payloads, seeds, and signing keys.
- Multiple image resolutions and codecs.
- Native video datasets rather than repeated still frames.
- Camera-display and print-camera experiments.
- Key-size, throughput, and energy reporting.
- Comparison using exact authenticated-message recovery, not only logo NC.
