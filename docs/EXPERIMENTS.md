# Validated experiments in this rebuild

## Clean embedding

Across all 13 supplied 512×512 hosts with the supplied 64×64 payload:

| Metric | Result |
|---|---:|
| Mean PSNR | 50.379869 dB |
| Minimum PSNR | 50.312240 dB |
| Clean NC on every host | 1.000000 |
| Spectrum, trace, determinant | Preserved on every host |
| Mean coset energy ratio | 0.586771 |

The embedding law remains the validated >50 dB coset profile; the new improvement is in evidence reliability.

## Witness-gating ablation

A three-host, 60-attack common-suite ablation was run with restoration search disabled to isolate the evidence rule.

| Decoder | Mean NC |
|---|---:|
| Original gain-normalized sum | 0.836565 |
| Identity-witness gating | **0.837766** |

Category changes for the selected witness profile:

| Category | Original | Witness-gated |
|---|---:|---:|
| Noise | 0.897012 | 0.896881 |
| Filtering | 0.731151 | 0.726055 |
| Geometric | 0.672354 | **0.673854** |
| Occlusion | 0.966887 | **0.997955** |
| Photometric | 0.980651 | **0.980814** |

The main gain is local-damage handling. Compression and filtering remain the main trade-off, so aggressive robust-median aggregation is retained only as an ablation option and is disabled in the default profile.

## Restoration-enabled subset

On two hosts and 15 representative attacks with the full restoration candidate search:

| Decoder | Mean NC |
|---|---:|
| Previous selection/evidence | 0.914055 |
| Witness-aware selection/evidence | **0.914538** |

The improvement is modest but positive and introduces no embedding distortion.

## Interpretation

The current result supports the narrower claim that the invariant witness improves resistance to localized corruption while retaining the >50 dB embedding profile. It does not yet solve rotation, crop, affine, perspective, or print–scan synchronization. Those attacks require a dedicated synchronization subspace rather than stronger QIM.
