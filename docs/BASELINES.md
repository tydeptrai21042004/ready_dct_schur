# Published baseline library

These methods are comparison implementations, not proposal variants. The `fidelity_tier` and disclosure fields must be reported with benchmark results.

| Canonical ID | Method | Tier | Fidelity | Payload | Original host? |
|---|---|---|---|---:|---|
| `dct_dew_langelaar2001_blind` | Langelaar 2001 \| DCT DEW \| Blind | `blind` | `paper_core_decoded_frame_DCT_adapter` | 256 bits | No |
| `dct_dm_qim_chen2001_blind` | Chen-Wornell 2001 \| DCT DM QIM \| Blind | `blind` | `paper_modulation_common_image_adapter` | 4096 bits | No |
| `dct_iss_malvar2003_blind` | Malvar-Florencio 2003 \| DCT ISS \| Blind | `blind` | `paper_modulation_common_image_adapter` | 4096 bits | No |
| `dct_stdm_qim_chen2001_blind` | Chen-Wornell 2001 \| DCT STDM QIM \| Blind | `blind` | `paper_modulation_common_image_adapter` | 4096 bits | No |
| `hessenberg_nha2023_blind` | Nha 2023 \| HESSENBERG QUANTIZATION \| Blind | `blind` | `common_payload_adaptation` | 4096 bits | No |
| `iwt_svd_qim_zhu2021_blind` | Zhu 2021 \| IWT SVD QIM \| Blind | `blind` | `conceptual_iwt_svd_qim_adapter` | 4096 bits | No |
| `qwt_qsvd_zhang2022_blind` | Zhang 2022 \| QWT QSVD \| Blind | `blind` | `paper_core_common_payload_adapter` | 4096 bits | No |
| `dwt_hessenberg_fwa_gaata2022_keyassisted` | Gaata 2022 \| DWT HESSENBERG FIREWORKS \| Key-assisted blind | `key_assisted_blind` | `major_algorithm_adaptation` | 4096 bits | No |
| `dwt_qr_fa_guo2017_keyassisted` | Guo 2017 \| DWT QR FIREFLY \| Key-assisted blind | `key_assisted_blind` | `paper_guided_implementation` | 4096 bits | No |
| `dct_ca_qim_mao2024_semiblind` | Mao 2024 \| DCT CA QIM A2 \| Semi-blind | `semi_blind` | `paper_core_common_image_adapter` | 4096 bits | No |
| `dwt_hessenberg_svd_paper2025_semiblind` | DWT-HD-SVD 2025 \| DWT HESSENBERG SVD \| Semi-blind | `semi_blind` | `paper_guided_adaptation` | 4096 bits | No |
| `dwt_svd_roy2018_semiblind` | Roy 2018 \| DWT SVD \| Semi-blind | `semi_blind` | `formula_aligned_adaptation` | 4096 bits | No |
| `dwt_wht_svd_kumar2024_semiblind` | Kumar et al. 2024 \| DWT_WHT SVD \| Semi-blind | `semi_blind` | `paper_guided_implementation` | 4096 bits | No |
| `qwt_qsvd_zhang2022_semiblind` | Zhang 2022 \| QWT QSVD \| Semi-blind | `semi_blind` | `paper_core_common_payload_adapter` | 4096 bits | No |
| `dct_spread_spectrum_cox1997_nonblind` | Cox 1997 \| DCT SPREAD SPECTRUM \| Non-blind | `non_blind` | `paper_core_informed_binary_sign_adapter` | 4096 bits | No |
| `dwt_entropy_kumar2021_nonblind` | Kumar-Singh 2021 \| DWT ENTROPY ALPHA BLENDING \| Non-blind | `non_blind` | `paper_guided_adaptation` | 4096 bits | Yes |

## Important comparison rules

- Compare methods within the same blindness tier before making broad claims.
- The non-blind methods use information unavailable to blind methods.
- DCT DEW uses a 256-bit payload and must not be presented as a common-4,096-bit comparison.
- Key-assisted methods may store cover-dependent schedules or statistics; key size should be reported.
- Paper-guided adaptations must retain their fidelity disclosure in tables and captions.

## Standalone use

```bash
python -m dct_schur baseline-embed \
  --method dct_dm_qim_chen2001_blind \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --output results/baseline_marked.png \
  --key results/baseline_key.pkl

python -m dct_schur baseline-extract \
  --image results/baseline_marked.png \
  --key results/baseline_key.pkl \
  --output results/baseline_recovered.png
```
