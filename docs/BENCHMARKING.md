# Unified benchmarking

## Fair execution boundary

Every method receives:

- the same host image;
- the same source watermark;
- the same deterministic attack object;
- the same image and watermark metric functions;
- the same seed where the method accepts a seed.

The adapter does not alter a method's embedding or extraction equation.

## Payload handling

Most methods use a 64×64 binary watermark (4,096 bits). DCT DEW is evaluated at 16×16 (256 bits), matching the disclosed native payload used in the reference benchmark. Result rows always include `payload_bits` and `comparison_group` so unlike-payload results are not silently treated as equivalent.

## Method selectors

```text
all
baselines
dct_schur
blind
semi_blind
key_assisted
non_blind
common_4096
<comma-separated canonical IDs>
```

## Outputs

- JSON contains complete configuration, trial, clean, timing, attack, and metadata results.
- Summary CSV contains one row per method.
- Attack CSV contains one row per method/host/payload/attack.

## Examples

```bash
python -m dct_schur benchmark \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods common_4096 \
  --suite extended \
  --output-json results/common_4096.json \
  --output-summary-csv results/common_4096_summary.csv \
  --output-attack-csv results/common_4096_attacks.csv
```

To fail immediately instead of recording an error row:

```bash
python -m dct_schur benchmark ... --strict
```

## Checkpointed long runs

For the full 17-method × 113-attack experiment, use one checkpoint per method:

```bash
python -m dct_schur benchmark-checkpointed \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods all \
  --suite extended \
  --output-directory results/checkpointed_full
```

Re-running the same command skips valid completed method JSON files. Use `--no-resume` to force recomputation. `index.json` records completed, failed, and pending methods.
