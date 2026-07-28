# Available pipelines

## 1. Binary image compatibility

```bash
dct-schur image-embed \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --output results/marked.png \
  --key results/image_key.json

dct-schur image-extract \
  --image results/marked.png \
  --key results/image_key.json \
  --output results/recovered.png
```

## 2. Arbitrary file transport

```bash
dct-schur data-embed \
  --host data/host/lenna.bmp \
  --payload examples/payload.txt \
  --output results/data_marked.png \
  --key results/data_key.json

dct-schur data-extract \
  --image results/data_marked.png \
  --key results/data_key.json \
  --output results/recovered_payload.txt
```

The transport automatically chooses the robust rate-1/3 profile for short payloads and the rate-1/2 profile when more capacity is required.

## 3. Signed provenance

```bash
dct-schur keygen \
  --private results/issuer_private.pem \
  --public results/issuer_public.pem

dct-schur provenance-embed \
  --host data/host/lenna.bmp \
  --private-key results/issuer_private.pem \
  --manifest examples/manifest.json \
  --output results/provenance_marked.png \
  --watermark-key results/provenance_key.json \
  --record results/provenance_record.json

dct-schur provenance-verify \
  --image results/provenance_marked.png \
  --watermark-key results/provenance_key.json \
  --public-key results/issuer_public.pem
```

## 4. Document pages

`pipelines.document.embed_document_pages` accepts a folder of rendered pages, embeds a linked provenance record in every page, and writes a bundle that verifies page order, common asset identity, signatures, and the record hash chain.

## 5. Video

```bash
dct-schur video-embed \
  --input input.mp4 \
  --output results/marked.mp4 \
  --private-key results/issuer_private.pem \
  --key-bundle results/video_keys.json \
  --stride 10

dct-schur video-verify \
  --video results/marked.mp4 \
  --key-bundle results/video_keys.json \
  --public-key results/issuer_public.pem
```

Only selected frames need to carry records. `--stride 10` embeds one linked record every ten frames.

## 6. Batch folder

```bash
dct-schur batch-data \
  --input-folder data/host \
  --payload examples/payload.txt \
  --output-folder results/batch
```

## 7. Benchmark and audit

```bash
dct-schur benchmark \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods all \
  --suite extended \
  --output-json results/benchmark.json \
  --output-summary-csv results/benchmark_summary.csv \
  --output-attack-csv results/benchmark_attacks.csv

dct-schur audit --root . --output results/repository_audit.json
```

`--host` and `--payload` accept either files or folders. Use `--methods blind`, `semi_blind`, `key_assisted`, `non_blind`, `baselines`, `dct_schur`, `common_4096`, or a comma-separated set of canonical IDs.

## 8. Published baseline pipeline

```bash
dct-schur baseline-embed \
  --method dct_dm_qim_chen2001_blind \
  --host data/host/lenna.bmp \
  --payload data/watermark/wm.png \
  --output results/dm_qim.png \
  --key results/dm_qim.pkl

dct-schur baseline-extract \
  --image results/dm_qim.png \
  --key results/dm_qim.pkl \
  --output results/dm_qim_recovered.png
```

The saved key bundle includes the original host only for methods whose extraction contract requires it.

## 9. Checkpointed publication benchmark

```bash
dct-schur benchmark-checkpointed \
  --host data/host \
  --payload data/watermark/wm.png \
  --methods all \
  --suite extended \
  --output-directory results/checkpointed_full
```

This pipeline writes one JSON and two CSV files per method and resumes from `index.json` after interruption.
