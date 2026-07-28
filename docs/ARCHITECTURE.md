# Repository architecture

The repository uses a root-package layout. There is no `src/` directory. It contains one proposal and a separately isolated baseline library.

```text
dct_schur/
├── math/          Schur geometry, QIM, coset optimization, layouts
├── engine/        Embedding, evidence, restoration, decoding
├── transport/     ECC/CRC transport for bytes, text, and JSON
├── provenance/    Ed25519 records and sequence verification
├── config.py
├── key.py
├── philosophy.py
└── cli.py

baselines/
├── common/          Shared baseline-only DCT/DWT/IWT/QWT utilities
├── implementations/
│   ├── dct/         DM-QIM, STDM-QIM, ISS, CA-QIM, Cox, DEW
│   └── transform/   DWT, Hessenberg, SVD, QWT, IWT methods
├── vendor/          Preserved third-party mathematical helper code
├── evidence/        Method provenance and fidelity disclosures
├── metadata.py
├── registry.py
└── evaluation.py

benchmarking/
├── registry.py       One proposal plus selectable baseline specifications
├── adapters.py       Narrow common embed/extract interface
├── runner.py         Trial and matrix execution
├── parameters.py     Baseline parameter loading
├── output.py         JSON and CSV writers
└── types.py

pipelines/
├── image.py          DCT-Schur binary-image compatibility
├── baseline.py       Standalone baseline embed/extract
├── data.py           Arbitrary data, text, and JSON
├── provenance.py     Signed-image provenance
├── document.py       Chained rendered pages
├── video.py          Chained video frames
├── batch.py          Folder processing
├── benchmark.py      Multi-method comparison orchestration
└── audit.py          Proposal/baseline/structure validation

evaluation/
├── attacks/          Deterministic common attack library
└── metrics.py        Common image and watermark metrics
```

## Dependency direction

```text
DCT-Schur core: math → engine → transport → provenance → pipelines

Baselines: common utilities → implementations → registry

Comparison: dct_schur ─┐
                       ├→ benchmarking → evaluation
             baselines ┘
```

The proposal and baselines do not call each other. The benchmark adapter is the only integration boundary.

## One proposal rule

`benchmarking.registry` exposes exactly one `method_kind="proposal"` entry:

```text
dct_schur_invariant_relational
```

All other entries are `method_kind="baseline"`. Repository audit checks this rule, verifies all 16 baseline IDs, rejects the old `src/` layout, and confirms no folder exceeds the configured file limit.
