# Release verification

Verification on 13 October 2026 used an independent clone of the public repository and the unchanged experiment environment. The tested code commit was `0802738d85e0bdb0c86dd557b6fdfa9cff5abed8`. This document and its accompanying record were added after those checks.

## Source and public data

- All 670 original scientific source/input files matched their SHA-256 manifests after cloning. Git preserves their bytes across platforms.
- All seven public v1.0.0 assets matched the original sizes and GitHub SHA-256 digests: 522,684,395 compressed bytes, 11,798,659,635 restored bytes, and 24,285 scientific files.
- The core archive was downloaded through its public URL without a GitHub credential and all 24,279 members were verified during restoration. All seven archives had also been restored locally with every member checked.
- The six complete BB fault-trace archives retain physical-coordinate enumeration. Their public digests were checked; the independent audit uses the core archive.

## Numerical and execution checks

The independent audit recomputed six historical and six BB single-fault risk sets. It checked 6,428 preserved sample records and 19,273 prediction/residual arrays, including original sample hashes and recorded failure counts. These artifact counts include historical controls. The final scientific experiment acceptance remains 324 effective jobs, 6,416 accepted result records, and 19,165 accepted prediction arrays.

Both paper tables were regenerated, and all six rows agreed with the manuscript at its displayed precision. Figure 2 was regenerated as PDF, PNG, and SVG.

Registered task `batch1-0148-direct` was rerun at epsilon `1e-6`, seed `520409012`, and 32,768 shots. Detector samples, observables, the seed, all three decoder predictions, logical residuals, and decoder statistics matched the published results exactly. This task used the original decoder implementation and configuration.

Official Hirai and Zhao repositories were fetched and checked at their pinned commits. Windows verification included Hirai paths up to 289 characters; every tracked file was present and both checkouts were clean. Version 1.0.1 enables Git long-path handling before checkout. Failed attempts were preserved separately during verification.

## Reproduction commands

From a clone with the README's pinned environment and editable installation:

```bash
qec-bridge verify
qec-bridge fetch-data
qec-bridge audit --output outputs/audit
qec-bridge summarize --output outputs/summary
qec-bridge replay --id batch1-0148-direct --output outputs/replay
qec-bridge fetch-upstream
```

For all complete traces and member identities:

```bash
qec-bridge fetch-data --all
qec-bridge verify --data
```

Machine-readable results are in `provenance/release-verification.json`. The code patch and verification documentation preserve the v1.0.0 scientific source manifests and data assets.
