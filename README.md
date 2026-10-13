# QEC bridge fault attribution

Code and experiment data for **Mechanism Analysis of Gate-Order Effects in Resource-Matched Quantum Error-Correction Bridge Circuits**.

The artifact separates physical single-fault ambiguity from decoder-induced decision loss, traces both to physical fault coordinates, and checks their finite-noise consequences. It includes the primary surface-code bridges, independent ordering selection, decoder controls, nonlocal BB-72/BB-108 circuits, and the reported hardware controls.

## Reproduce

Use Python 3.11.17 and the exact package versions in `environment/requirements.lock.txt`. The experiment versions are Stim 1.16.0, ldpc 2.4.1, QUITS 1.2.0, NumPy 2.4.6, and SciPy 1.17.1. Install in a dedicated environment:

```bash
python -m pip install -r environment/requirements.lock.txt
python -m pip install --no-deps -e .
qec-bridge verify
qec-bridge fetch-data
qec-bridge audit --output outputs
qec-bridge summarize --output outputs
```

`fetch-data` downloads the core scientific data from [release v1.0.0](https://github.com/XiaohanEating/qec-bridge-fault-attribution/releases/tag/v1.0.0), checks SHA-256, and restores original relative paths. `audit` recomputes the reported single-fault risks and verifies saved predictions, logical residuals, sample hashes, and failure counts. `summarize` regenerates both paper tables as JSON/CSV and Figure 2 as PDF/PNG/SVG. Every figure annotation uses a translucent white background.

The six large BB fault-trace archives are also public. They preserve the complete physical-coordinate enumeration and are unnecessary for the first risk/count audit:

```bash
qec-bridge fetch-data --all
qec-bridge verify --data
```

Archive sizes, checksums, and every original file identity are listed in `data/archive-manifest.json`. The core includes all conditional sampling layers, all eight BB noise rates, raw detector/observable samples, saved predictions, single-fault masses, circuit inputs, and statistical comparisons. Large traces contain all retained physical fault coordinates. The repository omits duplicate intermediate decoder partitions and local runtime logs; their committed merged scientific arrays are included.

## Run a registered experiment again

```bash
qec-bridge tasks --kind direct
qec-bridge replay --id batch1-0148-direct --output outputs/replay
```

The replay keeps the registered seed, shot count, prior, decoder configuration, and physical circuit. It writes fresh output outside the published inputs. `data/registered-tasks.json` lists all 324 effective completed jobs. Original batch registrations and source snapshots preserve historical versions and superseded Zhao tasks separately. Successful computations from the old Zhao adapter remain historical records, not validated controls.

Official Hirai and Zhao controls use their original repositories at fixed commits:

```bash
qec-bridge fetch-upstream
```

Version 1.0.1 enables Git long-path checkout for upstream repositories on Windows. Scientific source and data remain the v1.0.0 artifacts.

See `docs/upstream.md` for sources and conditions. BB enumeration and full finite-noise decoding are substantial computations; the original protocol and complete saved results allow their scope to be inspected before replay. Validated execution accelerators and their equality evidence are retained under `reproduction/ISCAS/research/review-completion-2026-10-12/src/iscas_revision/execution` and `provenance`. The basic replay uses the original decoder implementation.

## Layout

| Path | Content |
|---|---|
| `src/qec_bridge_artifact/` | Portable download, verification, audit, and replay commands |
| `reproduction/` | Original scientific source, source snapshots, circuit definitions, and result records |
| `data/` | Archive manifest and registered task definitions |
| `environment/` | Exact package versions |
| `docs/` | Statistical protocol, result interpretation, and upstream attribution |
| `provenance/` | Original-source hashes and release verification |

Original scientific modules are copied without rewriting their mathematics or decoder logic. The command layer supplies relative paths and directs new output to a separate folder. Original JSON records may contain historical Windows paths as provenance; reproduction commands resolve inputs within this artifact.

The paper uses four noisy extraction blocks and ideal final data readout. A logical failure is any nonzero logical residual. Resource-matched comparisons preserve ideal signed block action, directed-edge CX counts, time slots, and cumulative per-site idling. Keep the original-prior, matched-prior, and author-noise results distinct. The risk decomposition follows weighted classification; physical fault attribution and controlled circuit comparisons provide the experimental diagnosis.

## License and citation

The authors' source is MIT licensed. Generated data are CC BY 4.0, with upstream code and circuit sources retaining their original terms. See `LICENSE`, `LICENSE-DATA.md`, and `docs/upstream.md`.

Please cite the paper title and this repository, release v1.0.0.

Independent download and replay evidence is documented in [docs/verification.md](docs/verification.md). Use code v1.0.1 with data release v1.0.0.
