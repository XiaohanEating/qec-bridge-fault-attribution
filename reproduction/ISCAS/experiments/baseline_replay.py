"""ISCAS baseline replay harness (B1/B2 pipeline tail).

Replays a QUITS strategy on a C0 instance under the locked noise/decoder
protocol (PLAN.md §5): SD6-style ErrorModel at eps in {1e-3,3e-3,1e-2},
BP-OSD sliding-window decode, >=1e6 shots target (CLI-scalable).

Usage:
  python baseline_replay.py --family HGP --name hgp-seed3x3 --strategy cardinal \
      --rounds 3 --eps 1e-3 --shots 10000
Outputs one JSON line to results/replay_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from quits import BbCode, HgpCode, QlpCode, ErrorModel, get_stim_mem_result
from quits import sliding_window_bposd_circuit_mem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from circuit_metrics import circuit_metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "replay_log.jsonl"


def build_code(family: str, name: str, corpus_path=None):
    """Build a QUITS code object from a corpus file.

    ``corpus_path`` defaults to the shared C0 common core.  The C1
    optimization set (``shared/C1/corpus.json``) is passed explicitly by the
    HGP runs, which carry the real dv=3/dc=4 parity-check matrices; C1 re-lists
    the BB/QLP instances verbatim, so either corpus resolves the same code for
    those families.
    """
    if corpus_path is None:
        corpus_path = Path(__file__).resolve().parents[2] / "shared" / "C0" / "corpus.json"
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    inst = next(i for i in corpus["instances"] if i["family"] == family and i["name"] == name)
    ctor = inst["quits_ctor"]
    if family == "BB":
        return BbCode(**ctor)
    if family == "HGP":
        return HgpCode(np.array(ctor["h1"]), np.array(ctor["h2"]))
    if family == "QLP":
        return QlpCode(np.array(ctor["b1"]), np.array(ctor["b2"]), ctor["lift_size"])
    raise ValueError(family)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, default=10_000, help="target >=1e6 for paper numbers")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--osd-order", type=int, default=0)
    ap.add_argument("--corpus", default=None,
                    help="corpus json (default shared/C0/corpus.json; "
                         "pass shared/C1/corpus.json for the HGP optimization set)")
    args = ap.parse_args()

    code = build_code(args.family, args.name, args.corpus)
    em = ErrorModel(idle_error=args.eps, sqgate_error=args.eps,
                    tqgate_error=args.eps, spam_error=args.eps)
    circuit = code.build_circuit(strategy=args.strategy, num_rounds=args.rounds, error_model=em)
    metrics = circuit_metrics(circuit)

    samples, obs = get_stim_mem_result(circuit, args.shots, seed=args.seed)
    preds = sliding_window_bposd_circuit_mem(
        samples, circuit, code.hz, code.lz,
        W=args.window, F=args.window, max_iter=2, osd_order=args.osd_order,
    )
    failures = int(np.count_nonzero(np.any(preds != obs, axis=1)))
    p_l = failures / args.shots
    lo, hi = wilson_ci(failures, args.shots)

    record = {
        "family": args.family, "name": args.name, "strategy": args.strategy,
        "rounds": args.rounds, "eps": args.eps, "shots": args.shots, "seed": args.seed,
        "failures": failures, "p_l": p_l, "p_l_ci95": [lo, hi],
        **metrics,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
