"""P_L direction-consistency validation for the construction-space objective.

The construction-space objective (`optimizer.py::dem_hook_stats.logical_weight`)
is a *DEM-level proxy*: the probability-weighted sum of all error paths that
flip a logical observable. The previous round claimed a ~20-23% reduction of
that proxy by switching the logical storage basis from ``Z`` to ``X`` on the BB
``custom`` construction. This script checks that claim against the real
logical failure probability measured with an actual BP-OSD decoder, under the
locked PLAN.md §5 protocol:

  * circuits : ``code.build_circuit(strategy, num_rounds, basis=basis)``
  * noise    : SD6 == ``quits.ErrorModel(idle=sq=tq=spam=eps)``
               (identical to ``baseline_replay.py``), eps in {1e-3,3e-3,1e-2}
  * decoder  : BP-OSD sliding window (Huang & Puri, PRA 110, 012453 (2024)),
               W=1, F=1, max_iter=2, osd_order=0, ``osd_method='osd_cs'`` --
               the very same configuration ``baseline_replay.py`` uses.
  * shots    : >= 1e6 target, reported with a 95% Wilson CI.

Only the logical storage basis changes between the two arms of a pair, so
``P_L(basis=X) - P_L(basis=Z)`` is directly comparable to
``logical_weight(X) - logical_weight(Z)``.

Parallelism
-----------
``quits.sliding_window_circuit_mem`` decodes shots in a plain Python ``for``
loop, which is the bottleneck (200 shots/s single-threaded on BB/bb-108 on a
2-core laptop). This module re-implements the *same* per-shot recursion on top
of the same ``quits.decoder.base.spacetime`` matrices and the same
``ldpc.bposd_decoder.BpOsdDecoder``, and shards the shot range over a process
pool. Per-shot decoder state (``syn_update`` / ``accumulated_correction``) is
reset at every shot, so sharding is exact, not an approximation;
``--check-parity`` asserts bit-for-bit equality against the QUITS reference
implementation on the same samples.

Usage:
  python pl_validation.py --family BB --name bb-108 --strategy custom \
      --rounds 3 --eps 1e-3,3e-3,1e-2 --bases Z,X --shots 1000000
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ISCAS" / "src"))
sys.path.insert(0, str(ROOT / "ISCAS" / "experiments"))

from baseline_replay import build_code, wilson_ci  # noqa: E402
from circuit_metrics import circuit_metrics  # noqa: E402

from ldpc.bposd_decoder import BpOsdDecoder  # noqa: E402
from quits import ErrorModel  # noqa: E402
from quits.decoder.base import spacetime  # noqa: E402

RESULTS = ROOT / "ISCAS" / "results" / "pl_validation.jsonl"

# decoder configuration, locked to baseline_replay.py
DECODER_KWARGS = dict(bp_method="product_sum", max_iter=2, schedule="serial",
                      osd_method="osd_cs", osd_order=0)

# ---------------------------------------------------------------- parallel
_W: dict = {}


def _init_worker(check_set, obs_set, priors_set, update, num_cor_rounds,
                 hz_rows, W, F, decoder_kwargs):
    decoders = []
    for i in range(len(check_set) - 1):
        kw = dict(decoder_kwargs)
        kw["channel_probs"] = priors_set[i]
        decoders.append(BpOsdDecoder(check_set[i], **kw))
    kw = dict(decoder_kwargs)
    kw["channel_probs"] = priors_set[len(check_set) - 1]
    decoders.append(BpOsdDecoder(check_set[len(check_set) - 1], **kw))
    _W.update(decoders=decoders, obs_set=obs_set, update=update,
              num_cor_rounds=num_cor_rounds, hz_rows=hz_rows, W=W, F=F)


def _decode_chunk(payload):
    """Faithful copy of quits.decoder.sliding_window.sliding_window_circuit_mem's
    per-shot body; returns (failures, n) for this chunk."""
    samples, obs = payload
    decoders = _W["decoders"]
    obs_set = _W["obs_set"]
    update = _W["update"]
    num_cor_rounds = _W["num_cor_rounds"]
    hz_rows = _W["hz_rows"]
    W, F = _W["W"], _W["F"]
    fails = 0
    for i in range(samples.shape[0]):
        sample = samples[i]
        accumulated_correction = np.zeros(obs_set[0].shape[0], dtype=int)
        syn_update = np.zeros(hz_rows, dtype=int)
        for k in range(num_cor_rounds):
            diff_syndrome = sample[F * k * hz_rows:(F * k + W) * hz_rows].copy() % 2
            diff_syndrome[:hz_rows] = (diff_syndrome[:hz_rows] + syn_update) % 2
            decoded_errors = decoders[k].decode(diff_syndrome)
            m = obs_set[k].shape[1]
            correction = obs_set[k] @ decoded_errors[:m] % 2
            syn_update = update[k] @ decoded_errors[:m] % 2
            accumulated_correction = (accumulated_correction + correction) % 2
        diff_syndrome = sample[F * num_cor_rounds * hz_rows:].copy() % 2
        diff_syndrome[:hz_rows] = (diff_syndrome[:hz_rows] + syn_update) % 2
        decoded_errors = decoders[num_cor_rounds].decode(diff_syndrome)
        correction = obs_set[num_cor_rounds] @ decoded_errors % 2
        accumulated_correction = (accumulated_correction + correction) % 2
        if np.any(np.asarray(accumulated_correction).ravel() != obs[i]):
            fails += 1
    return fails, samples.shape[0]


def decode_failures(pool, samples, obs, init_args, chunk):
    n = samples.shape[0]
    payloads = []
    for s in range(0, n, chunk):
        payloads.append((samples[s:s + chunk], obs[s:s + chunk]))
    if pool is None:
        _init_worker(*init_args)
        total = 0
        for p in payloads:
            total += _decode_chunk(p)[0]
        return total, n
    return sum(f for f, _ in pool.map(_decode_chunk, payloads)), n


def run_memory(code, circuit, basis, shots, seed, window, workers, batch, chunk,
               check_parity=False):
    """Sample + BP-OSD decode one (circuit, basis) pair. Returns stats dict."""
    h, l = (code.hz, code.lz) if basis == "Z" else (code.hx, code.lx)
    num_rounds = circuit.num_detectors // h.shape[0] - 2
    num_cor_rounds = (2 + num_rounds - window) // window
    if (2 + num_rounds - window) % window != 0:
        num_cor_rounds += 1

    t0 = time.time()
    check_set, obs_set, priors_set, update = spacetime(circuit, h, window,
                                                       window, num_cor_rounds)
    init_args = (check_set, obs_set, priors_set, update, num_cor_rounds,
                 h.shape[0], window, window, dict(DECODER_KWARGS))

    sampler = circuit.compile_detector_sampler(seed=seed)
    pool = None
    if workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=init_args)
    else:
        _init_worker(*init_args)

    failures = 0
    done = 0
    parity = None
    while done < shots:
        nb = min(batch, shots - done)
        samples, obs = sampler.sample(shots=nb, separate_observables=True)
        if check_parity and parity is None:
            from quits import sliding_window_bposd_circuit_mem
            ref = sliding_window_bposd_circuit_mem(samples, circuit, h, l,
                                                   W=window, F=window, **DECODER_KWARGS)
            f_ref = int(np.count_nonzero(np.any(ref != obs, axis=1)))
            f_par, _ = decode_failures(pool, samples, obs, init_args, chunk)
            parity = {"reference_failures": f_ref, "parallel_failures": f_par,
                      "equal": f_ref == f_par, "shots": nb}
            if not parity["equal"]:
                raise RuntimeError(f"parallel decoder diverged from QUITS: {parity}")
            failures += f_ref          # identical by the assertion above
        else:
            f_batch, _ = decode_failures(pool, samples, obs, init_args, chunk)
            failures += f_batch
        done += nb
    if pool is not None:
        pool.close()
        pool.join()
    dt = time.time() - t0
    lo, hi = wilson_ci(failures, shots)
    return {"basis": basis, "shots": shots, "failures": failures, "p_l": failures / shots,
            "p_l_ci95": [lo, hi], "num_rounds": num_rounds,
            "num_windows": num_cor_rounds + 1, "wall_clock_s": round(dt, 1),
            "shots_per_s": round(shots / dt, 1), "parity_check": parity}


def sd6(eps: float) -> ErrorModel:
    """SD6 circuit-level noise: one uniform depolarizing eps on idle, 1q, 2q
    and SPAM -- the same object ``baseline_replay.py`` locks."""
    return ErrorModel(idle_error=eps, sqgate_error=eps, tqgate_error=eps,
                      spam_error=eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--bases", default="Z,X")
    ap.add_argument("--eps", default="1e-3,3e-3,1e-2")
    ap.add_argument("--shots-map", default=None,
                    help="per-eps shot budget, e.g. '1e-3:1000000,3e-3:400000'. "
                         "Falls back to --shots for eps not listed.")
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--batch", type=int, default=25_000)
    ap.add_argument("--chunk", type=int, default=5_000)
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--tag", default="pl_validation")
    ap.add_argument("--corpus", default=None,
                    help="corpus json (default shared/C0/corpus.json; "
                         "pass shared/C1/corpus.json for the HGP optimization set)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    budget = {}
    if args.shots_map:
        for item in args.shots_map.split(","):
            k, v = item.split(":")
            budget[float(k)] = int(v)

    code = build_code(args.family, args.name, args.corpus)
    out = Path(args.output) if args.output else RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)

    for eps in [float(e) for e in args.eps.split(",") if e.strip()]:
        for basis in [b.strip() for b in args.bases.split(",") if b.strip()]:
            circuit = code.build_circuit(strategy=args.strategy,
                                         num_rounds=args.rounds, basis=basis,
                                         error_model=sd6(eps))
            m = circuit_metrics(circuit)
            shots = budget.get(eps, args.shots)
            res = run_memory(code, circuit, basis, shots, args.seed, args.window,
                             args.workers, args.batch, args.chunk,
                             check_parity=args.check_parity)
            record = {
                "tag": args.tag, "family": args.family, "name": args.name,
                "strategy": args.strategy, "rounds": args.rounds,
                "noise": "SD6", "eps": eps, "decoder": "BP-OSD",
                "decoder_kwargs": DECODER_KWARGS, "window": args.window,
                "seed": args.seed,
                "rounds_consistent": res["num_rounds"] == args.rounds,
                **m, **res,
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in
                              ("name", "strategy", "basis", "eps", "shots",
                               "failures", "p_l", "p_l_ci95", "wall_clock_s",
                               "shots_per_s", "num_detectors")}, indent=2),
                  flush=True)


if __name__ == "__main__":
    main()
