"""Verify sparse support scoring keeps every OSD candidate and final choice."""
from datetime import datetime, timezone
import contextlib
import io
import time

import numpy as np
import stim
from quits.decoder.base import spacetime

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..execution.linear_osd import LinearOsd, linear_sweep, candidates, sparse_candidates


def run():
    rng = np.random.default_rng(99131012)
    # Include zero-cost and tied candidates, rank deficient cases, and arbitrary
    # column bases; both implementations must use the same strict comparison.
    for trial in range(240):
        n = int(rng.integers(4, 72))
        rank = int(rng.integers(1, n))
        columns = rng.permutation(n)
        pivots, free = columns[:rank], columns[rank:]
        reduced = rng.integers(2, size=(rank, n), dtype=np.uint8)
        rhs = rng.integers(2, size=rank, dtype=np.uint8)
        costs = rng.choice(np.array([0., .1, 1., 3.7, 10.]), size=n)
        order = int(rng.integers(4))
        expected = candidates(reduced, pivots, free, rhs, costs, order)
        actual = sparse_candidates(reduced, pivots, free, rhs, costs, order)
        assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
    matrices=[]
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            code = bb_code(instance)
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _, priors, _ = spacetime(circuit, code.hz, 4, 4, 1)
            with np.load(folder/"eps0.0001/physical.npz") as sample:
                records = sample["detectors"][:16, :checks[0].shape[0]]
            decoder = LinearOsd(checks[0], priors[0], exact_bp=True, prefix_reuse=True, zero_shortcut=True)
            rows=[]
            for record in records:
                decoder.bp.decode(record)
                t=time.perf_counter()
                expected=linear_sweep(decoder.check,record,decoder.bp.log_prob_ratios,decoder.costs,2)
                old=time.perf_counter()-t
                t=time.perf_counter()
                actual=linear_sweep(decoder.check,record,decoder.bp.log_prob_ratios,decoder.costs,2,sparse_support=True)
                new=time.perf_counter()-t
                assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
                rows.append(dict(original_seconds=old,sparse_seconds=new,osd0_and_final_equal=True))
            matrices.append(dict(instance=instance,order=order,records=rows))
            commit_json(WORK/"provenance/sparse-support-validation-progress.json",dict(completed=len(matrices),total=6,matrices=matrices))
            print(dict(instance=instance,order=order,original_seconds=sum(r["original_seconds"]for r in rows),
                sparse_seconds=sum(r["sparse_seconds"]for r in rows)),flush=True)
    commit_json(WORK/"provenance/sparse-support-validation.json",dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        matrices=matrices,randomized_cases=240,source_sha256=digest(WORK/"src/iscas_revision/execution/linear_osd.py"),
        validation_source_sha256=digest(__file__),running_snapshots_modified=False))


if __name__ == "__main__":
    run()
