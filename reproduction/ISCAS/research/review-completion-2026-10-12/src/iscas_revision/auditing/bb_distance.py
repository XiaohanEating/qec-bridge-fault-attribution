"""Exact task-specific absence of one- and two-fault invisible logical products."""
from datetime import datetime, timezone

import numpy as np

from ..artifacts import WORK, commit_json, digest, read_json


def run():
    rows = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            with np.load(folder/"single-fault.npz") as f:
                seen = {}
                zero_nonzero = 0
                conflicting = 0
                for d, l, mass in zip(f["detectors"], f["observables"], f["physical_coefficients"]):
                    assert mass>0
                    key = np.packbits(d, bitorder="little").tobytes()
                    value = np.packbits(l, bitorder="little").tobytes()
                    zero_nonzero += int(not d.any() and l.any())
                    if key in seen:
                        conflicting += int(seen[key] != value)
                    else:
                        seen[key] = value
            assert zero_nonzero == conflicting == 0
            assert read_json(folder/"physical-risk.json")["R_star"] == 0
            rows.append(dict(instance=instance, order=order, zero_record_nonzero_logical_signatures=0,
                equal_record_unequal_logical_pairs=0, observed_task_fault_distance_lower_bound=3,
                logical_task="all saved logical-Z memory observables, initial plus three noisy extraction blocks",
                scope="Lower bound for the declared memory task; full quantum-code fault distance and exact minimum remain separate",
                single_fault_sha256=digest(folder/"single-fault.npz"), prior_sha256=digest(folder/"prior.stim")))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), circuits=rows,
        proof="Every physical single-fault joint response is enumerated. Two-fault zero detector product requires equal D. Each D has one L, hence L1 XOR L2=0. The zero D label is zero. Mutually exclusive branches of one location are not double counted.")
    commit_json(WORK/"results/bb-memory-distance-lower-bound.json", result)
    print(dict(circuits=len(rows), observed_memory_fault_distance_lower_bound=3), flush=True)


if __name__ == "__main__":
    run()
