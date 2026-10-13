"""Compare prefix reuse and exactly rounded zero messages with native BP--OSD."""
from datetime import datetime, timezone
import time

import numpy as np
import stim

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..decoding import CachedDecoder
from ..execution.decoder import ExecutionDecoder
from .execution import CheckedCalls


def run():
    matrices = []
    began = time.perf_counter()
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            faults = dict(np.load(folder/"single-fault.npz"))
            native = CachedDecoder(bb_code(instance), circuit, single_fault=faults)
            optimized = ExecutionDecoder(bb_code(instance), circuit, single_fault=faults,
                accelerate_original=True, prefix_reuse=True, zero_shortcut=True)
            rows = []
            for label in native.states:
                optimized.states[label]["decoders"] = [CheckedCalls(a, b, (label, i), rows)
                    for i, (a, b) in enumerate(zip(native.states[label]["decoders"], optimized.states[label]["decoders"]))]
            records = [np.zeros(circuit.num_detectors, dtype=np.uint8)]
            indices = np.linspace(0, len(faults["detectors"])-1, 4, dtype=int)
            records.extend(faults["detectors"][indices])
            for point in ("eps1e-07", "eps1e-06", "eps0.0001"):
                with np.load(folder/point/"physical.npz") as sample:
                    records.extend(sample["detectors"][:3])
            records = np.asarray(records, dtype=np.uint8)
            expected = native.predictions(records)
            actual = optimized.predictions(records)
            assert set(expected) == set(actual)
            assert all(np.array_equal(expected[label], actual[label]) for label in expected)
            matrices.append(dict(instance=instance, order=order, records=len(records), windows=rows,
                prior_sha256=digest(folder/"prior.stim"), all_logical_predictions_equal=True,
                original_accelerated=True, prefix_reuse=True, rounded_zero_shortcut=True))
            commit_json(WORK/"provenance/prefix-execution-validation-progress.json", dict(
                completed=len(matrices), total=6, matrices=matrices, elapsed_seconds=time.perf_counter()-began))
            print(dict(instance=instance, order=order, checked=True, calls=len(rows)), flush=True)
    commit_json(WORK/"provenance/prefix-execution-validation.json", dict(
        updated_utc=datetime.now(timezone.utc).isoformat(), matrices=matrices,
        scientific_protocol_unchanged=True, running_snapshots_modified=False,
        optimized_source_sha256={str(p):digest(p) for p in (WORK/"src/iscas_revision/execution").glob("*.py")},
        validation_source_sha256=digest(__file__), elapsed_seconds=time.perf_counter()-began))


if __name__ == "__main__":
    run()
