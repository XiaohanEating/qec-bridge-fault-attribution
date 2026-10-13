"""Check all BB strong windows and full predictions against the native binary."""
from datetime import datetime, timezone
import contextlib
import io
import time

import numpy as np
import stim

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..decoding import CachedDecoder
from ..execution.decoder import ExecutionDecoder
from ..execution.linear_osd import LinearOsd


class CheckedCalls:
    def __init__(self, native, optimized, label, rows):
        self.native, self.optimized = native, optimized
        self.label, self.rows = label, rows

    def decode(self, syndrome):
        start = time.perf_counter()
        expected = self.native.decode(syndrome).copy()
        old = time.perf_counter()-start
        start = time.perf_counter()
        actual = self.optimized.decode(syndrome).copy()
        new = time.perf_counter()-start
        assert np.array_equal(expected, actual), f"Error vector mismatch: {self.label}"
        assert bool(self.native.converge) == bool(self.optimized.bp.converge)
        if syndrome.any():
            assert np.array_equal(self.native.log_prob_ratios, self.optimized.bp.log_prob_ratios, equal_nan=True), f"LLR mismatch: {self.label}"
        if not self.native.converge:
            assert np.array_equal(self.native.osd0_decoding, self.optimized.osd0_decoding)
            assert np.array_equal(self.native.osdw_decoding, self.optimized.osdw_decoding)
        self.rows.append(dict(window=self.label, syndrome=np.packbits(syndrome, bitorder="little").tobytes().hex(),
            native_seconds=old, optimized_seconds=new, complete_error_vector_equal=True,
            llr_equal=True, bp_converged=bool(self.native.converge)))
        return actual


def run():
    results = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            faults = dict(np.load(folder/"single-fault.npz")) if (folder/"single-fault.npz").exists() else None
            native = CachedDecoder(bb_code(instance), circuit, single_fault=faults)
            optimized = ExecutionDecoder(bb_code(instance), circuit, single_fault=faults)
            rows = []
            optimized.states["strong"]["decoders"] = [CheckedCalls(a, b, i, rows) for i, (a, b) in enumerate(
                zip(native.states["strong"]["decoders"], optimized.states["strong"]["decoders"]))]
            records = [np.zeros(circuit.num_detectors, dtype=np.uint8)]
            if faults is not None:
                indices = np.linspace(0, len(faults["detectors"])-1, 3, dtype=int)
                records.extend(faults["detectors"][indices])
            path = folder/"eps0.0001/physical.npz"
            if path.exists():
                records.extend(np.load(path)["detectors"][:3])
            else:
                # Read-only validation samples, never entered into scientific totals.
                records.extend(circuit.compile_detector_sampler(seed=981201012).sample(3))
            records = np.asarray(records, dtype=np.uint8)
            expected = native.predictions(records)
            actual = optimized.predictions(records)
            assert set(expected) == set(actual)
            assert all(np.array_equal(expected[label], actual[label]) for label in expected)
            results.append(dict(instance=instance, order=order, records=len(records), windows=rows,
                prior_sha256=digest(folder/"prior.stim"), all_logical_predictions_equal=True,
                labels=list(actual), single_fault_lookup_complete=faults is not None))
            commit_json(WORK/"provenance/optimized-execution-validation-progress.json", dict(completed=len(results), total=6, matrices=results))
            print(dict(instance=instance, order=order, checked=True, calls=len(rows)), flush=True)
    commit_json(WORK/"provenance/optimized-execution-validation.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        matrices=results, scientific_protocol_unchanged=True,
        optimized_source_sha256={str(p): digest(p) for p in (WORK/"src/iscas_revision/execution").glob("*.py")},
        validation_source_sha256=digest(__file__)))


if __name__ == "__main__":
    run()
