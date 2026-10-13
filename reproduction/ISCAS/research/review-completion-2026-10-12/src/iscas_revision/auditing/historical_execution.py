"""Exact arithmetic execution validation for all historical HGP windows."""
from datetime import datetime, timezone
import time

import numpy as np
import stim

from ..artifacts import WORK, HISTORICAL, commit_json, digest
from ..decoding import CachedDecoder
from ..execution.decoder import ExecutionDecoder
from .execution import CheckedCalls
from support import code_for


def run():
    matrices = []
    for instance in ("tiny-hgp", "holdout-hgp-3x4"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            circuit_path = HISTORICAL/"provenance/decoder-priors"/instance/(order+".stim")
            circuit = stim.Circuit(circuit_path.read_text(encoding="utf-8"))
            faults = dict(np.load(HISTORICAL/"results"/instance/order/"single-fault.npz"))
            native = CachedDecoder(code_for(instance), circuit, single_fault=faults)
            optimized = ExecutionDecoder(code_for(instance), circuit, single_fault=faults, accelerate_original=True)
            calls = []
            for label in native.states:
                optimized.states[label]["decoders"] = [CheckedCalls(a, b, label+str(i), calls) for i, (a, b) in enumerate(
                    zip(native.states[label]["decoders"], optimized.states[label]["decoders"]))]
            with np.load(WORK/"results/direct"/instance/order/"eps0.0001/physical.npz") as f:
                records = f["detectors"][:32]
            indices = np.linspace(0, len(faults["detectors"])-1, 8, dtype=int)
            records = np.concatenate([records, faults["detectors"][indices], np.zeros((1, circuit.num_detectors), dtype=np.uint8)])
            start = time.perf_counter()
            expected = native.predictions(records)
            actual = optimized.predictions(records)
            assert all(np.array_equal(expected[label], actual[label]) for label in expected)
            row = dict(instance=instance, order=order, checked_records=len(records), windows=calls,
                logical_predictions_equal=True, prior_sha256=digest(circuit_path), seconds=time.perf_counter()-start)
            matrices.append(row)
            commit_json(WORK/"provenance/historical-execution-validation-progress.json", dict(completed=len(matrices), matrices=matrices))
            print(dict(instance=instance, order=order, checked=True, windows=len(calls)), flush=True)
    commit_json(WORK/"provenance/historical-execution-validation.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        matrices=matrices, scientific_protocol_unchanged=True,
        optimized_source_sha256={str(p): digest(p) for p in (WORK/"src/iscas_revision/execution").glob("*.py")},
        source_sha256=digest(__file__)))


if __name__ == "__main__":
    run()
