"""Compare proposed linear OSD execution against unchanged native calls."""
from datetime import datetime, timezone
import os
import time

import numpy as np
import stim
from scipy.sparse import csc_matrix
from ldpc import BpOsdDecoder
from quits.decoder.base import spacetime

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..execution.linear_osd import LinearOsd, linear_sweep


def run():
    folder = WORK/"results/bb/bb-72/forward"
    code = bb_code("bb-72")
    circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
    checks, observations, priors, updates = spacetime(circuit, code.hz, 4, 4, 1)
    native = BpOsdDecoder(checks[0], channel_probs=priors[0], bp_method="product_sum", schedule="serial",
        max_iter=50, osd_method="osd_cs", osd_order=2)
    exact_bp = os.environ.get("ISCAS_REVIEW_EXACT_BP") == "1"
    suffix = "linear-osd-bp" if exact_bp else "linear-osd"
    fast = LinearOsd(checks[0], priors[0], exact_bp=exact_bp)
    with np.load(folder/"single-fault.npz") as f:
        rows = np.linspace(0, len(f["detectors"])-1, 12, dtype=int).tolist()
        records = f["detectors"][rows, :144]
    with np.load(folder/"eps0.0001/physical.npz") as f:
        records = np.concatenate([records, f["detectors"][:12, :144]])
    results = []
    for i, syndrome in enumerate(records):
        start = time.perf_counter()
        expected = native.decode(syndrome).copy()
        old_seconds = time.perf_counter()-start
        start = time.perf_counter()
        actual = fast.decode(syndrome).copy()
        new_seconds = time.perf_counter()-start
        assert np.array_equal(actual, expected), f"Native vector mismatch at row {i}"
        assert bool(native.converge) == bool(fast.bp.converge)
        if syndrome.any():
            assert np.array_equal(native.log_prob_ratios, fast.bp.log_prob_ratios, equal_nan=True), f"LLR mismatch at row {i}"
        if not native.converge:
            assert np.array_equal(native.osd0_decoding, fast.osd0_decoding), f"OSD0 mismatch at row {i}"
            assert np.array_equal(native.osdw_decoding, fast.osdw_decoding), f"OSDw mismatch at row {i}"
        results.append(dict(row=i, native_seconds=old_seconds, optimized_seconds=new_seconds,
            full_error_vector_equal=True, bp_converged=bool(native.converge), llr_equal=True))
        commit_json(WORK/"provenance"/(suffix+"-validation-progress.json"), dict(completed=len(results), total=len(records), records=results))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), records=results,
        inputs={name: digest(folder/name) for name in ("prior.stim", "single-fault.npz")},
        source_sha256=digest(__file__), optimized_source_sha256=digest(WORK/"src/iscas_revision/execution/linear_osd.py"),
        scope="BB72 forward first strong window, unchanged BP LLR and complete error vectors")
    commit_json(WORK/"provenance"/(suffix+"-validation.json"), result)
    print(dict(records=len(results), equal=True,
        native_seconds=sum(r["native_seconds"] for r in results),
        optimized_seconds=sum(r["optimized_seconds"] for r in results)), flush=True)


if __name__ == "__main__":
    run()
