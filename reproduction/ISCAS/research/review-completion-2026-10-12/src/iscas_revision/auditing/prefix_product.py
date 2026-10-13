"""Compare ordered prefix reuse with the complete scalar multiplication."""
from datetime import datetime, timezone
import contextlib
import io
import time

import numpy as np
import stim
from quits.decoder.base import spacetime

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..execution.serial_bp import ExactSerialBp


def run():
    matrices = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            code = bb_code(instance)
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _, priors, _ = spacetime(circuit, code.hz, 4, 4, 1)
            windows = []
            for index, (check, prior) in enumerate(zip(checks, priors)):
                width = check.shape[0]
                with np.load(folder/"eps0.0001/physical.npz") as f:
                    # First strong window includes four detector blocks. The
                    # terminal window is independently checked with its block.
                    records = f["detectors"][:16, :width] if index == 0 else f["detectors"][:16, -width:]
                full = ExactSerialBp(check, prior, 50)
                reuse = ExactSerialBp(check, prior, 50, prefix_reuse=True)
                rows = []
                for record in records:
                    t = time.perf_counter()
                    expected = full.decode(record)
                    old = time.perf_counter()-t
                    t = time.perf_counter()
                    actual = reuse.decode(record)
                    new = time.perf_counter()-t
                    assert np.array_equal(expected, actual)
                    assert np.array_equal(full.log_prob_ratios, reuse.log_prob_ratios, equal_nan=True)
                    assert full.converge == reuse.converge and full.iter == reuse.iter
                    rows.append(dict(full_seconds=old, prefix_seconds=new, exact_equal=True))
                windows.append(dict(window=index, records=rows))
            matrices.append(dict(instance=instance, order=order, windows=windows))
            commit_json(WORK/"provenance/prefix-product-validation-progress.json", dict(completed=len(matrices), total=6, matrices=matrices))
            print(dict(instance=instance, order=order,
                full_seconds=sum(r["full_seconds"] for w in windows for r in w["records"]),
                prefix_seconds=sum(r["prefix_seconds"] for w in windows for r in w["records"])), flush=True)
    commit_json(WORK/"provenance/prefix-product-validation.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        matrices=matrices, execution_source_sha256=digest(WORK/"src/iscas_revision/execution/serial_bp.py"),
        validation_source_sha256=digest(__file__), running_snapshots_modified=False))


if __name__ == "__main__":
    run()
