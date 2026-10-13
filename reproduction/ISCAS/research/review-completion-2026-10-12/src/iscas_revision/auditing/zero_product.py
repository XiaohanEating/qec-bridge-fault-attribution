"""Validate exact zero-product reuse against the full ordered multiplication."""
from datetime import datetime, timezone
import os
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
    stem = os.environ.get("ISCAS_REVIEW_SHORTCUT_REPORT", "zero-product")
    matrices = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            code = bb_code(instance)
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _, priors, _ = spacetime(circuit, code.hz, 4, 4, 1)
            with np.load(folder/"eps0.0001/physical.npz") as f:
                records = f["detectors"][:12, :4*code.hz.shape[0]]
            first = ExactSerialBp(checks[0], priors[0], 50, zero_shortcut=False)
            second = ExactSerialBp(checks[0], priors[0], 50, zero_shortcut=True)
            rows = []
            for record in records:
                t = time.perf_counter()
                expected = first.decode(record)
                old = time.perf_counter()-t
                t = time.perf_counter()
                actual = second.decode(record)
                new = time.perf_counter()-t
                assert np.array_equal(expected, actual)
                assert np.array_equal(first.log_prob_ratios, second.log_prob_ratios, equal_nan=True)
                assert first.converge == second.converge and first.iter == second.iter
                rows.append(dict(full_seconds=old, shortcut_seconds=new, exact_equal=True))
            matrices.append(dict(instance=instance, order=order, records=rows))
            commit_json(WORK/"provenance"/(stem+"-validation-progress.json"), dict(completed=len(matrices), total=6, matrices=matrices))
            print(dict(instance=instance, order=order, full_seconds=sum(r["full_seconds"] for r in rows),
                shortcut_seconds=sum(r["shortcut_seconds"] for r in rows)), flush=True)
    commit_json(WORK/"provenance"/(stem+"-validation.json"), dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        matrices=matrices, source_sha256=digest(WORK/"src/iscas_revision/execution/serial_bp.py")))


if __name__ == "__main__":
    run()
