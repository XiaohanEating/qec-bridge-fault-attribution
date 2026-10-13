"""Measure unchanged native decoder calls on saved BB inputs."""
from datetime import datetime, timezone
import time

import numpy as np
import stim

from ..artifacts import WORK, commit_json, digest
from ..baselines import bb_code
from ..decoding import CachedDecoder


class TimedCalls:
    def __init__(self, native, stage, rows, columns):
        self.native = native
        self.stage = stage
        self.rows = rows
        self.columns = columns
        self.calls = []

    def decode(self, syndrome):
        start = time.perf_counter()
        value = self.native.decode(syndrome)
        self.calls.append(dict(weight=int(np.count_nonzero(syndrome)),
            seconds=time.perf_counter()-start,
            syndrome=np.packbits(syndrome, bitorder="little").tobytes().hex()))
        return value


def run():
    folder = WORK/"results/bb/bb-72/forward"
    with np.load(folder/"single-fault.npz") as f:
        faults = dict(f)
    decoder = CachedDecoder(bb_code("bb-72"), stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8")), single_fault=faults)
    timers = {}
    for label, state in decoder.states.items():
        timers[label] = [TimedCalls(native, k, state["hz_rows"]*(state["W"] if k<state["num_cor_rounds"] else 1),
            state["obs_set"][k].shape[1]) for k, native in enumerate(state["decoders"])]
        state["decoders"] = timers[label]
    chosen = np.linspace(1800, len(faults["detectors"])-1, 4, dtype=int).tolist()
    points = [("single-fault", faults["detectors"][chosen])]
    high = folder/"eps0.0001/physical.npz"
    with np.load(high) as f:
        points.append(("finite-0.0001", f["detectors"][:4]))
    outputs = []
    for name, records in points:
        started = time.perf_counter()
        predictions = decoder.predictions(records)
        outputs.append(dict(kind=name, count=len(records), seconds=time.perf_counter()-started,
            predictions={label: values.tolist() for label, values in predictions.items()}))
    stages = {label: [dict(stage=timer.stage, rows=timer.rows, columns=timer.columns, calls=timer.calls) for timer in values]
        for label, values in timers.items()}
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), stages=stages, outputs=outputs,
        single_fault_rows=chosen, inputs={name: digest(folder/name) for name in ("prior.stim", "single-fault.npz")},
        scientific_rules_unchanged=True)
    commit_json(WORK/"provenance/bb-native-call-performance.json", result)
    print({label: [dict(stage=t.stage, seconds=sum(x["seconds"] for x in t.calls), calls=len(t.calls),
        unique_syndromes=len({x["syndrome"] for x in t.calls})) for t in values] for label, values in timers.items()}, flush=True)


if __name__ == "__main__":
    run()
