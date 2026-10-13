"""Compare detector representations while holding the decoder prior fixed."""
import os
import time

import numpy as np
import stim

from ..artifacts import WORK, HISTORICAL, read_json, digest, commit_json
from ..audit import paired_comparison
from ..decoding import full_joint_state, full_joint_predict
from ..experiments import save_npz
from ..stratification import clopper_pearson


def run():
    batch = os.environ.get("ISCAS_REVIEW_BATCH", "full-model-fixed")
    registration = read_json(WORK/"provenance"/batch/"registration.json")
    assert all(digest(path) == sha for path, sha in registration["source_sha256"].items())
    results = []
    for task in registration["tasks"]:
        order = task["order"]
        source = WORK/"results/joint-bridge"/order/"eps0.0001"
        sample = dict(np.load(source/"physical.npz"))
        matched = dict(np.load(source/"decoded.npz"))
        prior = HISTORICAL/"provenance/decoder-priors/tiny-hgp"/(order+".stim")
        circuit = stim.Circuit(prior.read_text(encoding="utf-8"))
        faults = dict(np.load(HISTORICAL/"results/tiny-hgp"/order/"single-fault.npz"))
        assert circuit.num_observables == sample["observables"].shape[1]
        started = time.perf_counter()
        rows = {}; arrays = {}
        for name, iterations, degree in (("full_joint_original",2,0),("full_joint_strong",50,2)):
            state = full_joint_state(circuit,iterations,degree)
            sf_prediction = full_joint_predict(state,faults["detectors"])
            prediction = full_joint_predict(state,sample["detectors"])
            residual = prediction ^ sample["observables"]
            failures = int(residual.any(axis=1).sum())
            risk = float(faults["physical_coefficients"] @ (sf_prediction ^ faults["observables"]).any(axis=1))
            rows[name] = dict(R_delta=risk,failures=failures,shots=len(prediction),
                interval=clopper_pearson(failures,len(prediction)),
                fixed_vs_matched=paired_comparison(residual,matched[name+"_residual"]))
            arrays[name+"_prediction"] = prediction
            arrays[name+"_residual"] = residual
            arrays[name+"_single_fault_prediction"] = sf_prediction
        target = WORK/"results/full-model-fixed"/order
        save_npz(target/"decoded.npz",**arrays)
        record = dict(task=task,prior_epsilon=.001,physical_epsilon=.0001,
            decoder=rows,seconds=time.perf_counter()-started,
            sample_source=str(source/"physical.npz"),sample_sha256=digest(source/"physical.npz"),
            prior_sha256=digest(prior),decoded_sha256=digest(target/"decoded.npz"),
            interpretation="Same fixed prior, serial BP and OSD budgets as sliding rules; complete model replaces window recursion and its matrix representation")
        commit_json(target/"result.json",record)
        results.append(record)
        print(record,flush=True)
    commit_json(WORK/"results/full-model-fixed.json",dict(status="complete",results=results))


if __name__ == "__main__":
    run()
