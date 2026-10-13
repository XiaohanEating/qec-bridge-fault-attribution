"""Original recursion, controlled configurations, and optimal single-fault lookup."""
from collections import defaultdict
import contextlib
import io
import sys
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from quits.decoder.base import spacetime
from .artifacts import ROOT
from .audit import optimal_table

HIST_SOURCE = ROOT / "ISCAS/research/mechanism-intervention-2026-10-11/src"
if str(HIST_SOURCE) not in sys.path:
    sys.path.insert(0, str(HIST_SOURCE))
from support import predict

CONFIGURATIONS = {
    "original": dict(window=1, max_iter=2, osd_order=0),
    "window2": dict(window=2, max_iter=2, osd_order=0),
    "window4": dict(window=4, max_iter=2, osd_order=0),
    "iterations10": dict(window=1, max_iter=10, osd_order=0),
    "iterations50": dict(window=1, max_iter=50, osd_order=0),
    "osd1": dict(window=1, max_iter=2, osd_order=1),
    "osd2": dict(window=1, max_iter=2, osd_order=2),
    "strong": dict(window=4, max_iter=50, osd_order=2),
}


def make_state(code, circuit, configuration):
    config = CONFIGURATIONS[configuration]
    window = config["window"]
    h = code.hz
    assert circuit.num_detectors % h.shape[0] == 0
    rounds = circuit.num_detectors // h.shape[0] - 2
    count = (2 + rounds - 1) // window
    with contextlib.redirect_stdout(io.StringIO()):
        checks, observations, priors, updates = spacetime(circuit, h, window, window, count)
    decoders = [BpOsdDecoder(check, channel_probs=prior, bp_method="product_sum", schedule="serial",
                    max_iter=config["max_iter"], osd_method="osd_cs", osd_order=config["osd_order"])
                for check, prior in zip(checks, priors)]
    return dict(decoders=decoders, obs_set=observations, update=updates,
                num_cor_rounds=count, hz_rows=h.shape[0], W=window, F=window)


class CachedDecoder:
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None):
        self.states = {name:make_state(code, prior_circuit, name) for name in configurations}
        self.cache = {name:{} for name in configurations}
        self.lookups = None
        if single_fault is not None:
            self.lookups, _ = optimal_table(single_fault["detectors"], single_fault["observables"],
                                           single_fault["physical_coefficients"])
            self.lookups[bytes(prior_circuit.num_detectors)] = np.zeros(prior_circuit.num_observables, dtype=np.uint8)
        for name, state in self.states.items():
            zero = predict(state, np.zeros((1, prior_circuit.num_detectors), dtype=np.uint8))[0]
            assert not zero.any(), "Nonzero noiseless decision"
            self.cache[name][bytes(prior_circuit.num_detectors)] = zero.copy()

    def predictions(self, detectors, include_lookup=True):
        detectors = np.asarray(detectors, dtype=np.uint8)
        unique, inverse = np.unique(detectors, axis=0, return_inverse=True)
        keys = [bytes(record) for record in unique]
        output = {}
        for name, state in self.states.items():
            cache = self.cache[name]
            missing = [i for i,key in enumerate(keys) if key not in cache]
            if missing:
                guesses = predict(state, unique[missing])
                cache.update({keys[i]:guess.copy() for i,guess in zip(missing,guesses)})
            output[name] = np.array([cache[key] for key in keys])[inverse]
        if include_lookup and self.lookups is not None:
            assert "strong" in output
            lookup = output["strong"].copy()
            for i,key in enumerate(keys):
                if key in self.lookups:
                    lookup[inverse == i] = self.lookups[key]
            output["single_fault_lookup"] = lookup
        return output


def full_joint_state(circuit, iterations=50, order=2):
    """Preserve distinct (D,L) columns, including equal D with different L."""
    from scipy.sparse import csc_matrix
    dem = circuit.detector_error_model().flattened()
    ds, ls, priors = [], [], []
    for item in dem:
        if item.type != "error":
            continue
        d = np.zeros(circuit.num_detectors, dtype=np.uint8)
        l = np.zeros(circuit.num_observables, dtype=np.uint8)
        for target in item.targets_copy():
            if target.is_relative_detector_id(): d[target.val] ^= 1
            elif target.is_logical_observable_id(): l[target.val] ^= 1
        ds.append(d); ls.append(l); priors.append(item.args_copy()[0])
    h = csc_matrix(np.asarray(ds, dtype=np.uint8).T)
    logical = csc_matrix(np.asarray(ls, dtype=np.uint8).T)
    decoder = BpOsdDecoder(h, channel_probs=priors, max_iter=iterations,
                         bp_method="product_sum", schedule="serial", osd_method="osd_cs", osd_order=order)
    return decoder, logical


def full_joint_predict(state, detectors):
    decoder, logical = state
    unique, inverse = np.unique(detectors.astype(np.uint8), axis=0, return_inverse=True)
    guesses = np.asarray([np.asarray(logical @ decoder.decode(record) % 2).ravel() for record in unique], dtype=np.uint8)
    return guesses[inverse]
