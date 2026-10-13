"""Drop-in strong-window execution using validated exact arithmetic reuse."""
import contextlib
import io

from quits.decoder.base import spacetime

from ..decoding import CachedDecoder, CONFIGURATIONS
from .linear_osd import LinearOsd


class ExecutionDecoder(CachedDecoder):
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None, accelerate_original=False, prefix_reuse=False, zero_shortcut=False, sparse_support=False):
        super().__init__(code, prior_circuit, configurations, single_fault)
        for label, state in self.states.items():
            if label != "strong" and not accelerate_original:
                continue
            configuration = CONFIGURATIONS[label]
            window = configuration["window"]
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _, priors, _ = spacetime(prior_circuit, code.hz, window, window, state["num_cor_rounds"])
            state["decoders"] = [LinearOsd(check, prior, max_iter=configuration["max_iter"], order=configuration["osd_order"], exact_bp=True, prefix_reuse=prefix_reuse, zero_shortcut=zero_shortcut, sparse_support=sparse_support)
                for check, prior in zip(checks, priors)]


class HistoricalExecutionDecoder(ExecutionDecoder):
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None):
        super().__init__(code, prior_circuit, configurations, single_fault, accelerate_original=True)


class ValidatedExecutionDecoder(ExecutionDecoder):
    """Ordered prefix reuse and identical support scoring; settings unchanged."""
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None):
        import numpy as np
        from ..audit import optimal_table
        from support import predict
        from .state_store import find_window_inputs
        stored=find_window_inputs(code,prior_circuit)
        if stored is None:
            super().__init__(code, prior_circuit, configurations, single_fault,
                accelerate_original=True, prefix_reuse=True, zero_shortcut=True, sparse_support=True)
            return
        self.states={}
        for label in configurations:
            source=stored[label]
            config=CONFIGURATIONS[label]
            state={name:source[name]for name in ("num_cor_rounds","hz_rows","W","F","obs_set","update")}
            state["decoders"]=[LinearOsd(check,prior,max_iter=config["max_iter"],order=config["osd_order"],
                exact_bp=True,prefix_reuse=True,zero_shortcut=True,sparse_support=True)
                for check,prior in zip(source["checks"],source["priors"])]
            self.states[label]=state
        self.cache={name:{}for name in configurations}
        self.lookups=None
        if single_fault is not None:
            self.lookups,_=optimal_table(single_fault["detectors"],single_fault["observables"],single_fault["physical_coefficients"])
            self.lookups[bytes(prior_circuit.num_detectors)]=np.zeros(prior_circuit.num_observables,dtype=np.uint8)
        for name,state in self.states.items():
            zero=predict(state,np.zeros((1,prior_circuit.num_detectors),dtype=np.uint8))[0]
            assert not zero.any(),"Nonzero noiseless decision"
            self.cache[name][bytes(prior_circuit.num_detectors)]=zero.copy()
