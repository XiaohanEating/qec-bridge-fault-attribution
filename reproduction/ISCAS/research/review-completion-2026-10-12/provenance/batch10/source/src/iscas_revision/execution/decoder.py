"""Drop-in strong-window execution using validated exact arithmetic reuse."""
import contextlib
import io

from quits.decoder.base import spacetime

from ..decoding import CachedDecoder, CONFIGURATIONS
from .linear_osd import LinearOsd


class ExecutionDecoder(CachedDecoder):
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None, accelerate_original=False):
        super().__init__(code, prior_circuit, configurations, single_fault)
        for label, state in self.states.items():
            if label != "strong" and not accelerate_original:
                continue
            configuration = CONFIGURATIONS[label]
            window = configuration["window"]
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _, priors, _ = spacetime(prior_circuit, code.hz, window, window, state["num_cor_rounds"])
            state["decoders"] = [LinearOsd(check, prior, max_iter=configuration["max_iter"], order=configuration["osd_order"], exact_bp=True)
                for check, prior in zip(checks, priors)]


class HistoricalExecutionDecoder(ExecutionDecoder):
    def __init__(self, code, prior_circuit, configurations=("original", "strong"), single_fault=None):
        super().__init__(code, prior_circuit, configurations, single_fault, accelerate_original=True)
