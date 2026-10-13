"""Decoder-aware construction-space objective (ISCAS round 5, "P_L with the
decoder in the loop").

Why a new proxy
---------------
Round 2/3 established that the circuit-cost dimensions (depth / #2Q / #idle)
are structurally saturated on the C1 instances, and the only construction-space
freedom that moves anything is *noise propagation*: the logical storage basis
``basis in {Z, X}`` and the CNOT-layer order inside one round.  The objective
used so far was

    ``logical_weight`` = sum of the DEM error probabilities of all error paths
    that flip a logical observable,

i.e. the **decoder-blind** first-order estimate

    P_L^(1,blind) = sum_e p_e ,  over logical-flipping single faults e.

Round 4 falsified it: on ``BB/bb-108 custom`` the proxy prefers ``basis=X``
(-20% of the proxy) while the measured BP-OSD logical failure rate at
eps=1e-3 *worsens* from 0.00749 (Z) to 0.01326 (X).  A proxy that sums every
logical-flipping edge is blind to the fact that a decoder *corrects* most of
those faults: what matters is not "does this fault flip a logical observable"
but "does the decoder, after seeing the syndrome of this fault, still emit the
wrong logical value".

This module implements that correction, at two levels of cost.

1. ``mechanism_stats`` / ``logical_weight_radius`` (deterministic, no decoder)
   The cheap "decoder-boundary" proxy: an error pattern of weight
   ``w <= floor((d-1)/2)`` is *guaranteed* to be uniquely decodable, so it can
   never be a logical failure regardless of the decoder.  Summing only the
   mechanisms whose physical Pauli weight exceeds that radius removes the
   provably-correctable part of ``logical_weight``.  Physical weights come from
   ``stim.Circuit.explain_detector_error_model_errors`` (fast: ~0.1 s for the
   2268 mechanisms of bb-108), so this proxy can drive an exhaustive
   construction-space search.

2. ``calibrate`` (decoder-aware, bounded Monte-Carlo over mechanisms)
   The exact first-order estimate with the decoder in the loop:

       P_L^(1,cal) = sum_e p_e * f_e ,

   where ``f_e`` is 1 iff BP-OSD, given *only* the single fault ``e``, emits a
   logical correction different from the true logical outcome.  The syndrome
   and the true logical value of a lone fault are exactly its DEM
   ``error(...) D.. L..`` line, so ``f_e`` is measured by feeding that one
   syndrome through the very same BP-OSD sliding-window decode path that
   ``ISCAS/experiments/pl_validation.py`` uses for the end-to-end shots
   (``quits.decoder.base.spacetime`` matrices + ``ldpc.BpOsdDecoder``,
   identical decoder kwargs).  A lone-fault decode is deterministic, so ``f_e``
   is a 0/1 value; the *noise* in the objective comes from which mechanisms we
   chose to spend the budget on, and it is reported as a Wilson confidence
   interval.

   ``budget`` limits how many mechanisms are calibrated (drawn with
   probability proportional to ``p_e`` -- importance sampling, which puts the
   shots where the logical failure mass is).  The search then has to make a
   decision under a noisy objective, which is what
   ``optimizer.select_under_noise`` does: a candidate only beats the baseline
   if the difference is statistically significant, otherwise the baseline
   (always part of the candidate set) is kept.

Scope / honesty
---------------
``f_e`` is a *lone-fault* quantity.  It is a lower-biased estimate of
``Pr[decoder fails | fault e occurs]`` (in a real shot other faults co-occur,
which can only make the decoder's job harder), so ``P_L^(1,cal)`` is a
lower bound on the first-order logical failure rate and it ignores all
higher-order terms.  It is used here strictly as a *construction-space
ranking objective* and it is validated, before any of it is used as a search
objective, against the measured end-to-end BP-OSD ``P_L`` of the five
``pl_validation.jsonl`` baselines (``experiments/proxy_direction_validation.py``).

Nothing under ``third_party/`` or the installed ``quits``/``ldpc`` packages is
modified.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_metrics import circuit_metrics  # noqa: E402

# noise channels stim exposes as single-error mechanisms; the probe/noise-free
# rebuild of a circuit only strips these
NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
             "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "CORRELATED_ERROR"}


# --------------------------------------------------------------------------
# circuits
# --------------------------------------------------------------------------
def sd6(eps: float):
    """SD6 = PLAN.md §5 uniform circuit-level depolarizing noise model."""
    from quits import ErrorModel

    return ErrorModel(idle_error=eps, sqgate_error=eps, tqgate_error=eps,
                      spam_error=eps)


def build_circuit(code, strategy: str, rounds: int, basis: str, eps: float):
    """Noisy circuit exactly as ``pl_validation.py`` builds it (SD6, same eps)."""
    return code.build_circuit(strategy=strategy, num_rounds=rounds, basis=basis,
                              error_model=sd6(eps))


# --------------------------------------------------------------------------
# error mechanisms
# --------------------------------------------------------------------------
def _channel_prob(gate: str, args) -> float:
    """Probability of one single-error mechanism produced by a noise channel.

    stim's ``DEPOLARIZE1(p)`` is ``(1-p)I + (p/3)(X+Y+Z)`` and
    ``DEPOLARIZE2(p)`` is ``(1-p)II + (p/15) sum_{P != II} P``, so every
    non-identity single-error mechanism of the channel carries ``p/3`` and
    ``p/15`` respectively; bit-flip channels carry ``p``.
    """
    if not args:
        return 0.0
    p = float(args[0])
    if gate == "DEPOLARIZE1":
        return p / 3.0
    if gate == "DEPOLARIZE2":
        return p / 15.0
    if gate == "PAULI_CHANNEL_1":          # X_ERROR(p) == PAULI_CHANNEL_1(p,0,0)
        return p
    if gate == "PAULI_CHANNEL_2":          # pragma: no cover - not emitted by QUITS
        return p / 15.0
    return p


def _det_id(t) -> int:
    """Relative detector id of a DEM target.

    stim <= 1.15 exposes ``t.relative_detector_id()`` as a getter; stim 1.16
    turned ``DemTarget.relative_detector_id`` / ``logical_observable_id`` into
    *constructors* and moved the getter to ``DemTarget.val``.  Supporting both
    keeps this module runnable on either installed stim.
    """
    if hasattr(t, "val"):
        return int(t.val)
    return int(t.relative_detector_id())  # pragma: no cover - stim <= 1.15


def _logical_id(t) -> int:
    if hasattr(t, "val"):
        return int(t.val)
    return int(t.logical_observable_id())  # pragma: no cover - stim <= 1.15


def mechanism_list(circuit: stim.Circuit) -> list[dict]:
    """Enumerate the circuit's single-fault error mechanisms.

    Each entry carries the mechanism's probability, the detector/logical
    signature it triggers (which *is* the noise-free sampling outcome of a lone
    occurrence of the fault), the physical Pauli weight, and the flattened
    instruction index of the noise channel that produces it.
    """
    explained = circuit.explain_detector_error_model_errors()
    mechs: list[dict] = []
    unexplained = 0
    for k, err in enumerate(explained):
        terms = err.dem_error_terms
        dets = tuple(sorted(_det_id(t.dem_target) for t in terms
                            if t.dem_target.is_relative_detector_id()))
        logicals = tuple(sorted(_logical_id(t.dem_target) for t in terms
                                if t.dem_target.is_logical_observable_id()))
        for loc in err.circuit_error_locations:
            info = loc.instruction_targets
            paulis = [(t.gate_target.qubit_value, t.gate_target.pauli_type)
                      for t in loc.flipped_pauli_product]
            mechs.append({
                "id": len(mechs),
                "prob": _channel_prob(info.gate, info.args),
                "detectors": dets,
                "logicals": logicals,
                "weight": len(paulis),
                "paulis": tuple(paulis),
                "position": loc.stack_frames[0].instruction_offset,
                "gate": info.gate,
            })
        if not err.circuit_error_locations:
            unexplained += 1
    for m in mechs:
        m["unexplained_dem_lines"] = unexplained
    return mechs


def syndrome_logical_census(mechs: list[dict]) -> dict:
    """Does a mechanism's detector signature determine its logical outcome?

    This is the exact, decoder-free statement of *why* ``calibrate`` can be
    degenerate.  If every mechanism that shares a detector signature also
    shares its logical flip, then a decoder handed that signature cannot be
    wrong about the logical value: any correction it emits is consistent with
    the observed syndrome, and every error in that coset has the same logical
    content.  In that case ``f_e = 0`` for every single fault and
    ``P_L^(1,cal)`` is **identically zero**, whichever construction
    (basis, layer order, ...) is being ranked.

    ``num_ambiguous_syndromes`` counts the signatures for which the logical
    outcome is *not* determined; those are the only syndromes on which a lone
    fault can actually turn into a logical failure.
    """
    by_syn: dict[tuple, set] = {}
    for m in mechs:
        by_syn.setdefault(m["detectors"], set()).add(m["logicals"])
    ambiguous = [syn for syn, lg in by_syn.items() if len(lg) > 1]
    sizes: dict[int, int] = {}
    for syn in by_syn:
        sizes[len(syn)] = sizes.get(len(syn), 0) + 1
    return {
        "num_mechanisms": len(mechs),
        "num_distinct_syndromes": len(by_syn),
        "num_ambiguous_syndromes": len(ambiguous),
        "ambiguous_fraction": len(ambiguous) / len(by_syn) if by_syn else 0.0,
        "first_order_objective_degenerate": len(ambiguous) == 0,
        "syndrome_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
    }


def mechanism_stats(mechs: list[dict], radii=(1, 2, 3)) -> dict:
    """Decoder-boundary statistics of a mechanism list.

    ``logical_weight`` is the *decoder-blind* first-order logical failure rate
    (the quantity round 4 falsified).  ``logical_weight_radius_t`` is the same
    sum restricted to mechanisms whose physical Pauli weight exceeds ``t``:
    patterns of weight ``<= t`` are uniquely decodable, hence provably not
    logical failures.  ``min_logical_weight`` is the smallest mechanism weight
    that flips a logical observable (an upper bound on the code distance).
    """
    stats = {
        "num_mechanisms": len(mechs),
        "mechanism_prob_sum": sum(m["prob"] for m in mechs),
        "num_logical_mechanisms": 0,
        "logical_weight": 0.0,
        "logical_weight_radius": {},
        "weight_histogram": {},
        "logical_weight_histogram": {},
        "min_logical_weight": None,
        "num_logical_syndromes": 0,
        "num_ambiguous_logical_syndromes": 0,
        "max_detectors": max((len(m["detectors"]) for m in mechs), default=0),
    }
    logical: list[dict] = []
    weight_hist: dict[int, float] = {}
    logical_hist: dict[int, float] = {}
    for m in mechs:
        if not m["logicals"]:
            continue
        stats["num_logical_mechanisms"] += 1
        stats["logical_weight"] += m["prob"]
        logical.append(m)
        weight_hist[m["weight"]] = weight_hist.get(m["weight"], 0.0) + m["prob"]
        logical_hist[m["weight"]] = logical_hist.get(m["weight"], 0.0) + m["prob"]
        for t in radii:
            if m["weight"] > t:
                cur = stats["logical_weight_radius"].get(f"w>{t}", 0.0)
                stats["logical_weight_radius"][f"w>{t}"] = cur + m["prob"]
    stats["weight_histogram"] = {str(k): round(v, 8)
                                 for k, v in sorted(weight_hist.items())}
    stats["logical_weight_histogram"] = {str(k): round(v, 8)
                                         for k, v in sorted(logical_hist.items())}
    stats["min_logical_weight"] = min((m["weight"] for m in logical), default=None)
    # syndromes that several mechanisms share: a lone fault with a unique
    # syndrome is always correctable, an ambiguous one may not be
    by_syn: dict[tuple, set] = {}
    for m in logical:
        by_syn.setdefault(m["detectors"], set()).add(bool(m["logicals"]))
    stats["num_logical_syndromes"] = len(by_syn)
    stats["num_ambiguous_logical_syndromes"] = sum(
        1 for syn in by_syn if sum(1 for m in logical
                                   if m["detectors"] == syn) > 1)
    return stats


# --------------------------------------------------------------------------
# decoder calibration
# --------------------------------------------------------------------------
def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _decoder_setup(code, circuit: stim.Circuit, basis: str, window: int,
                   osd_order: int):
    """BP-OSD sliding-window decoder state, built exactly as in
    ``pl_validation.run_memory`` (same ``spacetime`` matrices, same decoder
    kwargs) from the *noisy* circuit, so the channel priors the decoder sees
    are the real SD6 priors."""
    from quits.decoder.base import spacetime

    from ldpc.bposd_decoder import BpOsdDecoder

    # decoder configuration, locked to baseline_replay.py / pl_validation.py
    decoder_kwargs = dict(bp_method="product_sum", max_iter=2, schedule="serial",
                          osd_method="osd_cs", osd_order=osd_order)
    h, l = (code.hz, code.lz) if basis == "Z" else (code.hx, code.lx)
    num_rounds = circuit.num_detectors // h.shape[0] - 2
    num_cor_rounds = (2 + num_rounds - window) // window
    if (2 + num_rounds - window) % window != 0:
        num_cor_rounds += 1
    check_set, obs_set, priors_set, update = spacetime(circuit, h, window,
                                                       window, num_cor_rounds)
    decoders = []
    for i in range(len(check_set)):
        kw = dict(decoder_kwargs)
        kw["channel_probs"] = priors_set[i]
        decoders.append(BpOsdDecoder(check_set[i], **kw))
    return {"decoders": decoders, "obs_set": obs_set, "update": update,
            "num_cor_rounds": num_cor_rounds, "hz_rows": h.shape[0],
            "num_observables": circuit.num_observables,
            "num_detectors": circuit.num_detectors,
            "W": window, "F": window}


def decode_lone_faults(state: dict, samples: np.ndarray,
                       obs: np.ndarray) -> np.ndarray:
    """Per-shot "decoder got the logical value wrong" flags.

    Byte-for-byte the per-shot recursion of
    ``quits.decoder.sliding_window.sliding_window_circuit_mem`` (and of
    ``pl_validation._decode_chunk``), only returning a vector instead of a
    count.  Per-shot decoder state is reset, so shots are independent.
    """
    decoders = state["decoders"]
    obs_set = state["obs_set"]
    update = state["update"]
    num_cor_rounds = state["num_cor_rounds"]
    hz_rows = state["hz_rows"]
    W = state.get("W", 1)
    F = state.get("F", 1)
    fails = np.zeros(samples.shape[0], dtype=bool)
    for i in range(samples.shape[0]):
        sample = samples[i]
        accumulated = np.zeros(obs_set[0].shape[0], dtype=int)
        syn_update = np.zeros(hz_rows, dtype=int)
        for k in range(num_cor_rounds):
            diff = sample[F * k * hz_rows:(F * k + W) * hz_rows].copy() % 2
            diff[:hz_rows] = (diff[:hz_rows] + syn_update) % 2
            decoded = decoders[k].decode(diff)
            m = obs_set[k].shape[1]
            accumulated = (accumulated + obs_set[k] @ decoded[:m] % 2) % 2
            syn_update = update[k] @ decoded[:m] % 2
        diff = sample[F * num_cor_rounds * hz_rows:].copy() % 2
        diff[:hz_rows] = (diff[:hz_rows] + syn_update) % 2
        decoded = decoders[num_cor_rounds].decode(diff)
        accumulated = (accumulated + obs_set[num_cor_rounds] @ decoded % 2) % 2
        fails[i] = bool(np.any(np.asarray(accumulated).ravel() != obs[i]))
    return fails


def _sample_from_mechanism(mech: dict, num_detectors: int,
                           num_observables: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free sampling outcome of a lone occurrence of ``mech``.

    Exactly one mechanism is present, so the sampled detection events are the
    mechanism's detector set and the sampled observable flips are its logical
    set -- the ``error(p) D.. L..`` line of the DEM.
    """
    s = np.zeros(num_detectors, dtype=np.uint8)
    for d in mech["detectors"]:
        s[d] = 1
    o = np.zeros(num_observables, dtype=np.uint8)
    for lgd in mech["logicals"]:
        o[lgd] = 1
    return s, o


def calibrate(code, circuit: stim.Circuit, basis: str, mechs: list[dict],
              budget: int | None = None, seed: int = 2026, window: int = 1,
              osd_order: int = 0) -> dict:
    """Estimate ``P_L^(1,cal) = sum_e p_e f_e`` with a bounded mechanism budget.

    ``budget`` mechanisms are drawn with probability proportional to ``p_e``
    (importance sampling), each is decoded on its own lone-fault syndrome, and
    the estimate is ``prob_sum * failure_rate`` with a Wilson CI.  ``budget=None``
    (or >= the number of mechanisms) calibrates everything and the CI
    degenerates to zero width.
    """
    state = _decoder_setup(code, circuit, basis, window, osd_order)
    prob_sum = sum(m["prob"] for m in mechs)
    if not mechs:
        return {"calibrated_weight": 0.0, "ci95": [0.0, 0.0], "calibrated": 0,
                "failures": 0, "failure_rate": 0.0, "prob_sum": prob_sum,
                "mechanism_weight_calibrated": prob_sum, "dem_detectors":
                state["num_detectors"], "wall_clock_s": 0.0}
    rng = random.Random(seed)
    if budget is None or budget >= len(mechs):
        chosen = list(mechs)
    else:
        weights = np.array([m["prob"] for m in mechs], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(mechs))
        probs = weights / weights.sum()
        idx = rng.choices(range(len(mechs)), weights=probs, k=budget)
        chosen = [mechs[i] for i in idx]
    samples = np.zeros((len(chosen), state["num_detectors"]), dtype=np.uint8)
    obs = np.zeros((len(chosen), state["num_observables"]), dtype=np.uint8)
    for i, m in enumerate(chosen):
        s, o = _sample_from_mechanism(m, state["num_detectors"],
                                      state["num_observables"])
        samples[i] = s
        obs[i] = o
    t0 = time.time()
    fails = decode_lone_faults(state, samples, obs)
    dt = time.time() - t0
    n = len(chosen)
    k = int(fails.sum())
    rate = k / n
    lo, hi = _wilson(k, n)
    # weight of the calibrated subset, for honest reporting of the coverage
    sub_weight = sum(m["prob"] for m in chosen)
    return {
        "calibrated_weight": prob_sum * rate,
        "ci95": [prob_sum * lo, prob_sum * hi],
        "calibrated": n,
        "failures": k,
        "failure_rate": rate,
        "prob_sum": prob_sum,
        "mechanism_weight_calibrated": sub_weight,
        "coverage": sub_weight / prob_sum if prob_sum else 1.0,
        "wall_clock_s": round(dt, 2),
    }


# --------------------------------------------------------------------------
# one-shot analysis of a construction candidate
# --------------------------------------------------------------------------
def analyse(code, strategy: str, rounds: int, basis: str, eps: float,
            radii=(1, 2, 3), budget: int | None = None, seed: int = 2026,
            window: int = 1, osd_order: int = 0,
            circuit: stim.Circuit | None = None) -> dict:
    """Full decoder-aware analysis of one construction candidate.

    ``circuit`` may be passed by a caller that already built the candidate
    (e.g. a permuted CNOT-layer order); everything is rebuilt from the code
    otherwise.
    """
    if circuit is None:
        circuit = build_circuit(code, strategy, rounds, basis, eps)
    mechs = mechanism_list(circuit)
    out = {
        "basis": basis,
        "eps": eps,
        "metrics": circuit_metrics(circuit),
        "mechanism_stats": mechanism_stats(mechs, radii),
        "mechanisms": mechs,
    }
    if budget != 0:
        out["calibration"] = calibrate(code, circuit, basis, mechs,
                                       budget=budget, seed=seed, window=window,
                                       osd_order=osd_order)
    return out
