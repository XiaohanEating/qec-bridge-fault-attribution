"""Circuit metrics for ISCAS: depth, 2Q-gate count, reset/measurement counts.

Depth = schedule depth (dependency layers over shared qubits), computed from
the flattened Stim circuit. NOTE: stim instructions carry multiple targets
(e.g. one R covers all data qubits), so all counts are per-target, not
per-instruction. Idle = per-qubit time slots with no operation between that
qubit's first and last operation (ISCAS §3).
"""
from __future__ import annotations

import stim

ONE_Q = {"X", "Y", "Z", "H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"}
TWO_Q = {"CX", "CNOT", "CZ", "CY", "XCX", "XCZ", "ZCX", "ZCZ", "ISWAP", "SWAP"}
MEASURE = {"M", "MX", "MZ", "MR", "MRX", "MRZ"}
RESET = {"R", "RX", "RZ"}


def _qubit_targets(inst: stim.CircuitInstruction) -> list[int]:
    return [t.qubit_value for t in inst.targets_copy() if t.qubit_value is not None]


def circuit_metrics(circuit: stim.Circuit) -> dict:
    flat = circuit.flattened()
    last_time: dict[int, int] = {}
    busy: dict[int, set[int]] = {}  # qubit -> set of time slots with an op
    depth = 0
    n1q = n2q = nmeas = nreset = 0
    for inst in flat:
        name = inst.name
        targets = _qubit_targets(inst)
        if not targets:
            continue
        if name in TWO_Q or name in ONE_Q or name in MEASURE or name in RESET:
            t = max(last_time.get(q, 0) for q in targets) + 1
            for q in targets:
                last_time[q] = t
                busy.setdefault(q, set()).add(t)
            depth = max(depth, t)
        if name in TWO_Q:
            n2q += len(targets) // 2
        elif name in ONE_Q:
            n1q += len(targets)
        elif name in MEASURE:
            nmeas += len(targets)
        elif name in RESET:
            nreset += len(targets)
    # Idle slots: per qubit, time steps in (first_op, last_op) with no op.
    num_idle = 0
    for q, slots in busy.items():
        span = max(slots) - min(slots) + 1
        num_idle += span - len(slots)
    return {
        "depth": depth,
        "num_1q": n1q,
        "num_2q": n2q,
        "num_measure": nmeas,
        "num_reset": nreset,
        "num_idle": num_idle,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
    }


def lexicographic_key(metrics: dict) -> tuple[int, int, int]:
    """Objective from ISCAS §3: (depth, #2Q, #idle)."""
    return (metrics["depth"], metrics["num_2q"], metrics["num_idle"])


if __name__ == "__main__":
    import numpy as np

    from quits import HgpCode

    h = np.array([[1, 1, 0], [0, 1, 1]])
    c = HgpCode(h, h).build_circuit(strategy="cardinal", num_rounds=3)
    print(circuit_metrics(c))
