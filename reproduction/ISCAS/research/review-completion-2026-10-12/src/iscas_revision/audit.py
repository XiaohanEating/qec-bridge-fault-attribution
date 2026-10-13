"""Read-only reanalysis of manuscript inputs; no decoder package required."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import stim
from scipy.stats import binom, binomtest
from .artifacts import ROOT, WORK, HISTORICAL, commit_json, digest, read_json


def optimal_table(detectors, observables, masses):
    groups = defaultdict(lambda: defaultdict(float))
    for d, logical, mass in zip(detectors, observables, masses):
        groups[bytes(d)][bytes(logical)] += float(mass)
    zero = bytes(observables.shape[1])
    table, risk = {}, 0.0
    for record, labels in groups.items():
        selected = zero if not any(record) else min(labels, key=lambda label: (-labels[label], label))
        table[record] = np.frombuffer(selected, dtype=np.uint8).copy()
        risk += sum(labels.values()) - labels.get(selected, 0.0)
    return table, risk


def paired_comparison(first, second):
    a, b = first.any(axis=1), second.any(axis=1)
    first_only, second_only = int((a & ~b).sum()), int((b & ~a).sum())
    discordant = first_only + second_only
    return dict(shots=len(a), first_failures=int(a.sum()), second_failures=int(b.sum()),
                first_only=first_only, second_only=second_only,
                paired_difference=(second_only-first_only)/len(a),
                mcnemar_exact_two_sided_p=binomtest(first_only, discordant, .5).pvalue if discordant else 1.0)


def inspect_layout(instance):
    source = HISTORICAL / "results" / instance / "original.stim"
    circuit = stim.Circuit(source.read_text(encoding="utf-8"))
    plan = read_json(source.parent / "forward/plan.json")
    mapping = {int(q): int(site) for q, site in plan["initial_mapping"].items()}
    side = plan["side"]
    interactions = Counter()
    reset_sites, measured_sites = set(), set()
    for inst in circuit.flattened():
        targets = [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name == "CX":
            interactions.update(zip(targets[::2], targets[1::2]))
        elif inst.name in {"R", "RX"}:
            reset_sites.update(targets)
        elif inst.name in {"M", "MX"}:
            measured_sites.update(targets)
    # Initial and terminal data reset/readout also occur; checks are identified
    # by repeated measurement count rather than treating every measured qubit as an ancilla.
    measurements = Counter(t.qubit_value for inst in circuit.flattened() if inst.name in {"M","MX","MR","MRX"}
                           for t in inst.targets_copy() if t.is_qubit_target)
    checks = {q for q, count in measurements.items() if count > 1}
    gates = []
    for (control, target), count in sorted(interactions.items()):
        a, b = divmod(mapping[control], side), divmod(mapping[target], side)
        distance = abs(a[0]-b[0])+abs(a[1]-b[1])
        gates.append(dict(control=control, target=target, occurrences=count,
                          control_rc=list(a), target_rc=list(b), manhattan_distance=distance,
                          requires_bridge_on_saved_layout=distance > 1))
    return dict(source_sha256=digest(source), side=side,
                mapping={str(q):list(divmod(site,side)) for q,site in mapping.items()},
                measured_check_qubits=sorted(checks), measurement_counts=dict(measurements),
                source_cx_count=sum(interactions.values()), unique_interactions=gates,
                remote_cx_occurrences=sum(g["occurrences"] for g in gates if g["manhattan_distance"] > 1),
                remote_checks=sorted({q for g in gates if g["manhattan_distance"] > 1
                                      for q in (g["control"],g["target"]) if q in checks}),
                rounds_source="saved source has initial extraction plus three repeats; all four blocks noisy, terminal data readout noiseless")


def run():
    sources = [ROOT / "ISCAS/ISCAS review.md", ROOT / "ISCAS/paper/manuscript/main.tex"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                  instances={}, paired_confirmation={}, layouts={})
    for instance in ("tiny-hgp", "holdout-hgp-3x4"):
        result["layouts"][instance] = inspect_layout(instance)
        result["instances"][instance] = {}
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = HISTORICAL / "results" / instance / order
            paths = [folder / "single-fault.npz", folder / "risk.json", folder / "branches.json", folder / "plan.json"]
            sources.extend(paths)
            data = np.load(paths[0])
            saved = read_json(paths[1])
            weights = data["physical_coefficients"]
            table, optimum = optimal_table(data["detectors"], data["observables"], weights)
            pred = np.array([table[bytes(d)] for d in data["detectors"]])
            actual_lookup_risk = float(weights @ (pred ^ data["observables"]).any(axis=1))
            assert abs(actual_lookup_risk-optimum) < 1e-7
            assert abs(optimum-saved["optimal_single_fault_coefficient"]) < 1e-7
            n = saved["total_physical_coefficient"]
            assert n == int(n)
            risks = {label:float(weights @ (data[label]^data["observables"]).any(axis=1))
                     for label in ("original","strong")}
            assert all(abs(risks[label]-saved["decoder"][label]["coefficient"]) < 1e-7 for label in risks)
            result["instances"][instance][order] = dict(noise_locations=int(n), optimal_risk=optimum,
                actual_risks=risks, lookup_risk=actual_lookup_risk,
                fault_probabilities={str(eps):dict(expected_faults=n*eps,
                    no_fault=float(binom.pmf(0,int(n),eps)), one_fault=float(binom.pmf(1,int(n),eps)),
                    multi_fault=float(binom.sf(1,int(n),eps)),
                    linear_prediction={label:eps*risk for label,risk in risks.items()})
                    for eps in (1e-6,3e-6,1e-5,3e-5,1e-4)})
    for order in ("forward", "macro_reverse"):
        path = ROOT / "ISCAS/paper/review-debate/confirmation/results" / order / "decoded.npz"
        sources.append(path)
        data = np.load(path)
        result["paired_confirmation"][order] = paired_comparison(data["original_residual"], data["strong_residual"])
    before = {str(path.relative_to(ROOT)):digest(path) for path in sources}
    result["input_sha256"] = before
    commit_json(WORK / "results/historical-audit.json", result)
    assert before == {str(path.relative_to(ROOT)):digest(path) for path in sources}
    commit_json(WORK / "results/status.json", dict(status="running", environment="awaiting user selection",
        completed=["historical single-fault and optimal lookup audit", "paired McNemar", "exact layouts", "fault-count probabilities"],
        pending=["nonlocal BB", "low-noise stratification", "decoder one-factor scans", "dynamic routing", "Hirai", "Zhao", "idle sensitivity", "paper revision"]))
    print({"audit":str(WORK / "results/historical-audit.json"), "historical_inputs_unchanged":True,
           "paired":result["paired_confirmation"]}, flush=True)


if __name__ == "__main__":
    run()
