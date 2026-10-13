"""Physical distance-two injection certificates and exact budget modifications."""
from itertools import product
import numpy as np
import stim
from .artifacts import WORK, HISTORICAL, read_json, commit_json, digest
from .stratification import NOISE


def inject(circuit, branches):
    by_offset = {}
    for branch in branches:
        by_offset.setdefault(branch["instruction_offset"], []).extend(branch["paulis"])
    output = stim.Circuit()
    for offset, inst in enumerate(circuit.flattened()):
        for site, pauli in by_offset.get(offset, []):
            output.append(pauli+"_ERROR", [site], 1.)
        if inst.name not in NOISE:
            output.append(inst)
    return output


def distance_two():
    saved = read_json(WORK/"results/circuit-distance-audit.json")
    results = []
    target = WORK/"provenance/distance-two"
    target.mkdir(parents=True, exist_ok=True)
    for item in saved:
        if "pair_signature_indices" not in item: continue
        instance, order = item["instance"], item["order"]
        folder = HISTORICAL/"results"/instance/order
        source = HISTORICAL/"provenance/decoder-priors"/instance/(order+".stim")
        hashes = {str(path): digest(path) for path in (source, folder/"branches.json", folder/"single-fault.npz")}
        circuit = stim.Circuit(source.read_text(encoding="utf-8"))
        data = np.load(folder/"single-fault.npz")
        branches = read_json(folder/"branches.json")["branches"]
        first, second = item["pair_signature_indices"]
        pair = next((a, b) for a, b in product(
            [b for b in branches if b["signature"] == first],
            [b for b in branches if b["signature"] == second])
            if a["instruction_offset"] != b["instruction_offset"])
        expected_d = data["detectors"][first]
        expected_difference = data["observables"][first] ^ data["observables"][second]
        assert expected_d.any() and expected_difference.any()
        probes = []
        for label, chosen, signature in (("first", [pair[0]], first), ("second", [pair[1]], second),
                                          ("pair", pair, None)):
            injected = inject(circuit, chosen)
            d, logical = injected.compile_detector_sampler(seed=102612).sample(16, separate_observables=True)
            if signature is None:
                assert not d.any() and np.all(logical == expected_difference)
            else:
                assert np.all(d == data["detectors"][signature])
                assert np.all(logical == data["observables"][signature])
            path = target/f"{instance}-{order}-{label}.stim"
            path.write_text(str(injected), encoding="utf-8")
            probes.append(dict(label=label, deterministic_repetitions=16,
                detectors=np.flatnonzero(d[0]).tolist(), logical_vector=logical[0].astype(int).tolist(),
                source_sha256=digest(path)))
        assert all(digest(path) == sha for path, sha in hashes.items())
        results.append(dict(instance=instance, order=order, exact_fault_distance=2,
            full_single_fault_enumeration_no_distance_one=True, physical_branches=pair,
            signature_indices=[first, second], distinct_physical_noise_instructions=True,
            probes=probes, inputs_sha256=hashes,
            repetition_purpose="check deterministic simulator response; no failure-rate estimate"))
    commit_json(WORK/"results/distance-two-certificates.json", results)
    return results


def budget_change():
    folder = HISTORICAL/"results/budget-local"
    paths = {name: folder/name/"plan.json" for name in ("baseline", "region-1")}
    plans = {name: read_json(path) for name, path in paths.items()}
    a, b = plans["baseline"], plans["region-1"]
    assert len(a["slots"]) == len(b["slots"])
    changed = [i for i, (left, right) in enumerate(zip(a["slots"], b["slots"])) if left != right]
    assert changed == [38, 39]
    assert b["slots"][38:40] == a["slots"][38:40][::-1]
    other = {key: value for key, value in a.items() if key != "slots"}
    assert other == {key: value for key, value in b.items() if key != "slots"}
    from .decoding import CachedDecoder  # establishes the locked historical import path
    from construction import inherited, primitive_multiset
    def ideal_blocks(circuit):
        blocks, unitary = [], stim.Circuit()
        unitary.append("I", range(circuit.num_qubits))
        for inst in circuit:
            if inst.name in {"R", "RX", "M", "MX", "MR", "MRX"}:
                blocks.append((stim.Tableau.from_circuit(unitary), str(inst)))
                unitary = stim.Circuit()
                unitary.append("I", range(circuit.num_qubits))
            elif inst.name in {"H", "CX", "I"}:
                unitary.append(inst)
        blocks.append((stim.Tableau.from_circuit(unitary), "terminal"))
        return blocks
    matrices = []
    for name, plan in plans.items():
        ideal, _ = inherited.materialize(plan, 0.)
        matrices.append(ideal_blocks(ideal))
    assert matrices[0] == matrices[1]
    assert primitive_multiset(a) == primitive_multiset(b)
    records = {name: inherited.materialize(plan, .001)[1] for name, plan in plans.items()}
    assert records["baseline"] == records["region-1"]
    result = dict(index_convention="zero-based half-open slot interval [38,40)",
        operation="exchange the two complete saved slots; no other plan entry changes",
        before=a["slots"][38:40], after=b["slots"][38:40],
        unchanged_signed_block_tableaus=True, same_directed_physical_cx_multiset=True,
        metrics=records, cap=1.05*646, source_sha256={name: digest(path) for name, path in paths.items()})
    commit_json(WORK/"results/budget-modification-audit.json", result)
    return result


if __name__ == "__main__":
    print(dict(distance_certificates=len(distance_two()), budget=budget_change()), flush=True)
