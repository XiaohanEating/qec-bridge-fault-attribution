"""Probability-weighted physical Pauli branches, retaining source coordinates."""
from collections import defaultdict, Counter
import numpy as np
from support import stim, signature, decoder_state, predict, save, sha

NOISE = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR'}

def physical_branches(circuit, eps):
    """Stim explains distinct circuit locations before DEM parity merging.

    Branches with empty D/L response are omitted by Stim. Their count is the
    emitted physical channel branch total minus explained distinct locations;
    they contribute no failure for a decoder whose zero output is verified.
    """
    flat = circuit.flattened()
    total_branches, total_weight = 0, 0.0
    for inst in flat:
        count = len(inst.targets_copy())
        if inst.name == 'DEPOLARIZE1':
            total_branches += 3*count; total_weight += count*inst.gate_args_copy()[0]/eps
        elif inst.name == 'DEPOLARIZE2':
            total_branches += 15*count//2; total_weight += count//2*inst.gate_args_copy()[0]/eps
        elif inst.name in {'X_ERROR','Y_ERROR','Z_ERROR'}:
            total_branches += count; total_weight += count*inst.gate_args_copy()[0]/eps
    signatures, branches, seen = [], [], set()
    for explained in circuit.explain_detector_error_model_errors():
        ds, ls = signature([t.dem_target for t in explained.dem_error_terms])
        index = len(signatures)
        signatures.append((ds, ls))
        for loc in explained.circuit_error_locations:
            offset = loc.stack_frames[0].instruction_offset
            paulis = tuple((t.gate_target.qubit_value, t.gate_target.pauli_type) for t in loc.flipped_pauli_product)
            target_start = loc.instruction_targets.target_range_start
            identity = (offset, target_start, paulis)
            assert identity not in seen, identity
            seen.add(identity)
            inst = flat[offset]
            divisor = 3 if inst.name == 'DEPOLARIZE1' else 15 if inst.name == 'DEPOLARIZE2' else 1
            weight = inst.gate_args_copy()[0]/divisor/eps
            branches.append(dict(instruction_offset=offset, target_start=target_start, source=inst.name,
                                 paulis=paulis, coefficient=weight, signature=index))
    # There can be a signature empty in all labels only if explicit filter is used;
    # default Stim returns all nonempty D/L explanations.
    assert len(branches) <= total_branches
    assert sum(b['coefficient'] for b in branches) <= total_weight + 1e-8
    syn = np.zeros((len(signatures), circuit.num_detectors), dtype=np.uint8)
    obs = np.zeros((len(signatures), circuit.num_observables), dtype=np.uint8)
    for i, (ds, ls) in enumerate(signatures):
        syn[i,list(ds)] = 1; obs[i,list(ls)] = 1
    return branches, signatures, syn, obs, dict(total_physical_branches=total_branches,
            total_physical_coefficient=total_weight, invisible_branches=total_branches-len(branches),
            invisible_coefficient=total_weight-sum(b['coefficient'] for b in branches))

def oracle_coefficient(signatures, masses):
    by_syn = defaultdict(lambda: defaultdict(float))
    for (ds, ls), weight in zip(signatures, masses): by_syn[ds][ls] += weight
    unavoidable = 0.0
    for ds, groups in by_syn.items():
        unavoidable += (sum(weight for ls, weight in groups.items() if ls)
                        if not ds else sum(groups.values())-max(groups.values()))
    return unavoidable, sum(len(groups)>1 for groups in by_syn.values())

def risk(code, circuit, eps, folder):
    branches, signatures, syn, obs, total = physical_branches(circuit, eps)
    masses = np.zeros(len(signatures))
    for branch in branches: masses[branch['signature']] += branch['coefficient']
    predictions = {}
    records = {}
    for label in ('original', 'strong'):
        state = decoder_state(code, circuit, label)
        zero = predict(state, np.zeros((1,circuit.num_detectors), dtype=np.uint8))
        assert not zero.any(), 'No-fault decision must be handled separately'
        pred = predict(state, syn)
        residual = pred ^ obs
        predictions[label] = pred
        records[label] = dict(coefficient=float(masses @ residual.any(axis=1)),
                              per_logical_coefficient=(masses @ residual).tolist(),
                              single_signature_failures=int(residual.any(axis=1).sum()))
    unavoidable, ambiguous = oracle_coefficient(signatures, masses)
    dem = [i for i in circuit.detector_error_model().flattened() if i.type=='error']
    out = dict(**total, explained_physical_branches=len(branches), signature_count=len(signatures),
               optimal_single_fault_coefficient=unavoidable, ambiguous_syndromes=ambiguous,
               decoder=records, merged_dem_total_probability=sum(i.args_copy()[0] for i in dem),
               merged_dem_logical_probability=sum(i.args_copy()[0] for i in dem if signature(i.targets_copy())[1]),
               coefficient_semantics='sum physical channel Pauli branch p/epsilon; invisible D=L=0 mass retained separately')
    save(folder/'branches.json',dict(branches=branches, signatures=signatures, totals=total))
    np.savez_compressed(folder/'single-fault.npz', detectors=syn, observables=obs,
                        physical_coefficients=masses, **predictions)
    save(folder/'risk.json',out)
    return out

def verify_injections(circuit, folder, count=4):
    data = __import__('json').loads((folder/'branches.json').read_text(encoding='utf-8'))
    branches, signatures = data['branches'], data['signatures']
    selected = list(dict.fromkeys([0, len(branches)//3, 2*len(branches)//3, len(branches)-1]))
    records = []
    for index in selected[:count]:
        branch = branches[index]
        injected = stim.Circuit()
        for j, inst in enumerate(circuit.flattened()):
            if j == branch['instruction_offset']:
                for q, pauli in branch['paulis']: injected.append(pauli+'_ERROR',[q],1)
            if inst.name not in NOISE: injected.append(inst)
        syn, obs = injected.compile_detector_sampler(seed=20261031).sample(16,separate_observables=True)
        ds, ls = signatures[branch['signature']]
        expected_d = np.zeros(circuit.num_detectors, dtype=bool); expected_d[ds] = 1
        expected_l = np.zeros(circuit.num_observables, dtype=bool); expected_l[ls] = 1
        assert np.all(syn == expected_d) and np.all(obs == expected_l)
        records.append(dict(branch=index, injection_shots=16, detectors=ds, observables=ls, equal=True))
    save(folder/'physical-injection-oracle.json', records)
