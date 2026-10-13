"""Matched-budget interventions preserving each measurement-separated map."""
from collections import Counter
from copy import deepcopy
import contextlib
import io
import json
import math
from support import HOME, ROOT, stim, code_for, inherited, synthesize, sd6, sha, save, annotations

def compile_variant(original, placement, kind='bridge', macro_reverse=False, order_reverse=False):
    checks = []
    def block(gates, mapping, side, axis, order):
        chosen = list(gates)
        if order_reverse:
            # Only reverse a pairwise commuting block, verified algebraically below.
            chosen.reverse()
        primitives = []
        for a, b in chosen:
            path = inherited.path_between(mapping[a], mapping[b], side, 'row')
            macro = synthesize(path, kind)
            if macro_reverse:
                macro = list(reversed(macro))
            primitives.extend(macro)
        layers = inherited.pack_primitives(primitives)
        inherited.check_block(gates, layers, mapping, mapping, side)
        checks.append(dict(logical_gates=len(gates), signed_full_tableau_equal=True,
                           mapping_before=dict(mapping), mapping_after=dict(mapping)))
        return layers
    saved = inherited.restored_block
    inherited.restored_block = block
    try:
        plan = inherited.compile_plan(original, placement, 'restored', 'row', 'short')
    finally:
        inherited.restored_block = saved
    plan['mechanism_checks'] = checks
    plan['variant'] = dict(kind=kind, macro_reverse=macro_reverse, order_reverse=order_reverse)
    return plan

def match_budget(plans):
    """Pad before each annotation boundary; keep measurements and declared histories."""
    def split(plan):
        sections, current = [], []
        for slot in plan['slots']:
            if 'annotations' in slot:
                sections.append((current, slot)); current = []
            else: current.append(slot)
        sections.append((current, None))
        return sections
    sections = {label: split(plan) for label, plan in plans.items()}
    lengths = {len(value) for value in sections.values()}
    assert len(lengths) == 1
    matched = {key: deepcopy(value) for key, value in plans.items()}
    for plan in matched.values(): plan['slots'] = []
    for j in range(next(iter(lengths))):
        target = max(len(value[j][0]) for value in sections.values())
        for label, value in sections.items():
            slots, annotation = value[j]
            extra = target - len(slots)
            # Padding at segment start preserves all measurement record indices.
            noisy = slots[0]['noisy'] if slots else False
            matched[label]['slots'].extend([dict(operations=[], noisy=noisy) for _ in range(extra)])
            matched[label]['slots'].extend(slots)
            if annotation is not None: matched[label]['slots'].append(annotation)
    return matched

def primitive_multiset(plan):
    return Counter((name, tuple(targets), slot['noisy']) for slot in plan['slots']
                   if 'operations' in slot for name, targets in slot['operations'])

def materialize_sources(plan, eps, source='all'):
    """Keep all idle sites; distinguish idle from single-qubit active channels."""
    full, metrics = inherited.materialize(plan, eps)
    if source == 'all': return full, metrics
    out = stim.Circuit()
    for slot in plan['slots']:
        one = dict(plan, slots=[slot])
        part, _ = inherited.materialize(one, eps)
        operations = slot.get('operations', [])
        active_h = {q for name, targets in operations if name == 'H' for q in targets}
        for inst in part:
            keep = True
            if inst.name == 'DEPOLARIZE2': keep = source == '2q'
            elif inst.name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR'}: keep = source == 'spam'
            elif inst.name == 'DEPOLARIZE1':
                targets = [t.qubit_value for t in inst.targets_copy()]
                is_active = bool(active_h.intersection(targets))
                keep = source == ('sq' if is_active else 'idle')
            if keep: out.append(inst)
    return out, metrics

def original_for(name):
    folder = HOME / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'original.stim'
    if not path.exists():
        if name == 'tiny-hgp':
            source = ROOT / 'ISCAS/research/mechanism-2026-10-10/data/tiny-hgp-cardinal-baseline.stim'
            path.write_bytes(source.read_bytes())
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                original = code_for(name).build_circuit(strategy='cardinal', num_rounds=3, basis='Z', error_model=sd6(.003))
            path.write_text(str(original), encoding='utf-8')
    return stim.Circuit(path.read_text(encoding='utf-8')), sha(path)

def prepare(name):
    original, original_sha = original_for(name)
    side = math.ceil(math.sqrt(original.num_qubits))
    if name == 'tiny-hgp':
        previous = json.loads((ROOT / 'ISCAS/research/mechanism-2026-10-10/data/grid.json').read_text(encoding='utf-8'))
        placement = {int(q): tuple(rc) for q, rc in next(r for r in previous if r['name']==name)['baseline']['placement'].items()}
    else:
        placement = {q: divmod(q, side) for q in range(original.num_qubits)}
    configs = {'forward': (False, False), 'macro_reverse': (True, False), 'check_reverse': (False, True)}
    plans = {label: compile_variant(original, placement, macro_reverse=macro, order_reverse=order)
             for label, (macro, order) in configs.items()}
    native_metrics = {label: inherited.materialize(plan, 0)[1] for label, plan in plans.items()}
    plans = match_budget(plans)
    assert all(primitive_multiset(plan) == primitive_multiset(plans['forward']) for plan in plans.values())
    metrics = {label: inherited.materialize(plan, 0)[1] for label, plan in plans.items()}
    assert all(value == metrics['forward'] for value in metrics.values())
    # published construction is a same-task deployment comparison, not matched-CX intervention.
    plans['kutin'] = compile_variant(original, placement, kind='kutin')
    records = {}
    for label, plan in plans.items():
        folder = HOME / 'results' / name / label
        folder.mkdir(parents=True, exist_ok=True)
        ideal, metric = inherited.materialize(plan, 0)
        assert annotations(ideal) == annotations(original)
        assert ideal.num_measurements == original.num_measurements
        inverse = {int(site): int(q) for q, site in plan['initial_mapping'].items()}
        fixed_names = {'M','MX','MR','R','RX','H'}
        actual_sequence = [(i.name, inverse[t.qubit_value]) for i in ideal
                           if i.name in fixed_names for t in i.targets_copy()]
        original_sequence = [(i.name, t.qubit_value) for i in original.flattened()
                             if i.name in fixed_names for t in i.targets_copy()]
        assert actual_sequence == original_sequence
        ideal.detector_error_model()
        syn, obs = ideal.compile_detector_sampler(seed=20261031).sample(256, separate_observables=True)
        assert not syn.any() and not obs.any()
        save(folder / 'plan.json', plan)
        (folder / 'ideal.stim').write_text(str(ideal), encoding='utf-8')
        records[label] = dict(metrics=metric, unpadded_metrics=native_metrics.get(label, metric),
                             original_sha256=original_sha, ideal_sha256=sha(folder/'ideal.stim'),
                             plan_sha256=sha(folder/'plan.json'), declarations_equal=True,
                             measurement_reset_H_sequence_equal=True, noiseless_check=256,
                             signed_block_checks=plan['mechanism_checks'],
                             physical_multiset_match=label != 'kutin')
    save(HOME/'results'/name/'construction.json', records)
    return plans
