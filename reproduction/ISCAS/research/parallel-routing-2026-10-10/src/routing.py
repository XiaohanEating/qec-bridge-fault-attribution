"""Physical grid routing with explicit slots, maps, and exact block checks."""
from collections import Counter
import math
import stim


def path_between(a, b, side, axis):
    r,c = divmod(a,side)
    br,bc = divmod(b,side)
    path = [a]
    for direction in ((0,1) if axis == 'row' else (1,0)):
        if direction == 0:
            while r != br:
                r += 1 if r < br else -1
                path.append(r*side+c)
        else:
            while c != bc:
                c += 1 if c < bc else -1
                path.append(r*side+c)
    return path


def dependency(gates):
    pred = [0]*len(gates)
    succ = [[] for _ in gates]
    last = {}
    for i,edge in enumerate(gates):
        previous = {last[q] for q in edge if q in last}
        pred[i] = len(previous)
        for j in previous:
            succ[j].append(i)
        for q in edge:
            last[q] = i
    return pred,succ


def pack_primitives(gates):
    """Earliest slot over physical-qubit chains; stable deterministic packing."""
    last = {}
    slots = []
    for a,b in gates:
        t = max(last.get(a,-1), last.get(b,-1))+1
        while len(slots) <= t:
            slots.append([])
        slots[t].append(('CX',(a,b)))
        last[a] = last[b] = t
    return slots


def swap_primitives(a,b):
    return [(a,b),(b,a),(a,b)]


def restored_block(gates, mapping, side, axis, order):
    # Reordering only a logical ready set preserves all logical-qubit chains.
    pred,succ = dependency(gates)
    ready = {i for i,p in enumerate(pred) if p == 0}
    physical = []
    while ready:
        priority = lambda i:(-(len(path_between(mapping[gates[i][0]],mapping[gates[i][1]],side,axis))) if order == 'long' else 0,i)
        i = min(ready,key=priority)
        ready.remove(i)
        a,b = gates[i]
        path = path_between(mapping[a],mapping[b],side,axis)
        swaps = list(zip(path[:-2],path[1:-1]))
        for u,v in swaps:
            physical.extend(swap_primitives(u,v))
        physical.append((path[-2],path[-1]))
        for u,v in reversed(swaps):
            physical.extend(swap_primitives(u,v))
        for j in succ[i]:
            pred[j] -= 1
            if pred[j] == 0:
                ready.add(j)
    return pack_primitives(physical)


def dynamic_block(gates, mapping, inverse, side, axis, order):
    """Pack disjoint shortest paths, retain swapped placement across operations.

    Every batch routes a maximal set of ready logical CNOTs on vertex-disjoint
    paths. Paths are conservative reservations. Each SWAP subgate lasts one
    slot; swaps on disjoint paths overlap. There is no return-trip requirement.
    """
    pred,succ = dependency(gates)
    ready = {i for i,p in enumerate(pred) if p == 0}
    slots = []
    def swap_map(u,v):
        qa,qb = inverse[u],inverse[v]
        inverse[u],inverse[v] = qb,qa
        mapping[qa],mapping[qb] = v,u
    while ready:
        paths = {i:path_between(mapping[gates[i][0]],mapping[gates[i][1]],side,axis) for i in ready}
        priority = lambda i:(-len(paths[i]) if order == 'long' else len(paths[i]),i)
        chosen,used = [],set()
        for i in sorted(ready,key=priority):
            if not used.intersection(paths[i]):
                chosen.append(i)
                used.update(paths[i])
        assert chosen
        max_swaps = max(len(paths[i])-2 for i in chosen)
        for step in range(max_swaps):
            swap_edges = [(paths[i][step],paths[i][step+1]) for i in chosen if step < len(paths[i])-2]
            for sub in range(3):
                slots.append([('CX',edge if sub != 1 else (edge[1],edge[0])) for edge in swap_edges])
            for u,v in swap_edges:
                swap_map(u,v)
        cx = [(mapping[gates[i][0]],mapping[gates[i][1]]) for i in chosen]
        slots.append([('CX',edge) for edge in cx])
        for i in chosen:
            ready.remove(i)
            for j in succ[i]:
                pred[j] -= 1
                if pred[j] == 0:
                    ready.add(j)
    return slots


def check_block(gates, slots, before, after, side):
    n = side*side
    expected,actual = stim.Circuit(),stim.Circuit()
    expected.append('I',[n-1]); actual.append('I',[n-1])
    for a,b in gates:
        expected.append('CX',[before[a],before[b]])
    for layer in slots:
        for name,edge in layer:
            actual.append(name,list(edge))
    # Algebraic correction of output permutation only for this exact check.
    mapping = dict(after)
    inverse = {v:k for k,v in mapping.items()}
    for q in sorted(mapping):
        if mapping[q] != before[q]:
            a,b = mapping[q],before[q]
            actual.append('SWAP',[a,b])
            other = inverse[b]
            inverse[a],inverse[b] = other,q
            mapping[q],mapping[other] = b,a
    assert mapping == before
    assert stim.Tableau.from_circuit(actual) == stim.Tableau.from_circuit(expected)


def compile_plan(circuit, placement, router='dynamic', axis='row', order='long'):
    nlogical = circuit.num_qubits
    side = math.ceil(math.sqrt(nlogical))
    n = side*side
    mapping = {int(q):int(rc[0])*side+int(rc[1]) for q,rc in placement.items()}
    occupied = set(mapping.values())
    for q,site in zip(range(nlogical,n),sorted(set(range(n))-occupied)):
        mapping[q] = site
    assert len(mapping) == len(set(mapping.values())) == n
    inverse = {v:k for k,v in mapping.items()}
    initial_mapping = dict(mapping)
    flat = circuit.flattened()
    prefix = []
    for inst in flat:
        if inst.name == 'CX':
            break
        prefix.append(inst.name)
    initial_noisy = any(name in {'X_ERROR','Z_ERROR','DEPOLARIZE1'} for name in prefix)
    slots,checks = [],[]
    current = []
    noisy = initial_noisy
    i = 0
    while i < len(flat):
        inst = flat[i]
        name = inst.name
        if name in {'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}:
            i += 1
            continue
        if name in {'DETECTOR','OBSERVABLE_INCLUDE'}:
            if current:
                slots.append({'operations':current,'noisy':noisy})
                current = []
            slots.append({'annotations':[str(inst)]})
            i += 1
            continue
        if name == 'TICK':
            if current:
                slots.append({'operations':current,'noisy':noisy})
                current = []
            i += 1
            continue
        if name == 'CX':
            assert not current
            gates = []
            j = i
            while j < len(flat) and flat[j].name in {'CX','TICK','DEPOLARIZE1','DEPOLARIZE2'}:
                if flat[j].name == 'CX':
                    ts = [t.qubit_value for t in flat[j].targets_copy()]
                    gates.extend(zip(ts[::2],ts[1::2]))
                j += 1
            before = dict(mapping)
            if router == 'restored':
                layers = restored_block(gates,mapping,side,axis,order)
            elif router == 'dynamic':
                layers = dynamic_block(gates,mapping,inverse,side,axis,order)
            else:
                raise ValueError(router)
            check_block(gates,layers,before,mapping,side)
            slots.extend({'operations':layer,'noisy':noisy} for layer in layers)
            checks.append({'original_cx':len(gates),'physical_cx':sum(len(l) for l in layers),
                           'physical_slots':len(layers),'full_tableau_equal_after_permutation':True})
            i = j
            continue
        if name in {'R','RX','H','MR','M','MX'}:
            targets = tuple(mapping[t.qubit_value] for t in inst.targets_copy())
            if name in {'M','MX'}:
                # Only MR-based rounds are supported here; final measurement is noiseless.
                noisy = False
            current.append((name,targets))
            if name == 'MR':
                # Finish this measurement slot before switching from an ideal zeroth round.
                slots.append({'operations':current,'noisy':noisy})
                current = []
                noisy = True
            i += 1
            continue
        if name in {'I','QUBIT_COORDS','SHIFT_COORDS'}:
            i += 1
            continue
        raise ValueError(f'Unsupported instruction {name}')
    if current:
        slots.append({'operations':current,'noisy':noisy})
    plan = {'resource_sites':n,'side':side,'router':router,'axis':axis,'order':order,
            'slots':slots,'block_checks':checks,'initial_mapping':{str(q):v for q,v in initial_mapping.items()},
            'final_mapping':{str(q):v for q,v in mapping.items()}}
    return plan


def materialize(plan, eps):
    """Unit duration for each primitive CX/H/R/MR/M slot, SD6 on active/idle.

    SWAP consists of three CX subslots. SPAM is before measurement and after
    reset; idle covers every reserved physical site on every noisy slot.
    """
    n,side = plan['resource_sites'],plan['side']
    circuit = stim.Circuit()
    circuit.append('QUBIT_COORDS',[n-1],[side-1,side-1])
    metrics = Counter()
    for slot in plan['slots']:
        if 'annotations' in slot:
            circuit += stim.Circuit('\n'.join(slot['annotations']))
            continue
        active = set()
        operations = slot['operations']
        noisy = slot['noisy'] and eps > 0
        for name,targets in operations:
            assert len(targets) == len(set(targets))
            assert not active.intersection(targets), 'Resource overlap'
            active.update(targets)
            if name == 'CX':
                a,b = targets
                assert abs(a//side-b//side)+abs(a%side-b%side) == 1
            if noisy and name in {'M','MX','MR'}:
                circuit.append('Z_ERROR' if name == 'MX' else 'X_ERROR',list(targets),eps)
            circuit.append(name,list(targets))
            if noisy:
                if name == 'CX':
                    circuit.append('DEPOLARIZE2',list(targets),eps)
                elif name == 'H':
                    circuit.append('DEPOLARIZE1',list(targets),eps)
                elif name in {'R','RX','MR'}:
                    circuit.append('Z_ERROR' if name == 'RX' else 'X_ERROR',list(targets),eps)
            metrics['num_2q' if name == 'CX' else 'num_1q' if name == 'H' else 'num_reset' if name in {'R','RX'} else 'num_measure'] += len(targets)//2 if name == 'CX' else len(targets)
        idle = sorted(set(range(n))-active)
        metrics['full_lifetime_idle_slots'] += len(idle)
        if slot['noisy']:
            metrics['noisy_idle_locations'] += len(idle)
            metrics['noisy_slots'] += 1
        if noisy and idle:
            circuit.append('DEPOLARIZE1',idle,eps)
        circuit.append('TICK')
        metrics['physical_slots'] += 1
    return circuit,dict(metrics)
