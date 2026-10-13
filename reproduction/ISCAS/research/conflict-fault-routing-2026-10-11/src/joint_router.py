"""Load-aware placement and commutation-safe remote-CX route scheduling."""
from collections import Counter
import contextlib
import math
from pathlib import Path
import sys
import stim

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'ISCAS/research/parallel-routing-2026-10-10/src'))
import routing as inherited


def synthesize(path,kind):
    edges = list(zip(path[:-1],path[1:]))
    if kind=='swap':
        front = [g for edge in edges[:-1] for g in inherited.swap_primitives(*edge)]
        back = [g for edge in reversed(edges[:-1]) for g in inherited.swap_primitives(*edge)]
        return front+[edges[-1]]+back
    if kind=='bridge':
        if len(edges)==1:return edges
        return edges+list(reversed(edges[:-1]))+edges[1:]+list(reversed(edges[1:-1]))
    if kind=='kutin':
        n=len(path);k=(n+1)//2
        front=[]
        for i in range(k-1):front.append((path[i+1],path[i]))
        for i in range(k-1):front.append((path[i],path[i+1]))
        for i in range(n-2,k-1,-1):front.append((path[i+1],path[i]))
        for i in range(n-2,k-1,-1):front.append((path[i],path[i+1]))
        return front+[(path[k-1],path[k])]+list(reversed(front))
    raise ValueError(kind)


def check_identity():
    checks = []
    for d in range(1,13):
        expected = stim.Circuit();expected.append('CX',[0,d]);expected.append('I',[d])
        for kind in ('swap','bridge'):
            gates = synthesize(list(range(d+1)),kind)
            actual=stim.Circuit()
            for edge in gates:actual.append('CX',edge)
            assert stim.Tableau.from_circuit(actual)==stim.Tableau.from_circuit(expected)
            expected_count = 6*d-5 if kind=='swap' else 1 if d==1 else 4*d-4
            assert len(gates)==expected_count
            checks.append({'distance':d,'kind':kind,'num_cx':len(gates),'general_state_tableau_equal':True})
    return checks


def precedence(gates,commuting):
    if not commuting:return inherited.dependency(gates)
    # All prior opposite-role sharing must precede this gate; common-control
    # and common-target operations commute and impose no artificial chain.
    controls,targets = {},{}
    pred=[0]*len(gates);succ=[[] for _ in gates]
    for i,(a,b) in enumerate(gates):
        previous = set(targets.get(a,()))|set(controls.get(b,()))
        pred[i]=len(previous)
        for j in previous:succ[j].append(i)
        controls.setdefault(a,[]).append(i)
        targets.setdefault(b,[]).append(i)
    return pred,succ


def trial_end(primitives,last):
    clocks=dict(last)
    for a,b in primitives:
        t=max(clocks.get(a,0),clocks.get(b,0))+1
        clocks[a]=clocks[b]=t
    return max(clocks.values(),default=0),clocks


def schedule_block(gates,mapping,side,kind='swap',commuting=True,conflict=True):
    pred,succ=precedence(gates,commuting)
    ready={i for i,p in enumerate(pred) if p==0}
    primitives=[];last={};route_records=[];load=Counter()
    rank=[1]*len(gates)
    for i in reversed(range(len(gates))):rank[i]=1+max((rank[j] for j in succ[i]),default=0)
    while ready:
        best=None
        for i in sorted(ready):
            a,b=gates[i]
            for axis in ('row','col') if conflict else ('row',):
                path=inherited.path_between(mapping[a],mapping[b],side,axis)
                macro=synthesize(path,kind)
                finish,clocks=trial_end(macro,last)
                if conflict:
                    add=Counter(q for edge in macro for q in edge)
                    projected={q:load[q]+add[q] for q in load.keys()|add.keys()}
                    # The first two terms directly target congestion; finish is
                    # only a tie-breaker because the final packer may expose
                    # parallelism hidden by serial macro append order.
                    squared=0
                    for amount in projected.values():squared+=amount*amount
                    key=(max(projected.values(),default=0),squared,
                         finish,-rank[i],len(macro),i,axis)
                else:key=(i,)
                if best is None or key<best[0]:best=(key,i,path,macro,clocks)
        _,i,path,macro,last=best
        ready.remove(i);primitives.extend(macro)
        load.update(q for edge in macro for q in edge)
        route_records.append({'gate':i,'path':path})
        for j in succ[i]:
            pred[j]-=1
            if pred[j]==0:ready.add(j)
    layers=inherited.pack_primitives(primitives)
    load=Counter(q for edge in primitives for q in edge)
    squared=0
    for amount in load.values():squared+=amount*amount
    return layers,{'max_site_load':max(load.values(),default=0),'load_second_moment':squared,
                   'num_precedence_arcs':sum(map(len,succ)),'logical_gates':len(gates),
                   'physical_gates':len(primitives),'physical_slots':len(layers),
                   'route_digest':__import__('hashlib').sha256(str(route_records).encode()).hexdigest()}


def compile_joint(circuit,placement,kind='swap',commuting=True,conflict=True):
    stats=[];cache={}
    def substitute(gates,mapping,side,axis,order):
        key=(tuple(gates),tuple(sorted(mapping.items())))
        if key not in cache:cache[key]=schedule_block(gates,mapping,side,kind,commuting,conflict)
        layers,info=cache[key];stats.append(info)
        return layers
    old=inherited.restored_block
    inherited.restored_block=substitute
    try:plan=inherited.compile_plan(circuit,placement,'restored','row','short')
    finally:inherited.restored_block=old
    plan['router']='joint_'+kind
    plan['commuting']=commuting;plan['conflict']=conflict;plan['load_stats']=stats
    return plan


def block_gates(circuit):
    flat=circuit.flattened();i=0;out=[]
    while i<len(flat):
        if flat[i].name!='CX':i+=1;continue
        gates=[]
        while i<len(flat) and flat[i].name in {'CX','TICK','DEPOLARIZE1','DEPOLARIZE2'}:
            if flat[i].name=='CX':
                ts=[t.qubit_value for t in flat[i].targets_copy()];gates.extend(zip(ts[::2],ts[1::2]))
            i+=1
        if tuple(gates) not in [tuple(g) for g in out]:out.append(gates)
    return out


def load_objective(gates,placement,side,kind):
    # Optimizing a lower bound is useful only after evaluating actual schedule.
    load=Counter();cost=0
    for a,b in gates:
        u=placement[a][0]*side+placement[a][1];v=placement[b][0]*side+placement[b][1]
        chosen=None
        for axis in ('row','col'):
            macro=synthesize(inherited.path_between(u,v,side,axis),kind)
            add=Counter(q for edge in macro for q in edge)
            peak=0;second=0
            for site in load.keys()|add.keys():
                amount=load[site]+add[site]
                assert isinstance(amount,int),(site,type(amount),type(load),type(add))
                peak=max(peak,amount);second+=amount*amount
            key=(peak,second)
            if chosen is None or key<chosen[0]:chosen=(key,add,len(macro))
        load.update(chosen[1]);cost+=chosen[2]
    squared=0
    for amount in load.values():squared+=amount*amount
    return (max(load.values(),default=0),squared,cost),load


def improve_load_layout(circuit,initial,kind,moves=4):
    placement=dict(initial);side=math.ceil(math.sqrt(circuit.num_qubits))
    gates=block_gates(circuit)[0]
    neighbors={q:[] for q in placement}
    for a,b in gates:neighbors[a].append(b);neighbors[b].append(a)
    trace=[]
    for step in range(moves):
        before,load=load_objective(gates,placement,side,kind)
        inverse={site:q for q,site in placement.items()}
        heavy=sorted(placement,key=lambda q:-load[placement[q][0]*side+placement[q][1]])[:4]
        best=None
        for a in heavy:
            partners=neighbors[a]
            if not partners:continue
            center=(sum(placement[q][0] for q in partners)/len(partners),sum(placement[q][1] for q in partners)/len(partners))
            targets=sorted(placement,key=lambda b:(abs(placement[b][0]-center[0])+abs(placement[b][1]-center[1]),b))[:6]
            for b in targets:
                if a==b:continue
                placement[a],placement[b]=placement[b],placement[a]
                after,_=load_objective(gates,placement,side,kind)
                placement[a],placement[b]=placement[b],placement[a]
                if after<before and (best is None or after<best[0]):best=(after,a,b)
        if best is None:break
        after,a,b=best;placement[a],placement[b]=placement[b],placement[a]
        trace.append({'step':step,'swap':[a,b],'before':before,'after':after})
    return placement,trace
