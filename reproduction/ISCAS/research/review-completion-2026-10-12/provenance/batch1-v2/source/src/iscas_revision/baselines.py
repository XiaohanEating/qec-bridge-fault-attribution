"""Nonlocal BB construction and primary-source routing controls."""
from collections import Counter
import contextlib
import io
import math
import sys
import time
import numpy as np
import stim
from quits import BbCode
from quits.layout import ToricLayout
from .artifacts import ROOT, WORK, HISTORICAL, commit_json, digest, read_json
from .audit import optimal_table
from .decoding import CachedDecoder, full_joint_state, full_joint_predict
from .experiments import EPS, save_npz, decode_sample
from .stratification import clopper_pearson, NOISE
from construction import compile_variant, match_budget, primitive_multiset, inherited
from physical_risk import physical_branches
from pl_validation import sd6


def bb_code(name):
    corpus=read_json(ROOT/"shared/C0/corpus.json")
    item=next(item for item in corpus["instances"] if item["name"]==name)
    return BbCode(**item["quits_ctor"])


def encode_physical(circuit,folder,eps=.001):
    branches,signatures,d,l,total=physical_branches(circuit,eps)
    mass=np.zeros(len(signatures))
    for branch in branches: mass[branch["signature"]]+=branch["coefficient"]
    table,optimum=optimal_table(d,l,mass)
    save_npz(folder/"single-fault.npz",detectors=d,observables=l,physical_coefficients=mass)
    commit_json(folder/"branches.json",dict(branches=branches,signatures=signatures,totals=total))
    commit_json(folder/"physical-risk.json",dict(R_star=optimum,**total,signature_count=len(signatures),
            zero_record_logical_mass=float(mass[((d==0).all(axis=1)) & l.any(axis=1)].sum())))
    return optimum


def prepare_bb(name):
    folder=WORK/"results/bb"/name
    folder.mkdir(parents=True,exist_ok=True)
    code=bb_code(name)
    source_path=folder/"original.stim"
    if source_path.exists(): original=stim.Circuit(source_path.read_text(encoding="utf-8"))
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            original=code.build_circuit(strategy="custom",num_rounds=3,basis="Z",error_model=sd6(.001))
        source_path.write_text(str(original),encoding="utf-8")
    # Natural checkerboard toric layout, with periodic edges routed on an open grid.
    layout=ToricLayout(code)
    placement=layout.node_positions(data_qubits=code.data_qubits,zcheck_qubits=code.zcheck_qubits,
                                    xcheck_qubits=code.xcheck_qubits)
    placement={int(q):tuple(int(value) for value in rc) for q,rc in placement.items()}
    side=max(max(rc)+1 for rc in placement.values())
    side=max(side,math.ceil(math.sqrt(original.num_qubits)))
    original_for_routing=stim.Circuit(str(original))
    original_for_routing.append("I",[side*side-1])
    used=set(placement.values())
    unused=iter((r,c) for r in range(side) for c in range(side) if (r,c) not in used)
    for q in range(original.num_qubits,side*side): placement[q]=next(unused)
    variants={}
    started=time.perf_counter()
    for order,macro,checks in (("forward",False,False),("macro_reverse",True,False),("check_reverse",False,True)):
        variants[order]=compile_variant(original_for_routing,placement,macro_reverse=macro,order_reverse=checks)
    variants=match_budget(variants)
    assert all(primitive_multiset(p)==primitive_multiset(variants["forward"]) for p in variants.values())
    metrics={}
    for order,plan in variants.items():
        target=folder/order
        target.mkdir(parents=True,exist_ok=True)
        commit_json(target/"plan.json",plan)
        circuit,metric=inherited.materialize(plan,.001)
        ideal,_=inherited.materialize(plan,0.)
        d,l=ideal.compile_detector_sampler(seed=830261012).sample(256,separate_observables=True)
        assert not d.any() and not l.any()
        assert circuit.num_observables==code.lz.shape[0]
        (target/"prior.stim").write_text(str(circuit),encoding="utf-8")
        (target/"ideal.stim").write_text(str(ideal),encoding="utf-8")
        metrics[order]=metric
        print(dict(stage="bb_constructed",instance=name,order=order,metrics=metric,logical_bits=circuit.num_observables),flush=True)
    assert all(value==metrics["forward"] for value in metrics.values())
    interactions=Counter((a.qubit_value,b.qubit_value) for inst in original.flattened() if inst.name=="CX"
                         for a,b in zip(inst.targets_copy()[::2],inst.targets_copy()[1::2]))
    distances=Counter()
    for (a,b),count in interactions.items():
        r,c=placement[a];u,v=placement[b]
        distances[abs(r-u)+abs(c-v)]+=count
    result=dict(instance=name,n_data=code.hx.shape[1],k=code.lz.shape[0],hx_shape=list(code.hx.shape),hz_shape=list(code.hz.shape),
                rounds=3,basis="Z",all_logical_bits=True,side=side,placement={str(q):list(rc) for q,rc in placement.items()},
                source_sha256=digest(source_path),metrics=metrics,full_signed_block_checks=True,
                physical_multiset_equal=True,noiseless_check_shots=256,source_cx_distance_histogram=dict(distances),
                remote_cx_occurrences=sum(count for distance,count in distances.items() if distance>1),
                layout="QUITS natural toric checkerboard; open square grid, all periodic/nonlocal edges explicitly routed",
                construction_seconds=time.perf_counter()-started)
    commit_json(folder/"construction.json",result)
    return result


def bb_analysis(task):
    name,order=task["instance"],task["order"]
    folder=WORK/"results/bb"/name/order
    circuit=stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
    if not (folder/"physical-risk.json").exists(): encode_physical(circuit,folder)
    faults=dict(np.load(folder/"single-fault.npz"))
    decoder=CachedDecoder(bb_code(name),circuit,single_fault=faults)
    if not (folder/"decoder-risk.json").exists():
        pred=decoder.predictions(faults["detectors"])
        mass=faults["physical_coefficients"]
        rows={label:dict(R_delta=float(mass @ (p^faults["observables"]).any(axis=1))) for label,p in pred.items()}
        optimum=read_json(folder/"physical-risk.json")["R_star"]
        for row in rows.values(): row["E"]=row["R_delta"]-optimum
        save_npz(folder/"single-fault-decoded.npz",**pred)
        commit_json(folder/"decoder-risk.json",dict(R_star=optimum,decoder=rows))
    results=[]
    from .experiments import rescale_noise
    for i,eps in enumerate(EPS):
        experiment=dict(task,eps=eps,seed=task["seed"]+i*100)
        result=decode_sample(experiment,rescale_noise(circuit,eps/.001),decoder,folder/f"eps{eps}")
        results.append(result)
    return dict(instance=name,order=order,risks=read_json(folder/"decoder-risk.json"),samples=results)


def native_sd6(distance,eps):
    ideal=stim.Circuit.generated("surface_code:unrotated_memory_z",distance=distance,rounds=4).flattened()
    all_qubits=sorted({target.qubit_value for inst in ideal for target in inst.targets_copy() if target.is_qubit_target})
    output=stim.Circuit();active=set();noisy=False;next_noisy=False
    for inst in ideal:
        name=inst.name
        if name=="TICK":
            idle=sorted(set(all_qubits)-active)
            if noisy and idle: output.append("DEPOLARIZE1",idle,eps)
            output.append(inst);active=set();noisy=next_noisy
            continue
        if name in {"M","MX"}: noisy=next_noisy=False
        if name in {"R","RX","MR","MRX","M","MX","H","CX"}:
            targets=inst.targets_copy()
            active.update(target.qubit_value for target in targets)
            if noisy and name in {"M","MX","MR","MRX"}:
                output.append("Z_ERROR" if name in {"MX","MRX"} else "X_ERROR",targets,eps)
            output.append(inst)
            if noisy:
                if name=="CX": output.append("DEPOLARIZE2",targets,eps)
                elif name=="H": output.append("DEPOLARIZE1",targets,eps)
                elif name in {"R","RX","MR","MRX"}: output.append("Z_ERROR" if name in {"RX","MRX"} else "X_ERROR",targets,eps)
            if name in {"MR","MRX"}: next_noisy=True
        else: output.append(inst)
    return output


def generic_analysis(task,circuit,folder,source_details):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"circuit.stim").write_text(str(circuit),encoding="utf-8")
    branches,sig,d,l,total=physical_branches(circuit,task["eps"])
    masses=np.zeros(len(sig))
    for branch in branches: masses[branch["signature"]]+=branch["coefficient"]
    lookup,optimum=optimal_table(d,l,masses)
    lookup[bytes(circuit.num_detectors)]=np.zeros(circuit.num_observables,dtype=np.uint8)
    syn,obs=circuit.compile_detector_sampler(seed=task["seed"]).sample(task["shots"],separate_observables=True)
    save_npz(folder/"physical.npz",detectors=syn,observables=obs,seed=task["seed"])
    rows={}; arrays={};strong_prediction=None
    for label,iters,order in (("full_joint_original",2,0),("full_joint_strong",50,2)):
        state=full_joint_state(circuit,iters,order)
        sf_pred=full_joint_predict(state,d)
        prediction=full_joint_predict(state,syn)
        residual=prediction^obs
        risk=float(masses @ (sf_pred^l).any(axis=1))
        failures=int(residual.any(axis=1).sum())
        rows[label]=dict(R_delta=risk,E=risk-optimum,failures=failures,shots=task["shots"],
                        interval=clopper_pearson(failures,task["shots"]))
        arrays[label+"_prediction"]=prediction;arrays[label+"_residual"]=residual
        if label=="full_joint_strong": strong_prediction=prediction
    pred=strong_prediction.copy()
    for i,record in enumerate(syn):
        if bytes(record) in lookup: pred[i]=lookup[bytes(record)]
    residual=pred^obs
    failures=int(residual.any(axis=1).sum())
    rows["single_fault_lookup"]=dict(R_delta=optimum,E=0.,failures=failures,shots=task["shots"],
                       interval=clopper_pearson(failures,task["shots"]))
    arrays["single_fault_lookup_prediction"]=pred;arrays["single_fault_lookup_residual"]=residual
    save_npz(folder/"decoded.npz",**arrays)
    counts=dict(Counter({"CX":sum(len(inst.targets_copy())//2 for inst in circuit.flattened() if inst.name=="CX"),
                         "TICK":sum(inst.name=="TICK" for inst in circuit.flattened())}))
    result=dict(task=task,R_star=optimum,decoder=rows,resources=counts,noise_channels=total,
                source=source_details,sample_sha256=digest(folder/"physical.npz"),decoded_sha256=digest(folder/"decoded.npz"),
                detector_columns="separate complete (D,L) signatures; no first-logical-label merging")
    commit_json(folder/"result.json",result)
    return result


def native(task):
    circuit=native_sd6(task["distance"],task["eps"])
    folder=WORK/"results/native"/f'd{task["distance"]}'/f'eps{task["eps"]}'
    return generic_analysis(task,circuit,folder,dict(generator="Stim unrotated_memory_z",distance=task["distance"],
                  total_extraction_rounds=4,noisy_rounds=3,initial_extraction_and_terminal_readout_noiseless=True,
                  noise="SD6 active gates, correct-basis preparation/readout flips and every idle qubit per TICK"))


def zhao(task):
    upstream=WORK/"provenance/external/zhao/BB_codes"
    sys.path.insert(0,str(upstream))
    from src.bb_code import BBCode
    from parameters.code_config import get_config
    from circ_gen.circ_gen import gen_circ_only_z_detectors
    from circ_gen.circ_gen_coupler_de import gen_circ_50per_coupler,gen_circ_75per_coupler
    from noise_model.noise_model import standard_depolarizing_noise_model
    config=get_config(1 if task["instance"]=="bb-72" else 4)
    code=BBCode(config.get_params())
    assert code.n==(72 if task["instance"]=="bb-72" else 108)
    with contextlib.redirect_stdout(io.StringIO()):
        if task["order"]=="unrouted": ideal=gen_circ_only_z_detectors(code,3)
        elif task["order"]=="routing50": ideal=gen_circ_50per_coupler(code,3)
        else: ideal=gen_circ_75per_coupler(code,3)
    ideal.detector_error_model()
    d,l=ideal.compile_detector_sampler(seed=task["seed"]).sample(256,separate_observables=True)
    assert not d.any() and not l.any()
    # Original author model is preserved literally, including Pauli axes.
    circuit=standard_depolarizing_noise_model(ideal,code.full_qubit_set,task["eps"])
    folder=WORK/"results/zhao"/task["instance"]/task["order"]/f'eps{task["eps"]}'
    return generic_analysis(task,circuit,folder,dict(repository="Guangqi-Phys/Routing-Circuit",
        commit="295adf67633f223e1cc543d972cd34012c10a4e8",n=code.n,k=code.k,parameters=list(config.get_params()),
        rounds_parameter=3,noise="literal author standard_depolarizing_noise_model; Pauli axes audited separately",
        geometry="author periodic checkerboard and reduced long-range couplers; hardware difference separately reported"))


def hirai_excess(task):
    source=WORK/"provenance/external/hirai/simulation/circuit/uniform_depolarizing"/f'circuit_z_{task["order"]}_memory_d{task["distance"]}_p{task["eps"]}_uniform_depolarizing.stim'
    circuit=stim.Circuit(source.read_text(encoding="utf-8"))
    folder=WORK/"results/hirai-excess"/f'd{task["distance"]}-{task["order"]}'/f'eps{task["eps"]}'
    return generic_analysis(task,circuit,folder,dict(source=str(source),sha256=digest(source),
             purpose="R_star=0 distance-preserving control; decision excess under two full-joint BP-OSD settings"))
