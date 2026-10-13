"""Resumable tasks; commit physical samples before decoding."""
from collections import Counter
from datetime import datetime, timezone
import contextlib
import io
import json
import math
import os
import time
import numpy as np
import stim
from scipy.stats import binom
from .artifacts import ROOT, WORK, HISTORICAL, commit_json, digest, read_json
from .audit import optimal_table, paired_comparison
from .decoding import CachedDecoder, CONFIGURATIONS, full_joint_state, full_joint_predict
from .stratification import PhysicalResponses, NOISE, integrate, clopper_pearson
from support import code_for
from construction import inherited

EPS=(1e-6,3e-6,1e-5,3e-5,1e-4)
CONTEXTS={}


def rescale_noise(circuit, factor):
    output=stim.Circuit()
    for inst in circuit.flattened():
        if inst.name in NOISE:
            output.append(inst.name,inst.targets_copy(),[inst.gate_args_copy()[0]*factor])
        else: output.append(inst)
    return output


def context(instance,order):
    key=(instance,order)
    if key not in CONTEXTS:
        folder=HISTORICAL/"results"/instance/order
        circuit=stim.Circuit((HISTORICAL/"provenance/decoder-priors"/instance/(order+".stim")).read_text(encoding="utf-8"))
        faults=dict(np.load(folder/"single-fault.npz"))
        model=PhysicalResponses(circuit,read_json(folder/"branches.json"),faults)
        decoder=CachedDecoder(code_for(instance),circuit,single_fault=faults)
        # Verified historical fixed-prior predictions are an exact reusable cache.
        for label in ("original","strong"):
            decoder.cache[label].update({bytes(d):p.copy() for d,p in zip(faults["detectors"],faults[label])})
        CONTEXTS[key]=(circuit,faults,model,decoder)
    return CONTEXTS[key]


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)


def stratum(task):
    instance,order,k=task["instance"],task["order"],task["k"]
    circuit,faults,model,decoder=context(instance,order)
    folder=WORK/"results/strata"/instance/order/f"k{k:02d}"
    records=[]
    for shard in range(task["shots"]//task["shard_shots"]):
        target=folder/f"shard{shard:03d}"
        result_path=target/"result.json"
        if result_path.exists():
            record=read_json(result_path)
            assert record["sample_sha256"] == digest(target/"physical.npz")
            assert record["decoded_sha256"] == digest(target/"decoded.npz")
            records.append(record); continue
        started=time.perf_counter()
        seed=task["seed"]+shard
        sample_path=target/"physical.npz"
        if not sample_path.exists():
            d,l,sites,signatures=model.sample(k,task["shard_shots"],seed)
            save_npz(sample_path,detectors=d,observables=l,sites=sites,signatures=signatures,seed=seed)
        sample=np.load(sample_path)
        assert int(sample["seed"]) == seed
        predictions=decoder.predictions(sample["detectors"])
        arrays={}
        failures={}
        for name,pred in predictions.items():
            residual=pred^sample["observables"]
            arrays[name+"_prediction"]=pred
            arrays[name+"_residual"]=residual
            failures[name]=int(residual.any(axis=1).sum())
        save_npz(target/"decoded.npz",**arrays)
        record=dict(seed=seed,shots=len(sample["detectors"]),failures=failures,
                    sample_sha256=digest(sample_path),decoded_sha256=digest(target/"decoded.npz"),
                    seconds=time.perf_counter()-started)
        commit_json(result_path,record)
        records.append(record)
        commit_json(folder/"progress.json",dict(k=k,committed_shots=sum(r["shots"] for r in records),
                    target_shots=task["shots"],last_shard=shard,pid=os.getpid(),status="running"))
    result=dict(instance=instance,order=order,k=k,shots=sum(r["shots"] for r in records),
                failures={name:sum(r["failures"][name] for r in records) for name in ("original","strong","single_fault_lookup")},
                shard_count=len(records),noise_locations=model.n)
    commit_json(folder/"result.json",result)
    return result


def decode_sample(task,circuit,decoder,folder):
    sample_path=folder/"physical.npz"
    if not sample_path.exists():
        d,l=circuit.compile_detector_sampler(seed=task["seed"]).sample(task["shots"],separate_observables=True)
        save_npz(sample_path,detectors=d,observables=l,seed=task["seed"])
    sample=np.load(sample_path)
    assert len(sample["detectors"])==task["shots"] and int(sample["seed"])==task["seed"]
    arrays={}; predictions=decoder.predictions(sample["detectors"])
    rows={}
    for name,pred in predictions.items():
        residual=pred^sample["observables"]
        failures=int(residual.any(axis=1).sum())
        rows[name]=dict(shots=task["shots"],failures=failures,rate=failures/task["shots"],
                       interval=clopper_pearson(failures,task["shots"]),per_logical=residual.sum(axis=0).tolist())
        arrays[name+"_prediction"]=pred; arrays[name+"_residual"]=residual
    save_npz(folder/"decoded.npz",**arrays)
    result=dict(task=task,decoder=rows,sample_sha256=digest(sample_path),decoded_sha256=digest(folder/"decoded.npz"),
                paired=paired_comparison(arrays["original_residual"],arrays["strong_residual"]))
    commit_json(folder/"result.json",result)
    return result


def direct(task):
    circuit,faults,model,decoder=context(task["instance"],task["order"])
    circuit=rescale_noise(circuit,task["eps"]/.001)
    folder=WORK/"results/direct"/task["instance"]/task["order"]/f'eps{task["eps"]}'
    return decode_sample(task,circuit,decoder,folder)


def onefactor(task):
    instance,order=task["instance"],task["order"]
    circuit,faults,model,_=context(instance,order)
    sample_path=HISTORICAL/"results"/instance/order/"samples-v2/all-eps0.0001-n4096/physical-samples.npz"
    samples=np.load(sample_path)
    d,l=samples["detectors"],samples["observables"]
    prior=circuit if task["prior"] == "fixed" else rescale_noise(circuit,.1)
    decoder=CachedDecoder(code_for(instance),prior,configurations=tuple(CONFIGURATIONS),single_fault=faults)
    # All configurations' single-fault predictions computed under this precise prior.
    sf=decoder.predictions(faults["detectors"],include_lookup=False)
    risks={name:float(faults["physical_coefficients"] @ (pred^faults["observables"]).any(axis=1)) for name,pred in sf.items()}
    predictions=decoder.predictions(d)
    arrays={};rows={}
    for name,pred in predictions.items():
        residual=pred^l
        arrays[name+"_prediction"]=pred; arrays[name+"_residual"]=residual
        rows[name]=dict(failures=int(residual.any(axis=1).sum()),shots=len(d),single_fault_risk=risks.get(name))
    folder=WORK/"results/onefactor"/instance/order/task["prior"]
    save_npz(folder/"decoded.npz",**arrays)
    result=dict(instance=instance,order=order,prior=task["prior"],prior_epsilon=.001 if task["prior"]=="fixed" else .0001,
                sample_sha256=digest(sample_path),configurations=CONFIGURATIONS,results=rows,
                paired={name:paired_comparison(arrays["original_residual"],arrays[name+"_residual"])
                        for name in predictions if name != "original"})
    commit_json(folder/"result.json",result)
    return result


def idle_circuit(plan,eps,rho):
    if rho==1: return inherited.materialize(plan,eps)[0]
    output=stim.Circuit()
    for slot in plan["slots"]:
        piece,_=inherited.materialize(dict(plan,slots=[slot]),eps)
        active_h={q for name,targets in slot.get("operations",[]) if name=="H" for q in targets}
        for inst in piece:
            if inst.name != "DEPOLARIZE1": output.append(inst); continue
            targets=[t.qubit_value for t in inst.targets_copy()]
            active=[q for q in targets if q in active_h]
            idle=[q for q in targets if q not in active_h]
            if active: output.append("DEPOLARIZE1",active,eps)
            if idle and rho: output.append("DEPOLARIZE1",idle,rho*eps)
    return output


def sensitivity(task):
    instance,order=task["instance"],task["order"]
    folder0=HISTORICAL/"results"/instance/order
    plan=read_json(folder0/"plan.json")
    prior,faults,model,decoder=context(instance,order)
    circuit=idle_circuit(plan,task["eps"],task["rho"])
    folder=WORK/"results/sensitivity"/instance/order/f'rho{task["rho"]}-eps{task["eps"]}'
    return decode_sample(task,circuit,decoder,folder)


def historical_baseline(task):
    label=task["order"]
    if label in {"dynamic646","region1"}:
        folder0=HISTORICAL/"results/budget-local"/("baseline" if label=="dynamic646" else "region-1")
    else:
        folder0=HISTORICAL/"results/tiny-hgp"/label
    plan=read_json(folder0/"plan.json")
    prior,metrics=inherited.materialize(plan,.001)
    circuit=rescale_noise(prior,task["eps"]/.001)
    faults=dict(np.load(folder0/"single-fault.npz"))
    decoder=CachedDecoder(code_for("tiny-hgp"),prior,single_fault=faults)
    folder=WORK/"results/deployment"/label/f'eps{task["eps"]}'
    result=decode_sample(task,circuit,decoder,folder)
    result["metrics"]=metrics
    result["plan_sha256"]=digest(folder0/"plan.json")
    result["older_budget_cap"]=1.05*646
    result["within_cap"]=metrics["num_2q"] <= 1.05*646
    commit_json(folder/"result.json",result)
    return result


def hirai(task):
    import pymatching
    name,distance,eps=task["order"],task["distance"],task["eps"]
    source=WORK/"provenance/external/hirai/simulation/circuit/uniform_depolarizing"/f'circuit_z_{name}_memory_d{distance}_p{eps}_uniform_depolarizing.stim'
    circuit=stim.Circuit(source.read_text(encoding="utf-8"))
    graphdem=circuit.detector_error_model(decompose_errors=True)
    matching=pymatching.Matching.from_detector_error_model(graphdem)
    syn,obs=circuit.compile_detector_sampler(seed=task["seed"]).sample(task["shots"],separate_observables=True)
    prediction=matching.decode_batch(syn)
    if prediction.ndim==1: prediction=prediction[:,None]
    residual=prediction^obs
    folder=WORK/"results/hirai"/f'd{distance}-{name}'/f'eps{eps}'
    save_npz(folder/"physical.npz",detectors=syn,observables=obs,seed=task["seed"])
    save_npz(folder/"decoded.npz",matching_prediction=prediction,matching_residual=residual)
    from physical_risk import physical_branches
    branches,sig,det,logical,total=physical_branches(circuit,eps)
    weights=np.zeros(len(sig))
    for branch in branches: weights[branch["signature"]]+=branch["coefficient"]
    _,optimum=optimal_table(det,logical,weights)
    sf_prediction=matching.decode_batch(det)
    if sf_prediction.ndim==1: sf_prediction=sf_prediction[:,None]
    actual=float(weights @ (sf_prediction^logical).any(axis=1))
    result=dict(task=task,source_sha256=digest(source),source=str(source),
                R_star=optimum,R_matching=actual,E_matching=actual-optimum,
                graphlike_distance=len(circuit.shortest_graphlike_error()),
                shots=task["shots"],failures=int(residual.any(axis=1).sum()),
                interval=clopper_pearson(int(residual.any(axis=1).sum()),task["shots"]),
                decoder="official PyMatching; fixed-shot new confirmation, same raw published circuit",
                resource_counts=dict(Counter({"CX":sum(len(inst.targets_copy())//2 for inst in circuit.flattened() if inst.name=="CX"),
                     "TICK":sum(inst.name=="TICK" for inst in circuit.flattened())})),
                noise_locations=total["total_physical_coefficient"],noise="author's original uniform depolarizing",
                sample_sha256=digest(folder/"physical.npz"),decoded_sha256=digest(folder/"decoded.npz"))
    commit_json(folder/"result.json",result)
    return result


def dispatch(task):
    identifier=task["id"]
    complete=WORK/"results/tasks"/(identifier+".json")
    if complete.exists(): return read_json(complete)
    lock=WORK/"results/locks"/(identifier+".lock")
    lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    started=time.perf_counter()
    try:
        operations={"stratum":stratum,"direct":direct,"onefactor":onefactor,"sensitivity":sensitivity,
                   "deployment":historical_baseline,"hirai":hirai}
        if task["kind"] in {"bb","native","zhao","hirai_excess"}:
            from .baselines import bb_analysis,native,zhao,hirai_excess
            operations.update(bb=bb_analysis,native=native,zhao=zhao,hirai_excess=hirai_excess)
        if task["kind"]=="order_space":
            from .order_space import execute
            operations["order_space"]=execute
        operation=operations[task["kind"]]
        result=operation(task)
        summary=dict(id=identifier,kind=task["kind"],status="complete",pid=os.getpid(),seconds=time.perf_counter()-started,
                     completed_utc=datetime.now(timezone.utc).isoformat(),result=result)
        commit_json(complete,summary)
        return dict(id=identifier,status="complete",seconds=summary["seconds"])
    except Exception as error:
        import traceback
        commit_json(WORK/"results/errors"/(identifier+".json"),dict(task=task,error=str(error),traceback=traceback.format_exc()))
        raise
    finally: lock.unlink(missing_ok=True)


def integrate_results():
    records={}
    for instance in ("tiny-hgp","holdout-hgp-3x4"):
        for order in ("forward","macro_reverse","check_reverse"):
            saved=read_json(HISTORICAL/"results"/instance/order/"risk.json")
            n=int(saved["total_physical_coefficient"])
            maximum=int(binom.ppf(1-1e-8,n,max(EPS)))
            strata={}
            for k in range(2,maximum+1):
                path=WORK/"results/strata"/instance/order/f"k{k:02d}"/"result.json"
                if not path.exists(): break
                strata[k]=read_json(path)
            if not strata: continue
            optimum=saved["optimal_single_fault_coefficient"]
            coeff={name:saved["decoder"][name]["coefficient"] for name in ("original","strong")}
            coeff["single_fault_lookup"]=optimum
            values={str(eps):{name:integrate(n,eps,risk,{k:dict(shots=value["shots"],failures=value["failures"][name])
                    for k,value in strata.items()}) for name,risk in coeff.items()} for eps in EPS}
            records[instance+"/"+order]=dict(completed_strata=sorted(strata),target_kmax=maximum,
                   status="complete" if max(strata)==maximum else "partial; remaining mass bounded in intervals",values=values)
    commit_json(WORK/"results/low-noise-summary.json",records)
    return records
