"""Validate committed scientific arrays, residual counts, seeds, and sources."""
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import stim

from .artifacts import WORK, HISTORICAL, read_json, digest, commit_json
from .stratification import PhysicalResponses, clopper_pearson
from .publication import equal_arrays


def run():
    counts=Counter(); inputs={}; models={}; checked=[]
    def identity(path, expected=None):
        value=digest(path)
        if expected is not None: assert value==expected, str(path)
        if str(path) in inputs: assert inputs[str(path)]==value, str(path)
        inputs[str(path)]=value
        return value
    for path in sorted((WORK/"results").rglob("result.json")):
        record=read_json(path)
        if "sample_sha256" not in record or "decoded_sha256" not in record:
            continue
        folder=path.parent
        physical=folder/"physical.npz"
        if "sample_source" in record: physical=WORK.__class__(record["sample_source"])
        if not physical.exists() and "configurations" in record:
            physical=HISTORICAL/"results"/record["instance"]/record["order"]/"samples-v2/all-eps0.0001-n4096/physical-samples.npz"
        identity(physical,record["sample_sha256"])
        decoded=folder/"decoded.npz"
        identity(decoded,record["decoded_sha256"])
        with np.load(physical) as archive: sample={name:archive[name]for name in archive.files}
        with np.load(decoded) as archive: arrays={name:archive[name]for name in archive.files}
        n=len(sample["detectors"]); nl=sample["observables"].shape[1]
        task=record.get("task",{})
        if "seed" in sample:
            expected=task.get("seed",record.get("seed"))
            if expected is not None: assert int(sample["seed"])==expected, str(path)
        if "shots" in task: assert task["shots"]==n, str(path)
        rows=record.get("decoder",record.get("results",{}))
        for name,prediction in arrays.items():
            if not name.endswith("_prediction") or name.endswith("_single_fault_prediction"): continue
            label=name[:-11]
            residual=arrays[label+"_residual"]
            assert prediction.shape==residual.shape==(n,nl), str(path)
            assert np.all((prediction==0)|(prediction==1)), str(path)
            assert np.array_equal(prediction ^ sample["observables"],residual), str(path)
            failures=int(residual.any(axis=1).sum())
            if label in rows:
                assert failures==rows[label]["failures"], str(path)
                if "shots" in rows[label]: assert n==rows[label]["shots"]
                if "interval" in rows[label]: assert np.allclose(rows[label]["interval"],clopper_pearson(failures,n),atol=1e-13)
                if "per_logical" in rows[label]: assert residual.sum(axis=0).tolist()==rows[label]["per_logical"]
            elif isinstance(record.get("failures"),dict):
                assert failures==record["failures"][label], str(path)
            elif label=="matching": assert failures==record["failures"], str(path)
            else: raise AssertionError((str(path),label))
            counts["decision_arrays"]+=1
        if "sites" in sample:
            instance,order=folder.parents[2].name,folder.parents[1].name
            key=instance,order
            if key not in models:
                source=HISTORICAL/"results"/instance/order
                prior=HISTORICAL/"provenance/decoder-priors"/instance/(order+".stim")
                sf=dict(np.load(source/"single-fault.npz"))
                models[key]=PhysicalResponses(stim.Circuit(prior.read_text(encoding="utf-8")),read_json(source/"branches.json"),sf)
                identity(prior);identity(source/"single-fault.npz")
            model=models[key]
            sites=sample["sites"]; signatures=sample["signatures"]
            assert np.all(np.diff(np.sort(sites,axis=1),axis=1)>0),str(path)
            assert np.all((sites>=0)&(sites<model.n)),str(path)
            valid=np.any(model.channels[sites]==signatures[:,:,None],axis=2)
            assert valid.all(),str(path)
            response=np.bitwise_xor.reduce(model.responses[signatures],axis=1)
            assert np.array_equal(response,np.concatenate([sample["detectors"],sample["observables"]],axis=1)),str(path)
            counts["conditional_shards"]+=1
        checked.append(str(path));counts["result_records"]+=1
    for kind,basename,target_name in (("bb-parallel","decoded.npz","single-fault-shards"),("bb-finite-parallel","predictions.npz","decoded-shards")):
        for path in sorted((WORK/"results"/kind).rglob("result.json")):
            record=read_json(path); own=path.parent/basename
            arrays=dict(np.load(own))
            target=WORK/"results/bb"/record["instance"]/record["order"]
            if kind=="bb-finite-parallel": target=target/f"eps{record['eps']}"
            target=target/target_name/f"{record['start']:06d}.npz"
            equal_arrays(target,arrays)
            expected=record.get("artifact_sha256",record.get("predictions_sha256"))
            identity(own,expected);identity(target)
            for name,value in record["inputs"].items():
                identity(WORK/"results/bb"/record["instance"]/record["order"]/name,value)
            counts[kind+"_published_partitions"]+=1
    for path in (WORK/"results/full-model-fixed").glob("*/result.json"):
        r=read_json(path);sf=dict(np.load(HISTORICAL/"results/tiny-hgp"/r["task"]["order"]/"single-fault.npz"))
        arrays=dict(np.load(path.parent/"decoded.npz"))
        for label,row in r["decoder"].items():
            risk=float(sf["physical_coefficients"] @ (arrays[label+"_single_fault_prediction"] ^ sf["observables"]).any(axis=1))
            assert abs(row["R_delta"]-risk)<1e-8
            counts["fixed_prior_exact_risks"]+=1
    registrations=0
    for path in (WORK/"provenance").glob("*/registration.json"):
        r=read_json(path)
        for source,value in r.get("source_sha256",{}).items(): identity(source,value)
        registrations+=1
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,counts=dict(counts),
        registrations=registrations,checked_records=checked,input_sha256=inputs,
        scope="Committed result snapshot only; unfinished tasks excluded; seeds, all logical residuals, counts, physical XOR and publication equality checked")
    commit_json(WORK/"provenance/committed-results-audit.json",result)
    print({k:result[k]for k in ["passed","counts","registrations"]},flush=True)


if __name__=="__main__": run()
