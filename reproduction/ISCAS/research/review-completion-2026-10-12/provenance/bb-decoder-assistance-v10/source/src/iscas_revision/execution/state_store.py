"""Persist exact QUITS window inputs once; preserve sparse storage and priors."""
from concurrent.futures import ProcessPoolExecutor
import contextlib
from datetime import datetime, timezone
import io
import hashlib
import time

import numpy as np
from scipy import sparse
import stim
from quits.decoder.base import spacetime

from ..artifacts import WORK, digest, read_json, commit_json
from ..baselines import bb_code
from ..decoding import CONFIGURATIONS


def location(instance, order):
    return WORK/"provenance/bb-decoder-states"/instance/order


def save_window_inputs(instance, order):
    began=time.perf_counter()
    source=WORK/"results/bb"/instance/order/"prior.stim"
    folder=location(instance,order)
    manifest=folder/"identity.json"
    if manifest.exists():
        value=read_json(manifest)
        assert value["prior_sha256"]==digest(source)
        assert value["archive_sha256"]==digest(folder/"states.npz")
        if "circuit_text_sha256"not in value:
            commit_json(folder/"identity-before-text-binding.json",value)
            circuit=stim.Circuit(source.read_text(encoding="utf-8"))
            code=bb_code(instance)
            value["circuit_text_sha256"]=hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()
            value["hz_sha256"]=hashlib.sha256(np.asarray(code.hz.toarray() if sparse.issparse(code.hz)else code.hz).tobytes()).hexdigest()
            assert value["num_detectors"]==circuit.num_detectors and value["num_observables"]==circuit.num_observables
            commit_json(manifest,value)
        return value
    folder.mkdir(parents=True,exist_ok=True)
    circuit=stim.Circuit(source.read_text(encoding="utf-8"))
    code=bb_code(instance)
    values={}
    groups={}
    originals={}
    for label in ("original","strong"):
        configuration=CONFIGURATIONS[label]
        window=configuration["window"]
        rounds=circuit.num_detectors//code.hz.shape[0]-2
        count=(2+rounds-1)//window
        with contextlib.redirect_stdout(io.StringIO()):
            arrays=spacetime(circuit,code.hz,window,window,count)
        groups[label]=dict(num_cor_rounds=count,hz_rows=code.hz.shape[0],W=window,F=window,
            configuration=configuration,collections={})
        originals[label]=arrays
        for name,collection in zip(("checks","observations","priors","updates"),arrays):
            specs=[]
            for i,item in enumerate(collection):
                prefix=f"{label}_{name}_{i}"
                if sparse.issparse(item):
                    assert item.format in ("csc","csr")
                    values[prefix+"_data"]=item.data
                    values[prefix+"_indices"]=item.indices
                    values[prefix+"_indptr"]=item.indptr
                    specs.append(dict(prefix=prefix,format=item.format,shape=list(item.shape)))
                else:
                    values[prefix]=np.asarray(item)
                    specs.append(dict(prefix=prefix,format="dense"))
            groups[label]["collections"][name]=specs
    archive=folder/"states.npz"
    temporary=folder/"states.tmp.npz"
    np.savez_compressed(temporary,**values)
    temporary.replace(archive)
    value=dict(instance=instance,order=order,prior_sha256=digest(source),archive_sha256=digest(archive),
        circuit_text_sha256=hashlib.sha256(str(circuit).encode("utf-8")).hexdigest(),
        hz_sha256=hashlib.sha256(np.asarray(code.hz.toarray() if sparse.issparse(code.hz)else code.hz).tobytes()).hexdigest(),
        num_detectors=circuit.num_detectors,num_observables=circuit.num_observables,groups=groups,
        scientific_decoder_sha256=digest(WORK/"src/iscas_revision/decoding.py"),
        generator_source_sha256=digest(__file__),created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_seconds=time.perf_counter()-began)
    commit_json(manifest,value)
    loaded=load_window_inputs(instance,order,value["prior_sha256"])
    for label,arrays in originals.items():
        for name,expected in zip(("checks","obs_set","priors","update"),arrays):
            actual=loaded[label][name]
            assert len(actual)==len(expected)
            for a,b in zip(expected,actual):
                if sparse.issparse(a):
                    assert a.format==b.format and a.shape==b.shape
                    assert np.array_equal(a.data,b.data)
                    assert np.array_equal(a.indices,b.indices)
                    assert np.array_equal(a.indptr,b.indptr)
                else:
                    assert np.array_equal(a,b)
    value["all_window_inputs_storage_exact_equal"]=True
    commit_json(manifest,value)
    print(dict(instance=instance,order=order,seconds=value["preparation_seconds"],exact_equal=True),flush=True)
    return value


def load_window_inputs(instance,order,prior_sha):
    folder=location(instance,order)
    identity=read_json(folder/"identity.json")
    assert identity["prior_sha256"]==prior_sha
    assert identity["scientific_decoder_sha256"]==digest(WORK/"src/iscas_revision/decoding.py")
    assert identity["archive_sha256"]==digest(folder/"states.npz")
    states={}
    with np.load(folder/"states.npz",allow_pickle=False)as archive:
        for label,group in identity["groups"].items():
            assert group["configuration"]==CONFIGURATIONS[label]
            state={name:group[name]for name in ("num_cor_rounds","hz_rows","W","F")}
            for name,specs in group["collections"].items():
                items=[]
                for spec in specs:
                    prefix=spec["prefix"]
                    if spec["format"]=="dense":item=archive[prefix]
                    else:
                        constructor=sparse.csc_matrix if spec["format"]=="csc"else sparse.csr_matrix
                        item=constructor((archive[prefix+"_data"],archive[prefix+"_indices"],archive[prefix+"_indptr"]),shape=tuple(spec["shape"]))
                    items.append(item)
                state[{"observations":"obs_set","updates":"update"}.get(name,name)]=items
            states[label]=state
    return states


def find_window_inputs(code,circuit):
    circuit_sha=hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()
    hz_sha=hashlib.sha256(np.asarray(code.hz.toarray() if sparse.issparse(code.hz)else code.hz).tobytes()).hexdigest()
    for path in (WORK/"provenance/bb-decoder-states").glob("*/*/identity.json"):
        identity=read_json(path)
        if identity.get("circuit_text_sha256")==circuit_sha:
            assert identity["hz_sha256"]==hz_sha
            assert identity["all_window_inputs_storage_exact_equal"]
            assert identity["num_detectors"]==circuit.num_detectors
            assert identity["num_observables"]==circuit.num_observables
            return load_window_inputs(identity["instance"],identity["order"],identity["prior_sha256"])
    return None


def run():
    jobs=[(instance,order)for instance in ("bb-72","bb-108")
        for order in ("forward","macro_reverse","check_reverse")]
    with ProcessPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(save_window_inputs,*job)for job in jobs]
        results=[future.result()for future in futures]
    commit_json(WORK/"provenance/window-input-storage-validation.json",dict(
        matrices=results,all_exact_equal=all(r["all_window_inputs_storage_exact_equal"]for r in results),
        created_utc=datetime.now(timezone.utc).isoformat(),scientific_protocol_unchanged=True))


if __name__=="__main__":run()
