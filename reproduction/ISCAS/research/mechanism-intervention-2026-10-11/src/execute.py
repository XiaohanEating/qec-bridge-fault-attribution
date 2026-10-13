"""Sequential, resumable supplement; result creation is protected by an exclusive lock."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from support import HOME, ROOT, DECODERS, code_for, decoder_state, predict, sha, save, decode_lone_faults
from construction import prepare, materialize_sources
from physical_risk import risk, verify_injections

EPS = (0.00001, 0.0001, 0.001)
SOURCES = ('all','idle','2q','spam','sq')
SEED_OFFSETS = dict(forward=0, macro_reverse=100, check_reverse=200, kutin=300)

def benchmark(code, circuit):
    started = time.perf_counter()
    syn, obs = circuit.compile_detector_sampler(seed=20261031).sample(64,separate_observables=True)
    sample_seconds = time.perf_counter()-started
    timings = {}
    for label in DECODERS:
        started = time.perf_counter(); state=decoder_state(code,circuit,label)
        predictions = predict(state,syn)
        assert np.array_equal((predictions^obs).any(axis=1),decode_lone_faults(state,syn,obs))
        timings[label] = dict(wall_seconds=time.perf_counter()-started, shots=64,
                              original_recursion_parity=True)
    return dict(sample_seconds=sample_seconds,decoders=timings,
                decision='single process; routing patches are global; sample/decoder benchmark recorded before full runs')

def sample_decode(code, circuit, decoder_circuit, folder, seed, shots):
    result = folder/'sample-result.json'
    expected_sha=__import__('hashlib').sha256(str(circuit).encode()).hexdigest()
    if result.exists():
        record=json.loads(result.read_text(encoding='utf-8'))
        assert record['circuit_sha256']==sha(folder/'circuit.stim') and record['seed']==seed
        actual_circuit=__import__('stim').Circuit((folder/'circuit.stim').read_text(encoding='utf-8'))
        assert __import__('hashlib').sha256(str(actual_circuit).encode()).hexdigest()==expected_sha
        assert record['decoder']['original']['shots']==shots
        assert record['physical_samples_sha256']==sha(folder/'physical-samples.npz')
        assert record['decoded_sha256']==sha(folder/'decoded.npz')
        return record
    folder.mkdir(parents=True,exist_ok=True)
    circuit_path=folder/'circuit.stim'; circuit_path.write_text(str(circuit),encoding='utf-8')
    started=time.perf_counter()
    syn, obs=circuit.compile_detector_sampler(seed=seed).sample(shots,separate_observables=True)
    # Commit physical samples before any decoding; reruns reuse precisely those samples.
    sample_path=folder/'physical-samples.npz'
    if sample_path.exists():
        old=np.load(sample_path); syn,obs=old['detectors'],old['observables']
        assert len(syn)==shots and int(old['seed'])==seed
        assert str(old['circuit_sha256'])==expected_sha
    else: np.savez_compressed(sample_path,detectors=syn,observables=obs,seed=seed,circuit_sha256=expected_sha)
    records,arrays={},{}
    for label in DECODERS:
        state=decoder_state(code,decoder_circuit,label); t=time.perf_counter()
        prediction=predict(state,syn); residual=prediction^obs
        failures=residual.any(axis=1)
        records[label]=dict(shots=shots,failures=int(failures.sum()),p_any=float(failures.mean()),
                            per_logical_failures=residual.sum(axis=0).tolist(),decoder=DECODERS[label],
                            decode_seconds=time.perf_counter()-t)
        arrays[label+'_prediction']=prediction; arrays[label+'_residual']=residual
    np.savez_compressed(folder/'decoded.npz',**arrays)
    original=arrays['original_residual'].any(axis=1); strong=arrays['strong_residual'].any(axis=1)
    record=dict(seed=seed,circuit_sha256=sha(circuit_path),circuit_normalized_sha256=expected_sha,physical_samples_sha256=sha(sample_path),
                decoded_sha256=sha(folder/'decoded.npz'),decoder=records,
                original_only_failure=int((original&~strong).sum()),strong_only_failure=int((strong&~original).sum()),
                total_seconds=time.perf_counter()-started)
    registration=folder.parents[2]/'registered-prediction.json'
    if registration.exists():record['registered_prediction_sha256']=sha(registration)
    save(result,record)
    return record

def run(name, mode, shots):
    folder=HOME/'results'/name
    code=code_for(name)
    if not (folder/'construction.json').exists(): plans=prepare(name)
    else:
        plans={label:json.loads((folder/label/'plan.json').read_text(encoding='utf-8'))
               for label in ('forward','macro_reverse','check_reverse','kutin')}
    original=(folder/'original.stim')
    manifest=HOME/'provenance'/'run-source-manifest-v4.json'
    import quits.decoder.base, quits.decoder.sliding_window, decoder_aware, pl_validation, routing, joint_router
    files=list((HOME/'src').glob('*.py'))+[Path(module.__file__) for module in
            (quits.decoder.base,quits.decoder.sliding_window,decoder_aware,pl_validation,routing,joint_router)]
    current={str(path):sha(path) for path in files}
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        assert saved['sources']==current, 'Running source changed; new protocol version required'
        assert saved['protocol_sha256']==sha(HOME/'provenance/model-and-protocol.md')
    else:
        save(manifest,dict(python=sys.executable,version=sys.version,sources=current,
                          packages={p:importlib.metadata.version(p) for p in ('stim','numpy','scipy','ldpc','quits')},
                          protocol_sha256=sha(HOME/'provenance/model-and-protocol.md')))
    if not (folder/'benchmark.json').exists():
        circuit,_=materialize_sources(plans['forward'],.001)
        save(folder/'benchmark.json',benchmark(code,circuit))
    risks={}
    for label,plan in plans.items():
        target=folder/label
        if (target/'risk.json').exists(): out=json.loads((target/'risk.json').read_text(encoding='utf-8'))
        else:
            circuit,_=materialize_sources(plan,.001)
            out=risk(code,circuit,.001,target)
            verify_injections(circuit,target)
        risks[label]=out
        print(json.dumps(dict(stage='risk',instance=name,label=label,coefficient=out['decoder']['original']['coefficient'],oracle=out['optimal_single_fault_coefficient'])),flush=True)
    save(folder/'risk-summary.json',risks)
    if mode=='risk': return
    if name != 'tiny-hgp' and not (folder/'registered-prediction.json').exists():
        training=json.loads((HOME/'results/tiny-hgp/risk-summary.json').read_text(encoding='utf-8'))
        selected=min(('forward','macro_reverse','check_reverse'),key=lambda key:risks[key]['decoder']['original']['coefficient'])
        reference=risks['forward']['decoder']['original']['coefficient']
        coefficient=risks[selected]['decoder']['original']['coefficient']
        save(folder/'registered-prediction.json',dict(rule='min physical first-order original decoder risk among same-budget variants; no shot outcomes used',
               training_sha256=sha(HOME/'results/tiny-hgp/risk-summary.json'),original_sha256=sha(original),
               selected=selected,baseline='forward',risk_selected=coefficient,risk_baseline=reference,
               prediction='improve' if coefficient<reference else 'unknown',seed_base=20261101,seed_offsets=SEED_OFFSETS,shots=shots,
               allowed_noise=EPS,decision='95% difference CI; opposite significant result=failure, otherwise unknown',
               primary_source='all',primary_eps=.0001,primary_decoder='original',
               primary_registration_sha256=sha(HOME/'provenance/holdout-primary-registration.md'),
               registered_before_sampling=True))
    records=[]
    for eps in EPS:
        for source in SOURCES:
            for label,plan in plans.items():
                circuit,_=materialize_sources(plan,eps,source)
                decoder_circuit,_=materialize_sources(plan,.001,'all')
                target=folder/label/'samples-v2'/f'{source}-eps{eps}-n{shots}'
                seed=(20261031 if name=='tiny-hgp' else 20261101)+SEED_OFFSETS[label]+1000*SOURCES.index(source)+10000*EPS.index(eps)
                record=sample_decode(code,circuit,decoder_circuit,target,seed,shots)
                records.append(dict(instance=name,label=label,eps=eps,source=source,**record))
                print(json.dumps(dict(stage='samples',instance=name,label=label,eps=eps,source=source,
                                      original=record['decoder']['original']['failures'],strong=record['decoder']['strong']['failures'],seconds=record['total_seconds'])),flush=True)
    save(folder/'sample-summary.json',records)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=['tiny-hgp','holdout-hgp-3x4'])
    parser.add_argument('--mode',choices=['risk','full'],default='full');parser.add_argument('--shots',type=int,default=4096)
    args=parser.parse_args()
    lock=HOME/'provenance'/'execution.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        run(args.name,args.mode,args.shots)
    finally:lock.unlink(missing_ok=True)
