"""Replay fixed first 64 physical samples of each historical nine-logical circuit."""
import time
import json
import sys
from pathlib import Path
import numpy as np
from support import HOME, ROOT, sha, save, stim, decoder_state, predict, decode_lone_faults, DECODERS
from baseline_replay import build_code

def replay():
    code=build_code('HGP','hgp-c1-a',ROOT/'shared/C1/corpus.json')
    prior=ROOT/'ISCAS/research/conflict-fault-routing-2026-10-11/fixed-original/data/raw'
    sources={'bridge':'hgp-c1-a-bridge-eps0.001-n256-seed20261023',
             'kutin':'hgp-c1-a-kutin-eps0.001-n256-seed20261024'}
    for label,source in sources.items():
        folder=HOME/'results'/'multilogical'/label
        folder.mkdir(parents=True,exist_ok=True)
        circuit_path=prior/source/'circuit.stim'
        sample_path=prior/source/'shard-000000000.npz'
        circuit=stim.Circuit(circuit_path.read_text(encoding='utf-8'))
        data=np.load(sample_path)
        syn=np.unpackbits(data['detectors'],axis=1,bitorder='little')[:64,:int(data['num_detectors'])]
        obs=np.unpackbits(data['observables'],axis=1,bitorder='little')[:64,:int(data['num_observables'])]
        reference=np.unpackbits(data['fail'],bitorder='little')[:64]
        assert obs.shape==(64,9)
        registration=dict(source_circuit=str(circuit_path),source_circuit_sha256=sha(circuit_path),
                          source_physical_samples=str(sample_path),source_samples_sha256=sha(sample_path),
                          rows='first64 fixed before decoding',shots=64,observables=9,decoder=DECODERS,
                          purpose='same physical samples diagnose per-logical joint and strong decoder; no independent reliability replication')
        save(folder/'registration.json',registration)
        arrays=dict(detectors=syn,observables=obs,original_reference_fail=reference)
        records={}
        for decoder in DECODERS:
            final=folder/(decoder+'-result.json')
            if final.exists():
                records[decoder]=json.loads(final.read_text(encoding='utf-8'))
                continue
            started=time.perf_counter()
            state=decoder_state(code,circuit,decoder)
            setup_seconds=time.perf_counter()-started
            t=time.perf_counter()
            pilot=predict(state,syn[:1])
            pilot_seconds=time.perf_counter()-t
            save(folder/(decoder+'-benchmark.json'),dict(setup_seconds=setup_seconds,one_sample_decode_seconds=pilot_seconds,
                                                       registered_shots=64,configuration=DECODERS[decoder]))
            print(json.dumps(dict(label=label,decoder=decoder,stage='benchmark',setup_seconds=setup_seconds,one_sample_seconds=pilot_seconds)),flush=True)
            t=time.perf_counter()
            prediction=predict(state,syn)
            residual=prediction^obs
            failure=residual.any(axis=1)
            assert np.array_equal(failure[:1],decode_lone_faults(state,syn[:1],obs[:1]))
            if decoder=='original':assert np.array_equal(failure,reference)
            np.savez_compressed(folder/(decoder+'-decoded.npz'),prediction=prediction,residual=residual,
                                detectors=syn,observables=obs,reference_fail=reference)
            records[decoder]=dict(shots=64,failures=int(failure.sum()),per_logical_failures=residual.sum(axis=0).tolist(),
                                  empirical_joint_distribution={str(tuple(row)):int(count) for row,count in
                                    zip(*np.unique(residual,axis=0,return_counts=True))},
                                  source_registration_sha256=sha(folder/'registration.json'),
                                  decoded_sha256=sha(folder/(decoder+'-decoded.npz')),
                                  wall_seconds=time.perf_counter()-t,
                                  original_reference_equal=decoder=='original')
            save(final,records[decoder]);print(json.dumps(dict(label=label,decoder=decoder,stage='complete',**records[decoder])),flush=True)
        save(folder/'summary.json',records)

if __name__=='__main__':replay()
