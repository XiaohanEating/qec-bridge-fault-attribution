"""Validate full fixed-message reuse without trusting aggregate failure counts."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import time

import numpy as np
import stim

from ..artifacts import WORK,commit_json,digest
from ..baselines import bb_code
from ..execution.decoder import ValidatedExecutionDecoder,StableExecutionDecoder
from .stored_execution import CompareCalls


def matrix(instance,order):
    folder=WORK/'results/bb'/instance/order
    circuit=stim.Circuit((folder/'prior.stim').read_text(encoding='utf-8'))
    faults=dict(np.load(folder/'single-fault.npz'))
    code=bb_code(instance)
    old=ValidatedExecutionDecoder(code,circuit,single_fault=faults)
    new=StableExecutionDecoder(code,circuit,single_fault=faults)
    rows=[]
    for label in old.states:
        new.states[label]['decoders']=[CompareCalls(a,b,(label,i),rows)
            for i,(a,b)in enumerate(zip(old.states[label]['decoders'],new.states[label]['decoders']))]
    records=[np.zeros(circuit.num_detectors,dtype=np.uint8)]
    records.extend(faults['detectors'][np.linspace(0,len(faults['detectors'])-1,16,dtype=int)])
    for rate in ('eps1e-07','eps1e-06','eps0.0001'):
        with np.load(folder/rate/'physical.npz')as f:records.extend(f['detectors'][:16])
    records=np.asarray(records,dtype=np.uint8)
    actual=new.predictions(records);expected=old.predictions(records)
    assert all(np.array_equal(actual[label],expected[label])for label in expected)
    result=dict(instance=instance,order=order,records=len(records),calls=rows,all_logical_predictions_equal=True,
        source_sha256={str(p):digest(p)for p in (WORK/'src/iscas_revision/execution').glob('*.py')})
    commit_json(WORK/'provenance/stable-execution'/instance/(order+'.json'),result)
    print(dict(instance=instance,order=order,calls=len(rows),old_seconds=sum(r['old_seconds']for r in rows),
        new_seconds=sum(r['new_seconds']for r in rows),passed=True),flush=True)
    return result


def run():
    jobs=[(instance,order)for instance in ('bb-72','bb-108')for order in ('forward','macro_reverse','check_reverse')]
    with ProcessPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(matrix,*job)for job in jobs]
        results=[future.result()for future in futures]
    commit_json(WORK/'provenance/stable-execution-validation.json',dict(matrices=results,
        created_utc=datetime.now(timezone.utc).isoformat(),running_snapshots_modified=False,
        validation_source_sha256=digest(__file__)))


if __name__=='__main__':run()
