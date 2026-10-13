"""Check full window decisions after loading persisted, byte-identical inputs."""
from datetime import datetime, timezone
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import stim

from ..artifacts import WORK, read_json, digest, commit_json
from ..baselines import bb_code
from ..execution.decoder import ExecutionDecoder, ValidatedExecutionDecoder


class CompareCalls:
    def __init__(self,old,new,label,rows):
        self.old,self.new,self.label,self.rows=old,new,label,rows
    def decode(self,syndrome):
        t=time.perf_counter();expected=self.old.decode(syndrome).copy();old_seconds=time.perf_counter()-t
        t=time.perf_counter();actual=self.new.decode(syndrome).copy();new_seconds=time.perf_counter()-t
        assert np.array_equal(expected,actual)
        old_bp=getattr(self.old,"bp",self.old)
        assert old_bp.converge==self.new.bp.converge
        if hasattr(old_bp,"iter"):
            assert old_bp.iter==self.new.bp.iter
        if syndrome.any():
            assert np.array_equal(old_bp.log_prob_ratios,self.new.bp.log_prob_ratios,equal_nan=True)
        if not old_bp.converge:
            assert np.array_equal(self.old.osd0_decoding,self.new.osd0_decoding)
            assert np.array_equal(self.old.osdw_decoding,self.new.osdw_decoding)
        self.rows.append(dict(label=self.label,old_seconds=old_seconds,new_seconds=new_seconds,full_error_and_llr_equal=True))
        return actual


def matrix(instance,order):
    folder=WORK/"results/bb"/instance/order
    circuit=stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
    faults=dict(np.load(folder/"single-fault.npz"))
    code=bb_code(instance)
    old=ExecutionDecoder(code,circuit,single_fault=faults)
    new=ValidatedExecutionDecoder(code,circuit,single_fault=faults)
    rows=[]
    for label in old.states:
        a,b=old.states[label],new.states[label]
        for key in ("num_cor_rounds","hz_rows","W","F"):assert a[key]==b[key]
        for key in ("obs_set","update"):
            assert len(a[key])==len(b[key])
            for x,y in zip(a[key],b[key]):assert x.shape==y.shape and (x!=y).nnz==0
        new.states[label]["decoders"]=[CompareCalls(x,y,(label,i),rows)for i,(x,y)in enumerate(zip(a['decoders'],b['decoders']))]
    records=[np.zeros(circuit.num_detectors,dtype=np.uint8)]
    records.extend(faults["detectors"][np.linspace(0,len(faults["detectors"])-1,5,dtype=int)])
    for point in ("eps1e-07","eps1e-06","eps0.0001"):
        with np.load(folder/point/"physical.npz")as f:records.extend(f["detectors"][:3])
    records=np.asarray(records,dtype=np.uint8)
    actual=new.predictions(records)
    expected=old.predictions(records)
    assert all(np.array_equal(actual[label],expected[label])for label in expected)
    output=dict(instance=instance,order=order,records=len(records),calls=rows,
        all_logical_predictions_equal=True,full_storage_used=True,
        source_sha256={str(p):digest(p)for p in (WORK/"src/iscas_revision/execution").glob("*.py")})
    commit_json(WORK/"provenance/stored-execution"/instance/(order+".json"),output)
    print(dict(instance=instance,order=order,calls=len(rows),passed=True),flush=True)
    return output


def run():
    jobs=[(instance,order)for instance in ("bb-72","bb-108")for order in ("forward","macro_reverse","check_reverse")]
    with ProcessPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(matrix,*job)for job in jobs]
        results=[future.result()for future in futures]
    commit_json(WORK/"provenance/stored-execution-validation.json",dict(matrices=results,
        created_utc=datetime.now(timezone.utc).isoformat(),running_snapshots_modified=False,
        validation_source_sha256=digest(__file__)))


if __name__=="__main__":run()
