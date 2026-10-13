"""Check exact binary elimination with unchanged column order and scoring."""
from datetime import datetime,timezone
import contextlib,io,time
import numpy as np
import stim
from quits.decoder.base import spacetime

from ..artifacts import WORK,commit_json,digest
from ..baselines import bb_code
from ..execution.linear_osd import linear_sweep,packed_linear_sweep,LinearOsd,packed_rows


def run():
    rng=np.random.default_rng(99731012)
    for trial in range(240):
        nr=int(rng.integers(1,30));nc=int(rng.integers(nr,200))
        check=rng.integers(2,size=(nr,nc),dtype=np.uint8)
        error=rng.integers(2,size=nc,dtype=np.uint8);syndrome=check@error%2
        llr=rng.choice(np.array([-2.,0.,.1,1.,3.]),size=nc)
        costs=rng.choice(np.array([0.,.1,1.,3.7,10.]),size=nc)
        order=int(rng.integers(4))
        a=linear_sweep(check,syndrome,llr,costs,order,sparse_support=True)
        b=packed_linear_sweep(check,syndrome,llr,costs,order)
        assert all(np.array_equal(x,y)for x,y in zip(a,b))
    matrices=[]
    for instance in ('bb-72','bb-108'):
        for order in ('forward','macro_reverse','check_reverse'):
            folder=WORK/'results/bb'/instance/order
            code=bb_code(instance);circuit=stim.Circuit((folder/'prior.stim').read_text(encoding='utf-8'))
            with contextlib.redirect_stdout(io.StringIO()):checks,_,priors,_=spacetime(circuit,code.hz,4,4,1)
            decoder=LinearOsd(checks[0],priors[0],exact_bp=True,prefix_reuse=True,zero_shortcut=True)
            decoder.bp.stable_reuse=True;packed=packed_rows(decoder.check)
            rows=[]
            for rate in ('eps1e-07','eps1e-06','eps0.0001'):
                with np.load(folder/rate/'physical.npz')as f:records=f['detectors'][:12,:checks[0].shape[0]]
                for record in records:
                    decoder.bp.decode(record)
                    t=time.perf_counter();a=linear_sweep(decoder.check,record,decoder.bp.log_prob_ratios,decoder.costs,2,sparse_support=True);old=time.perf_counter()-t
                    t=time.perf_counter();b=packed_linear_sweep(decoder.check,record,decoder.bp.log_prob_ratios,decoder.costs,2,packed);new=time.perf_counter()-t
                    assert all(np.array_equal(x,y)for x,y in zip(a,b))
                    rows.append(dict(rate=rate,plain_seconds=old,packed_seconds=new,osd0_and_final_equal=True))
            matrices.append(dict(instance=instance,order=order,records=rows))
            commit_json(WORK/'provenance/packed-basis-validation-progress.json',dict(completed=len(matrices),total=6,matrices=matrices))
            print(dict(instance=instance,order=order,plain_seconds=sum(r['plain_seconds']for r in rows),packed_seconds=sum(r['packed_seconds']for r in rows)),flush=True)
    commit_json(WORK/'provenance/packed-basis-validation.json',dict(matrices=matrices,randomized_cases=240,
        updated_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(WORK/'src/iscas_revision/execution/linear_osd.py'),
        running_snapshots_modified=False))


if __name__=='__main__':run()
