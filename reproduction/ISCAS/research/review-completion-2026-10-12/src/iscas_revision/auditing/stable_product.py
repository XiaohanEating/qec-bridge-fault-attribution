"""Check exact fixed transformed-message reuse across six BB matrices."""
from datetime import datetime,timezone
import contextlib,io,time
import numpy as np
import stim
from quits.decoder.base import spacetime

from ..artifacts import WORK,commit_json,digest
from ..baselines import bb_code
from ..execution.serial_bp import ExactSerialBp


def run():
    matrices=[]
    for instance in ('bb-72','bb-108'):
        for order in ('forward','macro_reverse','check_reverse'):
            folder=WORK/'results/bb'/instance/order
            code=bb_code(instance);circuit=stim.Circuit((folder/'prior.stim').read_text(encoding='utf-8'))
            with contextlib.redirect_stdout(io.StringIO()):checks,_,priors,_=spacetime(circuit,code.hz,4,4,1)
            rows=[]
            first=ExactSerialBp(checks[0],priors[0],50,prefix_reuse=True,zero_shortcut=True)
            second=ExactSerialBp(checks[0],priors[0],50,prefix_reuse=True,zero_shortcut=True,stable_reuse=True)
            for rate in ('eps1e-07','eps1e-06','eps0.0001'):
                with np.load(folder/rate/'physical.npz')as f:records=f['detectors'][:12,:checks[0].shape[0]]
                for record in records:
                    t=time.perf_counter();a=first.decode(record);old=time.perf_counter()-t
                    t=time.perf_counter();b=second.decode(record);new=time.perf_counter()-t
                    assert np.array_equal(a,b)
                    assert np.array_equal(first.log_prob_ratios,second.log_prob_ratios,equal_nan=True)
                    assert first.converge==second.converge and first.iter==second.iter
                    rows.append(dict(rate=rate,full_seconds=old,stable_seconds=new,exact_equal=True))
            matrices.append(dict(instance=instance,order=order,records=rows))
            commit_json(WORK/'provenance/stable-product-validation-progress.json',dict(completed=len(matrices),total=6,matrices=matrices))
            print(dict(instance=instance,order=order,full_seconds=sum(r['full_seconds']for r in rows),stable_seconds=sum(r['stable_seconds']for r in rows)),flush=True)
    commit_json(WORK/'provenance/stable-product-validation.json',dict(matrices=matrices,updated_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(WORK/'src/iscas_revision/execution/serial_bp.py'),running_snapshots_modified=False))


if __name__=='__main__':run()
