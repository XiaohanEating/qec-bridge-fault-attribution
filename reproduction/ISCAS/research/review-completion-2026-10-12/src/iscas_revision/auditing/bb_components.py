"""Assemble BB evidence components only when every relevant block exists."""
from datetime import datetime,timezone
import numpy as np

from ..artifacts import WORK,commit_json,read_json,digest
from ..audit import optimal_table
from ..stratification import clopper_pearson

LABELS=('original','strong','single_fault_lookup')
EPS=(1e-9,1e-8,1e-7,1e-6,3e-6,1e-5,3e-5,1e-4)


def run():
    matrices=[];points=[]
    for instance in ('bb-72','bb-108'):
        for order in ('forward','macro_reverse','check_reverse'):
            folder=WORK/'results/bb'/instance/order
            with np.load(folder/'single-fault.npz')as f:sf={name:f[name]for name in f.files}
            blocks=[folder/'single-fault-shards'/f'{start:06d}.npz'for start in range(0,len(sf['detectors']),256)]
            if all(path.exists()for path in blocks):
                pieces=[dict(np.load(path))for path in blocks]
                prediction={label:np.concatenate([piece[label]for piece in pieces])for label in LABELS}
                table,optimum=optimal_table(sf['detectors'],sf['observables'],sf['physical_coefficients'])
                assert optimum==read_json(folder/'physical-risk.json')['R_star']
                assert all(np.array_equal(prediction['single_fault_lookup'][i],table[bytes(record)])
                    for i,record in enumerate(sf['detectors']))
                risks={}
                for label,pred in prediction.items():
                    assert pred.shape==sf['observables'].shape
                    assert np.all((pred==0)|(pred==1))
                    risk=float(sf['physical_coefficients']@(pred^sf['observables']).any(axis=1))
                    assert risk>=optimum-1e-8
                    risks[label]=dict(R_delta=risk,E=risk-optimum)
                matrices.append(dict(instance=instance,order=order,R_star=optimum,decoder=risks,
                    all_logical_bits=True,physical_source_sha256=digest(folder/'single-fault.npz'),
                    raw_prediction_sha256={str(path):digest(path)for path in blocks}))
            for eps in EPS:
                point=folder/f'eps{eps}'
                blocks=[point/'decoded-shards'/f'{start:06d}.npz'for start in range(0,32768,128)]
                if not all(path.exists()for path in blocks):continue
                with np.load(point/'physical.npz')as f:sample={name:f[name]for name in f.files}
                pieces=[dict(np.load(path))for path in blocks]
                prediction={label:np.concatenate([piece[label]for piece in pieces])for label in LABELS}
                decoder={}
                for label,pred in prediction.items():
                    assert pred.shape==sample['observables'].shape
                    assert np.all((pred==0)|(pred==1))
                    residual=pred^sample['observables']
                    fail=int(residual.any(axis=1).sum())
                    decoder[label]=dict(shots=32768,failures=fail,probability=fail/32768,
                        marginal_interval=clopper_pearson(fail,32768),
                        simultaneous_144_interval=clopper_pearson(fail,32768,alpha=.05/144),
                        per_logical=residual.sum(axis=0).tolist())
                points.append(dict(instance=instance,order=order,eps=eps,seed=int(sample['seed']),
                    decoder=decoder,physical_sample_sha256=digest(point/'physical.npz'),
                    raw_prediction_sha256={str(path):digest(path)for path in blocks}))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),single_fault_matrices=matrices,finite_points=points,
        single_fault_complete=len(matrices)==6,finite_complete=len(points)==48,
        scientific_inputs_modified=False,scope='Only fully published scientific components; partial blocks excluded',
        rate_interval_family=144,rate_simultaneous_confidence_level=.95)
    commit_json(WORK/'results/bb-components.json',result)
    print(dict(single_fault_complete=len(matrices),finite_points_complete=len(points)),flush=True)


if __name__=='__main__':run()
