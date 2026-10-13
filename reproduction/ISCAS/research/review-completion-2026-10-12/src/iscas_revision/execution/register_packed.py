"""Freeze packed GF(2) execution only after full-vector equivalence checks."""
import argparse
from datetime import datetime,timezone
import shutil

from ..artifacts import WORK,read_json,commit_json,digest


def register(batch,original,workers,kind):
    target=WORK/'provenance'/batch
    assert not target.exists()
    proofs={name:read_json(WORK/'provenance'/name)for name in ('packed-basis-validation.json','packed-execution-validation.json')}
    assert all(len(value['matrices'])==6 for value in proofs.values())
    for m in proofs['packed-execution-validation.json']['matrices']:
        assert m['all_logical_predictions_equal']and all(r['full_error_and_llr_equal']for r in m['calls'])
    assert proofs['packed-basis-validation.json']['source_sha256']==digest(WORK/'src/iscas_revision/execution/linear_osd.py')
    prior=read_json(WORK/'provenance'/original/'registration.json')
    target.mkdir()
    shutil.copytree(WORK/'provenance'/original/'source/src',target/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    destination=target/'source/src/iscas_revision'
    for name in ('serial_bp.py','decoder.py','linear_osd.py'):
        shutil.copy2(WORK/'src/iscas_revision/execution'/name,destination/'execution'/name)
    for filename in ('parallel_decode.py','finite_parallel.py'):
        path=destination/filename
        text=path.read_text(encoding='utf-8')
        if 'StableExecutionDecoder as CachedDecoder'in text:
            text=text.replace('StableExecutionDecoder as CachedDecoder','PackedExecutionDecoder as CachedDecoder')
        else:
            assert 'ValidatedExecutionDecoder as CachedDecoder'in text
            text=text.replace('ValidatedExecutionDecoder as CachedDecoder','PackedExecutionDecoder as CachedDecoder')
        path.write_text(text,encoding='utf-8')
    science={name:digest(destination/name)==digest(WORK/'provenance'/original/'source/src/iscas_revision'/name)
        for name in ('decoding.py','baselines.py','experiments.py','stratification.py')}
    assert all(science.values())
    sources={str(path):digest(path)for path in destination.rglob('*.py')}
    sources.update({path:sha for path,sha in prior['source_sha256'].items()if '/source/'not in path.replace('\\','/')})
    validation={name:dict(path=str(WORK/'provenance'/name),sha256=digest(WORK/'provenance'/name))for name in proofs}
    commit_json(target/'registration.json',dict(prior,registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,workers=workers,kind=kind,scientific_modules_bytes_equal=science,
        packed_basis_validation=validation,
        execution_change='Identical GF2 row swaps/XOR, native ordered columns, OSD0/final candidates and ascending support costs; binary row storage packed into64-bit words',
        complete_scope_retained=True,no_existing_processes_terminated=True))
    print(dict(batch=batch,workers=workers,kind=kind),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--original',required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--kind',required=True)
    a=p.parse_args();register(a.batch,a.original,a.workers,a.kind)
