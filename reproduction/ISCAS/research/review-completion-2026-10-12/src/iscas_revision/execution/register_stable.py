"""Register a new helper only after exact fixed-message-state validation."""
import argparse
from datetime import datetime,timezone
import shutil

from ..artifacts import WORK,read_json,commit_json,digest


def register(batch,original,workers,kind):
    target=WORK/'provenance'/batch
    assert not target.exists()
    proofs={name:read_json(WORK/'provenance'/name)for name in ('stable-product-validation.json','stable-execution-validation.json')}
    assert all(len(value['matrices'])==6 for value in proofs.values())
    for m in proofs['stable-execution-validation.json']['matrices']:
        assert m['all_logical_predictions_equal']and all(r['full_error_and_llr_equal']for r in m['calls'])
    assert proofs['stable-product-validation.json']['source_sha256']==digest(WORK/'src/iscas_revision/execution/serial_bp.py')
    prior=read_json(WORK/'provenance'/original/'registration.json')
    target.mkdir()
    shutil.copytree(WORK/'provenance'/original/'source/src',target/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    destination=target/'source/src/iscas_revision'
    for name in ('serial_bp.py','decoder.py'):
        shutil.copy2(WORK/'src/iscas_revision/execution'/name,destination/'execution'/name)
    for filename in ('parallel_decode.py','finite_parallel.py'):
        path=destination/filename
        text=path.read_text(encoding='utf-8')
        assert 'ValidatedExecutionDecoder as CachedDecoder'in text
        path.write_text(text.replace('ValidatedExecutionDecoder as CachedDecoder','StableExecutionDecoder as CachedDecoder'),encoding='utf-8')
    # Avoid 12,288 individual filesystem probes on every fast worker result.
    # Discover published block filenames once per directory; new commits are
    # seen on the next iteration and the same atomic lock resolves any race.
    path=destination/'execution/assistance.py'
    text=path.read_text(encoding='utf-8')
    text=text.replace("for i,eps in enumerate(EPS):\n                for start", "for i,eps in enumerate(EPS):\n                committed_names={p.name for p in (folder/f'eps{eps}'/'decoded-shards').glob('*.npz')}\n                for start")
    text=text.replace("if (folder/f'eps{eps}'/'decoded-shards'/f'{start:06d}.npz').exists():", "if f'{start:06d}.npz' in committed_names:")
    path.write_text(text,encoding='utf-8')
    science={name:digest(destination/name)==digest(WORK/'provenance'/original/'source/src/iscas_revision'/name)
        for name in ('decoding.py','baselines.py','experiments.py','stratification.py')}
    assert all(science.values())
    sources={str(path):digest(path)for path in destination.rglob('*.py')}
    sources.update({path:sha for path,sha in prior['source_sha256'].items()if '/source/'not in path.replace('\\','/')})
    validation={name:dict(path=str(WORK/'provenance'/name),sha256=digest(WORK/'provenance'/name))for name in proofs}
    commit_json(target/'registration.json',dict(prior,registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,workers=workers,kind=kind,scientific_modules_bytes_equal=science,
        stable_state_validation=validation,
        execution_change='Same mathematical maximum50 iterations and native return flags; exact invariant transformed-message state repeats remaining iterations, reuse it; same candidate support arithmetic',
        traversal_change='One directory listing per point, identical jobs and lock/publication protocol',
        complete_scope_retained=True,no_existing_processes_terminated=True))
    print(dict(batch=batch,workers=workers,kind=kind),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--original',required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--kind',required=True)
    a=p.parse_args();register(a.batch,a.original,a.workers,a.kind)
