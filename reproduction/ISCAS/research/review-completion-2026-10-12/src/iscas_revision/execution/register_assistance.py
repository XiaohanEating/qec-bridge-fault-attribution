"""Freeze a lock-coordinated helper using verified v5 numerical execution."""
import argparse
from datetime import datetime,timezone
import shutil

from ..artifacts import WORK,read_json,commit_json,digest


def register(batch,original,workers,kind):
    target=WORK/'provenance'/batch
    assert not target.exists()
    target.mkdir()
    prior=read_json(WORK/'provenance'/original/'registration.json')
    tasks=prior.get('tasks')
    if tasks is None:
        tasks=read_json(WORK/'provenance/bb-assembly-v2/registration.json')['tasks']
    assert len(tasks)==6 and all(task['kind']=='bb'for task in tasks)
    shutil.copytree(WORK/'provenance'/original/'source/src',target/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    destination=target/'source/src/iscas_revision'
    shutil.copy2(WORK/'src/iscas_revision/execution/assistance.py',destination/'execution/assistance.py')
    sources={str(path):digest(path)for path in destination.rglob('*.py')}
    sources.update({path:sha for path,sha in prior['source_sha256'].items()if '/source/'not in path.replace('\\','/')})
    for name in ('decoding.py','baselines.py','experiments.py','stratification.py'):
        assert digest(destination/name)==digest(WORK/'provenance'/original/'source/src/iscas_revision'/name)
    commit_json(target/'registration.json',dict(prior,registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,workers=workers,kind=kind,tasks=tasks,
        execution_change='Coexist with original workers using identical atomic per-partition locks; reverse traversal only; all registered samples retained',
        purpose='Exact validated decisions; no process termination or overwrite',
        preserved_existing_batches=['bb-decoder-parallel-v4','bb-finite-parallel-v4'],
        source_v5_registered_but_never_launched=True))
    print(dict(batch=batch,workers=workers,kind=kind),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--original',required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--kind',required=True)
    a=p.parse_args();register(a.batch,a.original,a.workers,a.kind)
