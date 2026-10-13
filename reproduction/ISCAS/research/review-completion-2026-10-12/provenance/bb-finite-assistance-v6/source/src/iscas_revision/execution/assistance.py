"""Coordinate exact BB assistance without terminating an existing worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import os
import time

from ..artifacts import WORK, read_json, commit_json, digest
from ..parallel_decode import decode_partition
from ..finite_parallel import partition, EPS


def jobs(tasks,kind):
    available=[]
    target=committed=0
    for task in tasks:
        instance,order=task['instance'],task['order']
        folder=WORK/'results/bb'/instance/order
        if kind=='single':
            count=read_json(folder/'physical-risk.json')['signature_count']
            for start in range(0,count,256):
                target+=1
                if (folder/'single-fault-shards'/f'{start:06d}.npz').exists():
                    committed+=1;continue
                lock=WORK/'results/bb-parallel'/instance/order/f'{start:06d}'/'worker.lock'
                if not lock.exists():available.append((instance,order,start,min(count,start+256)))
        else:
            for i,eps in enumerate(EPS):
                for start in range(0,task['shots'],128):
                    target+=1
                    if (folder/f'eps{eps}'/'decoded-shards'/f'{start:06d}.npz').exists():
                        committed+=1;continue
                    lock=WORK/'results/bb-finite-parallel'/instance/order/f'eps{eps}'/f'{start:06d}'/'worker.lock'
                    if not lock.exists():available.append((instance,order,eps,start,task['seed']+i*100))
    # Existing v4 traverses smallest rates/rows first. Assistance traverses from
    # the other end, and both use the same atomic per-partition lock and files.
    if kind=='single':available.sort(key=lambda j:(-j[2],j[0],j[1]))
    else:available.sort(key=lambda j:(-j[2],j[0],j[1],-j[3]))
    return available,committed,target


def run(kind,workers):
    batch=os.environ['ISCAS_REVIEW_BATCH']
    folder=WORK/'provenance'/batch
    registration=read_json(folder/'registration.json')
    assert all(digest(path)==sha for path,sha in registration['source_sha256'].items())
    lock=folder/'supervisor.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    failures=[];contended=[];pending={};began=time.perf_counter()
    operation=decode_partition if kind=='single'else partition
    try:
        with ProcessPoolExecutor(max_workers=workers)as pool:
            while True:
                available,committed,target=jobs(registration['tasks'],kind)
                owned=set(pending.values())
                failed_jobs={tuple(item['job'])for item in failures}
                available=[job for job in available if job not in owned and job not in failed_jobs]
                for job in available[:max(0,workers-len(pending))]:pending[pool.submit(operation,job)]=job
                if not pending and committed==target:break
                if not pending and failures:break
                if pending:
                    done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=pending.pop(future)
                        try:future.result()
                        except FileExistsError:
                            # Another supervisor won the atomic claim. Its
                            # committed result supplies this task exactly once.
                            contended.append(dict(job=job,time=datetime.now(timezone.utc).isoformat()))
                        except Exception as error:
                            import traceback
                            failures.append(dict(job=job,error=str(error),traceback=traceback.format_exc()))
                else:time.sleep(20)
                commit_json(WORK/'results'/(batch+'-status.json'),dict(status='running',kind=kind,
                    workers=workers,pid=os.getpid(),committed_blocks=committed,target_blocks=target,
                    active=list(pending.values()),failed=failures,lock_contention=contended,
                    elapsed_seconds=time.perf_counter()-began,updated_utc=datetime.now(timezone.utc).isoformat()))
        _,committed,target=jobs(registration['tasks'],kind)
        commit_json(WORK/'results'/(batch+'-status.json'),dict(status='complete'if committed==target else'requires repair',
            kind=kind,committed_blocks=committed,target_blocks=target,failed=failures,lock_contention=contended,
            elapsed_seconds=time.perf_counter()-began,updated_utc=datetime.now(timezone.utc).isoformat()))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('single','finite'),required=True)
    p.add_argument('--workers',type=int,required=True);a=p.parse_args();run(a.kind,a.workers)
