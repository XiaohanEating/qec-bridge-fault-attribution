"""Help existing immutable tasks without stealing their exclusive locks."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import os
import time

from .artifacts import WORK, read_json, digest, commit_json
from .experiments import dispatch


def run(workers):
    batch=os.environ["ISCAS_REVIEW_BATCH"]
    registration=read_json(WORK/"provenance"/batch/"registration.json")
    assert all(digest(path)==value for path,value in registration["source_sha256"].items())
    tasks=registration["tasks"]
    lock=WORK/"provenance"/batch/"supervisor.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    failures=[];contended=[];started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers)as pool:
            pending={}
            while True:
                remaining=[task for task in tasks if not (WORK/"results/tasks"/(task["id"]+".json")).exists()]
                owned={task["id"]for task in pending.values()}
                available=[task for task in remaining if task["id"]not in owned
                    and not (WORK/"results/locks"/(task["id"]+".lock")).exists()
                    and task["id"]not in {r["id"]for r in failures}]
                for task in available[:max(0,workers-len(pending))]:
                    pending[pool.submit(dispatch,task)]=task
                if not remaining: break
                if not pending and failures: break
                if pending:
                    done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                    for future in done:
                        task=pending.pop(future)
                        try: print(future.result(),flush=True)
                        except FileExistsError:
                            contended.append(dict(id=task["id"],kind="exclusive task claimed by existing runner"))
                        except Exception as error:
                            failures.append(dict(id=task["id"],error=str(error)))
                else: time.sleep(20)
                commit_json(WORK/"results"/(batch+"-status.json"),dict(status="running",pid=os.getpid(),
                    workers=workers,total_tasks=len(tasks),completed_tasks=len(tasks)-len(remaining),
                    active=[task["id"]for task in pending.values()],failed=failures,lock_contention=contended,
                    elapsed_seconds=time.perf_counter()-started,updated_utc=datetime.now(timezone.utc).isoformat()))
        complete=sum((WORK/"results/tasks"/(task["id"]+".json")).exists()for task in tasks)
        commit_json(WORK/"results"/(batch+"-status.json"),dict(status="complete"if complete==len(tasks)else"requires repair",
            completed_tasks=complete,total_tasks=len(tasks),failed=failures,lock_contention=contended,
            elapsed_seconds=time.perf_counter()-started))
    finally: lock.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4)
    run(parser.parse_args().workers)
