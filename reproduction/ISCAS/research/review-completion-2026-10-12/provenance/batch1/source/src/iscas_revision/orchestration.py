"""Register immutable batches and run exclusive, resumable process tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time
from scipy.stats import binom
from .artifacts import ROOT, WORK, HISTORICAL, commit_json, read_json, digest
from .experiments import EPS, dispatch, integrate_results


def tasks_ready():
    tasks=[]
    # Published circuits give a quick distance-preserving and hook-prone control.
    for distance in (3,5):
        for order in ("normal","normal_hook_proned","ZX_interleaved_Z2Z_X2X","alternating"):
            for eps in (.0001,.0002,.0005,.001,.01):
                tasks.append(dict(kind="hirai",distance=distance,order=order,eps=eps,
                                  shots=1048576 if eps <= .0005 else 131072))
    routes=[(instance,order) for instance in ("tiny-hgp","holdout-hgp-3x4")
            for order in ("forward","macro_reverse","check_reverse")]
    maxima={}
    for instance,order in routes:
        n=int(read_json(HISTORICAL/"results"/instance/order/"risk.json")["total_physical_coefficient"])
        maxima[(instance,order)]=int(binom.ppf(1-1e-8,n,max(EPS)))
    # Interleave orders at low k before progressing to the high-fault strata.
    for k in range(2,max(maxima.values())+1):
        for route,(instance,order) in enumerate(routes):
            if k <= maxima[(instance,order)]:
                tasks.append(dict(kind="stratum",instance=instance,order=order,k=k,
                                  shots=32768,shard_shots=512,seed=120261012+route*1000000+k*1000))
        if k==2:
            for instance,order in routes:
                for prior in ("fixed","matched"):
                    tasks.append(dict(kind="onefactor",instance=instance,order=order,prior=prior))
    for instance,order in routes:
        for eps in EPS:
            tasks.append(dict(kind="direct",instance=instance,order=order,eps=eps,shots=32768))
    for order in ("forward","macro_reverse","check_reverse"):
        for rho in (0.,.1,1.):
            for eps in (1e-5,1e-4):
                tasks.append(dict(kind="sensitivity",instance="tiny-hgp",order=order,rho=rho,eps=eps,shots=32768))
    for order in ("dynamic646","region1","kutin","forward","macro_reverse","check_reverse"):
        for eps in (1e-6,1e-5,1e-4,.001):
            tasks.append(dict(kind="deployment",order=order,eps=eps,shots=32768))
    for index,task in enumerate(tasks):
        task["id"]=f'batch1-{index:04d}-{task["kind"]}'
        if "seed" not in task: task["seed"]=520261012+index*1000
    return tasks


def register():
    assert read_json(WORK/"provenance/physical-verification.json")["passed"]
    snapshot=WORK/"provenance/batch1/source/src/iscas_revision"
    if snapshot.exists(): raise FileExistsError("Batch source already registered; use run to resume")
    snapshot.mkdir(parents=True)
    for path in (WORK/"src/iscas_revision").glob("*.py"):
        shutil.copy2(path,snapshot/path.name)
    sources={str(path):digest(path) for path in snapshot.glob("*.py")}
    frozen_dependencies=[ROOT/"ISCAS/research/mechanism-intervention-2026-10-11/src"/name
                         for name in ("support.py","construction.py","physical_risk.py")]
    frozen_dependencies.extend([ROOT/"ISCAS/research/parallel-routing-2026-10-10/src/routing.py",
                                ROOT/"ISCAS/research/conflict-fault-routing-2026-10-11/src/joint_router.py"])
    sources.update({str(path):digest(path) for path in frozen_dependencies})
    tasks=tasks_ready()
    strata=[task for task in tasks if task["kind"]=="stratum"]
    record=dict(registered_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,python_version=sys.version,
                packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","stim","ldpc","quits","pymatching","bposd","pyscipopt")},
                source_sha256=sources,protocol_sha256=digest(WORK/"protocol.md"),tasks=tasks,
                conditional_shots=sum(t["shots"] for t in strata),conditional_strata=len(strata),
                raw_confirmation_shots=sum(t.get("shots",0) for t in tasks if t["kind"] not in {"stratum","onefactor"}),
                pending_full_scope=["BB-72 and BB-108 construction, risks and finite-noise",
                                   "Zhao official routing, original and common-noise controls",
                                   "standard nearest-neighbor surface-code SD6 d=3,5",
                                   "source-specific sensitivity risk reweighting and completed paper revision"],
                stopping="fixed registered samples; unfinished requirements retained explicitly",
                serial_benchmark=read_json(WORK/"provenance/conditional-benchmark.json"))
    commit_json(WORK/"provenance/batch1/registration.json",record)
    print({"registered_tasks":len(tasks),"conditional_strata":len(strata),"conditional_shots":record["conditional_shots"],
           "snapshot_pythonpath":str(snapshot.parent)},flush=True)


def run(workers):
    record=read_json(WORK/"provenance/batch1/registration.json")
    assert sys.executable.casefold()==record["python"].casefold()
    assert all(digest(path)==sha for path,sha in record["source_sha256"].items())
    tasks=record["tasks"]
    for task in tasks:
        lock=WORK/"results/locks"/(task["id"]+".lock")
        if lock.exists():
            # Never steal a lock from an active process, including another batch runner.
            pid=int(lock.read_text())
            import subprocess
            alive=subprocess.run(["tasklist","/FI",f"PID eq {pid}","/NH"],capture_output=True,text=True).stdout
            if str(pid) not in alive: lock.unlink()
    unfinished=[task for task in tasks if not (WORK/"results/tasks"/(task["id"]+".json")).exists()]
    runner_lock=WORK/"provenance/batch1/supervisor.lock"
    fd=os.open(runner_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.perf_counter()
    failed=[]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator=iter(unfinished); pending={}
            for _ in range(workers):
                task=next(iterator,None)
                if task is not None: pending[pool.submit(dispatch,task)]=task
            completed=len(tasks)-len(unfinished)
            while pending:
                done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    task=pending.pop(future)
                    try:
                        result=future.result(); completed+=1
                        print(json.dumps(result,ensure_ascii=False),flush=True)
                    except Exception as error:
                        failed.append(dict(id=task["id"],error=str(error)))
                        print(json.dumps(dict(id=task["id"],status="failed",error=str(error)),ensure_ascii=False),flush=True)
                    task=next(iterator,None)
                    if task is not None: pending[pool.submit(dispatch,task)]=task
                commit_json(WORK/"results/batch1-status.json",dict(status="running",pid=os.getpid(),workers=workers,
                    completed_tasks=completed,total_tasks=len(tasks),failed=failed,
                    active=[task["id"] for task in pending.values()],elapsed_seconds=time.perf_counter()-started,
                    updated_utc=datetime.now(timezone.utc).isoformat(),full_scope_remaining=record["pending_full_scope"]))
                integrate_results()
        status="complete" if not failed else "requires repair of failed tasks"
        commit_json(WORK/"results/batch1-status.json",dict(status=status,completed_tasks=completed,total_tasks=len(tasks),failed=failed,
                     full_scope_remaining=record["pending_full_scope"],elapsed_seconds=time.perf_counter()-started))
    finally: runner_lock.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["register","run"])
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.mode=="register": register()
    else: run(args.workers)
