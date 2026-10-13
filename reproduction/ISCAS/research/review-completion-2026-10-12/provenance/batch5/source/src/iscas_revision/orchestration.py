"""Register immutable batches and run exclusive, resumable process tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from scipy.stats import binom
from .artifacts import ROOT, WORK, HISTORICAL, commit_json, read_json, digest
from .experiments import EPS, dispatch, integrate_results

BATCH=os.environ.get("ISCAS_REVIEW_BATCH","batch1")


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
    snapshot=WORK/"provenance"/BATCH/"source/src/iscas_revision"
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
    tasks=tasks_baselines() if BATCH.startswith("batch2") else tasks_ready()
    if BATCH.startswith("batch3"):
        tasks=[dict(id="batch3-0000-order-space",kind="order_space",seed=820261012,shots=65536)]
    if BATCH.startswith("batch4"):
        tasks=[dict(task,id=task["id"].replace("batch2","batch4"),revision="-v2")
               for task in tasks_baselines() if task["kind"]=="zhao"]
    if BATCH.startswith("batch5"):
        controls=[dict(kind="native",distance=distance,eps=eps,shots=32768,initial_noisy=True)
                  for distance in (3,5) for eps in EPS]
        controls.extend(dict(task,initial_noisy=True,revision="-v3") for task in tasks_baselines()
                        if task["kind"]=="zhao" and task["noise"]=="common_SD6")
        controls.extend(dict(kind="joint_bridge",order=order,eps=eps,shots=32768)
                        for order in ("forward","macro_reverse","check_reverse") for eps in EPS)
        tasks=[dict(task,id=f"batch5-{index:04d}-{task['kind']}",seed=920261012+1000*index)
               for index,task in enumerate(controls)]
    strata=[task for task in tasks if task["kind"]=="stratum"]
    record=dict(registered_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,python_version=sys.version,
                packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","stim","ldpc","quits","pymatching","bposd","pyscipopt")},
                source_sha256=sources,protocol_sha256=digest(WORK/"protocol.md"),tasks=tasks,
                conditional_shots=sum(t["shots"] for t in strata),conditional_strata=len(strata),
                raw_confirmation_shots=sum(t.get("shots",0)*(8 if t["kind"]=="bb" else 1) for t in tasks if t["kind"] not in {"stratum","onefactor"}),
                pending_full_scope=["BB-72 and BB-108 construction, risks and finite-noise",
                                   "Zhao official routing, original and common-noise controls",
                                   "standard nearest-neighbor surface-code SD6 d=3,5",
                                   "source-specific sensitivity risk reweighting and completed paper revision"],
                stopping="fixed registered samples; unfinished requirements retained explicitly",
                serial_benchmark=read_json(WORK/"provenance/conditional-benchmark.json"))
    record["execution_revision"]=BATCH
    if BATCH.startswith("batch1-"): record["repair"]="Supervisor JSON import and progress recording repair; same tasks, seeds and scientific settings"
    if BATCH.startswith("batch2"):
        record["purpose"]="BB-72/108, actual Zhao routing, matched SD6 native controls and excess loss at zero optimal risk"
        record["pending_full_scope"]=["joint synthesis of both batches", "idle reweighted single-fault risks", "verified manuscript and page-count revision"]
        record["bb_eps"]=[1e-9,1e-8,1e-7,1e-6,3e-6,1e-5,3e-5,1e-4]
    if BATCH.startswith("batch3"):
        record["purpose"]="64 physically different noisy-round orderings, resource matching, coefficient-only selection and independent confirmation"
        record["pending_full_scope"]=["joint synthesis and verified manuscript revision"]
    if BATCH.startswith("batch4"):
        record["purpose"]="Zhao adapter repair: flatten repeated circuits for physical fault coordinates; distinguish auxiliary measurements from terminal data readout in common SD6"
        record["prior_attempts_retained"]=True
    commit_json(WORK/"provenance"/BATCH/"registration.json",record)
    print({"registered_tasks":len(tasks),"conditional_strata":len(strata),"conditional_shots":record["conditional_shots"],
           "snapshot_pythonpath":str(snapshot.parent)},flush=True)


def tasks_baselines():
    tasks=[]
    for distance in (3,5):
        for eps in EPS:
            tasks.append(dict(kind="native",distance=distance,eps=eps,shots=32768,z_only=True))
    for distance in (3,5):
        for order in ("normal","normal_hook_proned","ZX_interleaved_Z2Z_X2X","alternating"):
            tasks.append(dict(kind="hirai_excess",distance=distance,order=order,eps=.0001,shots=32768))
    for instance in ("bb-72","bb-108"):
        for order in ("unrouted","routing50","routing75"):
            for noise in ("author","common_SD6"):
                for eps in (1e-6,1e-5,1e-4):
                    tasks.append(dict(kind="zhao",instance=instance,order=order,noise=noise,eps=eps,shots=32768))
    for instance in ("bb-72","bb-108"):
        for order in ("forward","macro_reverse","check_reverse"):
            tasks.append(dict(kind="bb",instance=instance,order=order,shots=32768))
    for index,task in enumerate(tasks):
        task["id"]=f"batch2-{index:04d}-{task['kind']}"
        task["seed"]=720261012+index*10000
    return tasks


def run(workers):
    record=read_json(WORK/"provenance"/BATCH/"registration.json")
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
    runner_lock=WORK/"provenance"/BATCH/"supervisor.lock"
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
                status_name=BATCH.split("-")[0]+"-status.json"
                commit_json(WORK/"results"/status_name,dict(status="running",pid=os.getpid(),workers=workers,
                    completed_tasks=completed,total_tasks=len(tasks),failed=failed,
                    active=[task["id"] for task in pending.values()],elapsed_seconds=time.perf_counter()-started,
                    updated_utc=datetime.now(timezone.utc).isoformat(),full_scope_remaining=record["pending_full_scope"]))
                integrate_results()
        status="complete" if not failed else "requires repair of failed tasks"
        status_name=BATCH.split("-")[0]+"-status.json"
        commit_json(WORK/"results"/status_name,dict(status=status,completed_tasks=completed,total_tasks=len(tasks),failed=failed,
                     full_scope_remaining=record["pending_full_scope"],elapsed_seconds=time.perf_counter()-started))
    finally: runner_lock.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["register","run"])
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.mode=="register": register()
    else: run(args.workers)
