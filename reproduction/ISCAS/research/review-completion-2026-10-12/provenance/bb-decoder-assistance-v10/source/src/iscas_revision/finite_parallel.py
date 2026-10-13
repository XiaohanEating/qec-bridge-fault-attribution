"""Resume registered BB finite-noise decisions using independent process blocks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import os
import time
import numpy as np
import stim
from .artifacts import WORK, read_json, commit_json, digest
from .baselines import bb_code
from .execution.decoder import PackedExecutionDecoder as CachedDecoder
from .experiments import save_npz
from .publication import equal_arrays, publish_arrays

EPS = (1e-9,1e-8,1e-7,1e-6,3e-6,1e-5,3e-5,1e-4)
CONTEXT = None


def context(instance, order):
    global CONTEXT
    folder=WORK/"results/bb"/instance/order
    inputs={name:digest(folder/name) for name in ("prior.stim","single-fault.npz")}
    identity=(instance,order,tuple(inputs.items()))
    if CONTEXT is None or CONTEXT[0]!=identity:
        faults=dict(np.load(folder/"single-fault.npz"))
        decoder=CachedDecoder(bb_code(instance),stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8")),single_fault=faults)
        # Exact, committed decisions seed the same cache. No guesses are added.
        for path in (folder/"single-fault-shards").glob("*.npz"):
            start=int(path.stem)
            values=dict(np.load(path))
            for label in decoder.states:
                for record,prediction in zip(faults["detectors"][start:start+256],values[label]):
                    key=bytes(record)
                    if key in decoder.cache[label]: assert np.array_equal(decoder.cache[label][key],prediction)
                    decoder.cache[label][key]=prediction.copy()
        CONTEXT=identity,decoder
    return CONTEXT[1],inputs


def partition(job):
    instance,order,eps,start,seed=job
    folder=WORK/"results/bb"/instance/order/f"eps{eps}"
    own=WORK/"results/bb-finite-parallel"/instance/order/f"eps{eps}"/f"{start:06d}"
    own.mkdir(parents=True,exist_ok=True)
    lock=own/"worker.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    began=time.perf_counter()
    try:
        decoder,inputs=context(instance,order)
        path=folder/"physical.npz"
        sample=dict(np.load(path))
        sample_sha=digest(path)
        assert int(sample["seed"])==seed and len(sample["detectors"])==32768
        stop=min(start+128,len(sample["detectors"]))
        pieces=[]
        for a in range(start,stop,32):
            b=min(a+32,stop)
            target=own/f"rows-{a:06d}.npz"
            if target.exists(): values=dict(np.load(target))
            else:
                values=decoder.predictions(sample["detectors"][a:b])
                save_npz(target,**values)
            assert all(v.shape==(b-a,sample["observables"].shape[1]) for v in values.values())
            pieces.append(values)
            commit_json(own/"progress.json",dict(pid=os.getpid(),start=start,stop=stop,
                committed_shots=b-start,seconds=time.perf_counter()-began))
        values={name:np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
        target=own/"predictions.npz"
        if target.exists(): equal_arrays(target,values)
        else: save_npz(target,**values)
        published=folder/"decoded-shards"/f"{start:06d}.npz"
        published.parent.mkdir(parents=True,exist_ok=True)
        publication=publish_arrays(target,published,values)
        assert digest(path)==sample_sha
        record=dict(instance=instance,order=order,eps=eps,start=start,stop=stop,seed=seed,
            inputs=inputs,sample_sha256=sample_sha,predictions_sha256=digest(target),
            seconds=time.perf_counter()-began,pid=os.getpid(),complete_lookup=True,publication=publication,
            decision_configuration="unchanged fixed batch2 prior, original/strong, all logical bits")
        commit_json(own/"result.json",record)
        return record
    finally: lock.unlink(missing_ok=True)


def run(workers):
    batch=os.environ.get("ISCAS_REVIEW_BATCH","bb-finite-parallel")
    registration=read_json(WORK/"provenance"/batch/"registration.json")
    assert all(digest(path)==sha for path,sha in registration["source_sha256"].items())
    tasks=registration["tasks"]
    runner=WORK/"provenance"/batch/"supervisor.lock"
    fd=os.open(runner,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    claimed=set();completed=[];failed=[];began=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={}
            while True:
                available=[];prepared=0;committed=0
                for task in tasks:
                    folder=WORK/"results/bb"/task["instance"]/task["order"]
                    if not (folder/"physical-risk.json").exists(): continue
                    for i,eps in enumerate(EPS):
                        point=folder/f"eps{eps}"
                        if not (point/"physical.npz").exists(): continue
                        prepared+=1
                        for start in range(0,task["shots"],128):
                            job=(task["instance"],task["order"],eps,start,task["seed"]+i*100)
                            if (point/"decoded-shards"/f"{start:06d}.npz").exists():
                                committed+=1;continue
                            if job not in claimed: available.append(job)
                # Complete the smallest-rate point before higher rates. Every
                # registered point remains queued; this changes no sample size.
                available.sort(key=lambda j:(j[2],j[0],j[1],j[3]))
                for job in available[:max(0,workers-len(pending))]:
                    claimed.add(job);pending[pool.submit(partition,job)]=job
                if not pending and prepared==48 and not available: break
                if pending:
                    done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=pending.pop(future)
                        try: completed.append(future.result())
                        except Exception as error:
                            import traceback
                            failed.append(dict(job=job,error=str(error),traceback=traceback.format_exc()))
                else: time.sleep(20)
                commit_json(WORK/"results/bb-finite-parallel-status.json",dict(
                    status="running",pid=os.getpid(),workers=workers,prepared_points=prepared,target_points=48,
                    committed_blocks=committed,target_blocks=48*256,active=list(pending.values()),failed=failed,
                    elapsed_seconds=time.perf_counter()-began,updated_utc=datetime.now(timezone.utc).isoformat()))
        commit_json(WORK/"results/bb-finite-parallel-status.json",dict(
            status="decisions complete" if not failed else "requires repair",failed=failed,
            completed_worker_blocks=len(completed),elapsed_seconds=time.perf_counter()-began,
            scope="sample counts and results merged by the original registered BB task"))
    finally: runner.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4)
    run(parser.parse_args().workers)
