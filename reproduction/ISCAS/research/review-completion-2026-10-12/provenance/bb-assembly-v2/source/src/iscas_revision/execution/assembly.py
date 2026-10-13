"""Assemble registered BB tasks only after all exact decisions are published."""
from datetime import datetime, timezone
import os
import time

import numpy as np

from ..artifacts import WORK, read_json, commit_json, digest
from ..experiments import dispatch
EPS = (1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def ready(task):
    folder = WORK/"results/bb"/task["instance"]/task["order"]
    if not (folder/"physical-risk.json").exists():
        return False
    with np.load(folder/"single-fault.npz") as f:
        count = f["detectors"].shape[0]
    if any(not (folder/"single-fault-shards"/f"{start:06d}.npz").exists() for start in range(0, count, 256)):
        return False
    for eps in EPS:
        point = folder/f"eps{eps}"
        if not (point/"physical.npz").exists():
            return False
        if any(not (point/"decoded-shards"/f"{start:06d}.npz").exists() for start in range(0, task["shots"], 128)):
            return False
    return True


def run():
    batch = os.environ["ISCAS_REVIEW_BATCH"]
    folder = WORK/"provenance"/batch
    registration = read_json(folder/"registration.json")
    assert all(digest(path) == sha for path, sha in registration["source_sha256"].items())
    lock = folder/"supervisor.lock"
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.perf_counter()
    try:
        while True:
            remaining = [task for task in registration["tasks"] if not (WORK/"results/tasks"/(task["id"]+".json")).exists()]
            for task in remaining:
                if ready(task) and not (WORK/"results/locks"/(task["id"]+".lock")).exists():
                    print(dispatch(task), flush=True)
            commit_json(WORK/"results"/(batch+"-status.json"), dict(status="complete" if not remaining else "waiting for exact published decisions",
                pid=os.getpid(), remaining=[task["id"] for task in remaining], elapsed_seconds=time.perf_counter()-started,
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if not remaining:
                break
            time.sleep(20)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    run()
