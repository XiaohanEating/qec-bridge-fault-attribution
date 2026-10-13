"""Independent process partitions of the registered BB single-fault decisions.

Physical arrays, priors, and scientific code are read from the batch2 snapshot.
Companion partitions use separate files. A completed 256-row result is published
into the sequential runner's checkpoint directory without replacing any file.
Concurrent duplicate computation is checked by array equality, never combined.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import numpy as np
import stim
from .artifacts import WORK, read_json, commit_json, digest
from .baselines import bb_code
from .execution.decoder import ValidatedExecutionDecoder as CachedDecoder
from .experiments import save_npz
from .publication import equal_arrays, publish_arrays
from .decision_cache import DecisionCache

LABELS = ("original", "strong", "single_fault_lookup")
CONTEXT = None


def decode_partition(job):
    global CONTEXT
    instance, order, start, stop = job
    folder = WORK/"results/bb"/instance/order
    output = WORK/"results/bb-parallel"/instance/order/f"{start:06d}"
    output.mkdir(parents=True, exist_ok=True)
    lock = output/"worker.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    began = time.perf_counter()
    try:
        inputs = {name: digest(folder/name) for name in ("single-fault.npz", "prior.stim")}
        identity = (instance, order, tuple(inputs.items()))
        if CONTEXT is None or CONTEXT[0] != identity:
            if CONTEXT is not None: CONTEXT[3].close()
            faults = dict(np.load(folder/"single-fault.npz"))
            circuit = stim.Circuit((folder/"prior.stim").read_text(encoding="utf-8"))
            decoder = CachedDecoder(bb_code(instance), circuit, single_fault=faults)
            cache = DecisionCache(WORK/"results/bb-decision-cache"/instance/(order+".sqlite"),
                dict(inputs=inputs,scientific_source_sha256=digest(os.path.join(os.path.dirname(__file__),"decoding.py")),
                    environment="locked qldpc-submission-audit"),circuit.num_detectors,circuit.num_observables)
            CONTEXT = identity, faults, decoder, cache
        _, faults, decoder, cache = CONTEXT
        assert start % 256 == 0 and stop == min(start+256, len(faults["detectors"]))
        pieces = []
        for substart in range(start, stop, 32):
            substop = min(substart+32, stop)
            path = output/f"rows-{substart:06d}.npz"
            if path.exists():
                piece = dict(np.load(path))
            else:
                piece = cache.predictions(decoder, faults["detectors"][substart:substop])
                assert set(piece) == set(LABELS)
                save_npz(path, **piece)
            assert all(value.shape == (substop-substart, faults["observables"].shape[1]) for value in piece.values())
            cache.store(faults["detectors"][substart:substop],piece)
            pieces.append(piece)
            commit_json(output/"progress.json", dict(pid=os.getpid(), start=start, stop=stop,
                committed_rows=substop-start, seconds=time.perf_counter()-began))
        arrays = {name: np.concatenate([piece[name] for piece in pieces]) for name in LABELS}
        own = output/"decoded.npz"
        if own.exists(): equal_arrays(own, arrays)
        else: save_npz(own, **arrays)
        target = folder/"single-fault-shards"/f"{start:06d}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        publication = publish_arrays(own, target, arrays)
        assert all(digest(folder/name) == value for name, value in inputs.items())
        record = dict(instance=instance, order=order, start=start, stop=stop,
            seconds=time.perf_counter()-began, pid=os.getpid(), inputs=inputs,
            artifact_sha256=digest(own), publication=publication,
            complete_lookup=True, all_logical_vectors=True)
        commit_json(output/"result.json", record)
        return record
    finally:
        lock.unlink(missing_ok=True)


def available_jobs():
    # Interleave row partitions across matrices; the first two partitions are
    # reserved for existing sequential work. Its original loop handles them.
    jobs = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            if not (folder/"physical-risk.json").exists(): continue
            count = len(np.load(folder/"single-fault.npz")["detectors"])
            for start in range(0, count, 256):
                checkpoint = folder/"single-fault-shards"/f"{start:06d}.npz"
                output = WORK/"results/bb-parallel"/instance/order/f"{start:06d}"
                if checkpoint.exists() or (output/"result.json").exists(): continue
                jobs.append((instance, order, start, min(start+256, count)))
    return sorted(jobs, key=lambda job: (job[2], job[0], job[1]))


def run(workers):
    batch = os.environ.get("ISCAS_REVIEW_BATCH", "bb-decoder-parallel")
    registration = read_json(WORK/"provenance"/batch/"registration.json")
    assert all(digest(path) == sha for path, sha in registration["source_sha256"].items())
    runner_lock = WORK/"provenance"/batch/"supervisor.lock"
    fd = os.open(runner_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    started = time.perf_counter()
    completed, failed, claimed = [], [], set()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}
            while True:
                jobs = [job for job in available_jobs() if job not in claimed]
                for job in jobs[:max(0, workers-len(pending))]:
                    claimed.add(job); pending[pool.submit(decode_partition, job)] = job
                if not pending:
                    break
                done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try: completed.append(future.result())
                    except Exception as error:
                        import traceback
                        failed.append(dict(job=job, error=str(error), traceback=traceback.format_exc()))
                commit_json(WORK/"results/bb-parallel-status.json", dict(
                    status="running", pid=os.getpid(), workers=workers,
                    completed_partitions=len(completed), active=list(pending.values()),
                    failed=failed, elapsed_seconds=time.perf_counter()-started,
                    updated_utc=datetime.now(timezone.utc).isoformat()))
        commit_json(WORK/"results/bb-parallel-status.json", dict(
            status="available partitions complete" if not failed else "requires repair",
            completed_partitions=len(completed), failed=failed,
            elapsed_seconds=time.perf_counter()-started,
            note="BB physical enumeration and finite-noise jobs retain their original registered scope"))
    finally:
        runner_lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)
