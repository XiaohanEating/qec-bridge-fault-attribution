"""Register independently verified arithmetic reuse with complete provenance."""
import argparse
from datetime import datetime, timezone
import shutil

from ..artifacts import WORK, read_json, commit_json, digest


def register(batch, original, workers):
    target = WORK/"provenance"/batch
    assert not target.exists(), "Immutable execution registration already exists"
    names = ("prefix-product-validation.json", "prefix-execution-validation.json", "sparse-support-validation.json", "window-input-storage-validation.json", "stored-execution-validation.json")
    validations = {name:read_json(WORK/"provenance"/name) for name in names}
    assert all(len(value["matrices"]) == 6 for value in validations.values())
    for value in validations["prefix-execution-validation.json"]["matrices"]:
        assert value["all_logical_predictions_equal"]
        assert all(row["llr_equal"] and row["complete_error_vector_equal"] for row in value["windows"])
    assert validations["sparse-support-validation.json"]["source_sha256"] == digest(WORK/"src/iscas_revision/execution/linear_osd.py")
    prior = read_json(WORK/"provenance"/original/"registration.json")
    target.mkdir()
    shutil.copytree(WORK/"provenance"/original/"source/src",target/"source/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    destination = target/"source/src/iscas_revision"
    for filename in ("__init__.py", "serial_bp.py", "linear_osd.py", "decoder.py", "state_store.py"):
        shutil.copy2(WORK/"src/iscas_revision/execution"/filename,destination/"execution"/filename)
    for filename in ("parallel_decode.py", "finite_parallel.py"):
        path = destination/filename
        if path.exists():
            source = path.read_text(encoding="utf-8")
            source = source.replace("from .execution.decoder import ExecutionDecoder as CachedDecoder",
                "from .execution.decoder import ValidatedExecutionDecoder as CachedDecoder")
            path.write_text(source,encoding="utf-8")
    science = {name:digest(destination/name)==digest(WORK/"provenance"/original/"source/src/iscas_revision"/name)
        for name in ("decoding.py", "baselines.py", "experiments.py", "stratification.py")}
    assert all(science.values())
    sources = {str(path):digest(path) for path in destination.rglob("*.py")}
    sources.update({path:sha for path,sha in prior["source_sha256"].items()
        if "/source/" not in path.replace("\\","/")})
    for folder in (WORK/"provenance/bb-decoder-states").glob("*/*"):
        for filename in ("identity.json","states.npz"):
            sources[str(folder/filename)]=digest(folder/filename)
    record = dict(prior, registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,workers=workers,supersedes_scheduler=original,
        scientific_module_bytes_unchanged=science,
        execution_change="Reuse ascending serial left-fold prefixes; skip suffix only when binary64 yields exactly zero check message; same OSD candidates, order and ascending nonzero-column scoring",
        validated_by={name:dict(path=str(WORK/"provenance"/name),sha256=digest(WORK/"provenance"/name)) for name in names},
        complete_registered_range_retained=True,no_new_dependencies=True)
    commit_json(target/"registration.json",record)
    print(dict(batch=batch,workers=workers,source_files=len(sources)),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--batch",required=True)
    p.add_argument("--original",required=True)
    p.add_argument("--workers",type=int,required=True)
    a=p.parse_args()
    register(a.batch,a.original,a.workers)
