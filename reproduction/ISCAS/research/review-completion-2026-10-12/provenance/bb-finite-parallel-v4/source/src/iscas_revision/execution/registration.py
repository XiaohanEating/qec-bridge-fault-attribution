"""Register immutable, validated execution optimizations without protocol edits."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil

from ..artifacts import WORK, read_json, commit_json, digest


def register(batch, original, kind, workers):
    target = WORK/"provenance"/batch
    assert not target.exists(), "Immutable execution registration already exists"
    validation = WORK/"provenance"/("historical-execution-validation.json" if kind=="historical" else "optimized-execution-validation.json")
    checked = read_json(validation)
    assert len(checked["matrices"]) == 6
    target.mkdir()
    prior = read_json(WORK/"provenance"/original/"registration.json")
    shutil.copytree(WORK/"provenance"/original/"source/src", target/"source/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    destination = target/"source/src/iscas_revision"
    shutil.copytree(WORK/"src/iscas_revision/execution", destination/"execution",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if kind == "historical":
        path = destination/"experiments.py"
        text = path.read_text(encoding="utf-8")
        assert "from .decoding import CachedDecoder, CONFIGURATIONS" in text
        text = text.replace("from .decoding import CachedDecoder, CONFIGURATIONS", "from .decoding import CONFIGURATIONS")
        text += "\n# Exact execution implementation; mathematical configuration is unchanged.\nfrom .execution.decoder import HistoricalExecutionDecoder as CachedDecoder\n"
        path.write_text(text, encoding="utf-8")
    else:
        for filename in ("parallel_decode.py", "finite_parallel.py"):
            path = destination/filename
            text = path.read_text(encoding="utf-8")
            assert "from .decoding import CachedDecoder" in text
            text = text.replace("from .decoding import CachedDecoder", "from .execution.decoder import ExecutionDecoder as CachedDecoder")
            path.write_text(text, encoding="utf-8")
    source = {str(path): digest(path) for path in destination.rglob("*.py")}
    source.update({path: sha for path, sha in prior["source_sha256"].items() if "/source/" not in path.replace("\\", "/")})
    protocol_files = ("decoding.py", "baselines.py", "stratification.py")
    unchanged = {filename: digest(destination/filename)==digest(WORK/"provenance"/original/"source/src/iscas_revision"/filename)
        for filename in protocol_files}
    assert all(unchanged.values())
    registration = dict(prior, registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source, workers=workers, supersedes_scheduler=original,
        scheduling_change="Reuse unchanged UCRT tanh values and linear candidate solves; same BP updates, scalar product order, candidate order, weights and strict comparison",
        validated_by=str(validation), validation_sha256=digest(validation), protocol_sources_unchanged=unchanged,
        execution_packages={package: importlib.metadata.version(package) for package in ("numba", "llvmlite")},
        no_new_dependencies=True, core_versions_unchanged=True)
    commit_json(target/"registration.json", registration)
    print(dict(batch=batch, source_files=len(source), workers=workers), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--kind", choices=("historical", "bb"), required=True)
    p.add_argument("--workers", type=int, required=True)
    a = p.parse_args()
    register(a.batch, a.original, a.kind, a.workers)
