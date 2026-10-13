"""Atomic commits and immutable input identities."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

WORK = Path(os.environ.get("ISCAS_REVIEW_WORKDIR", str(Path(__file__).resolve().parents[2])))
ROOT = WORK.parents[2]
HISTORICAL = ROOT / "ISCAS/research/mechanism-intervention-2026-10-11"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def commit_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
