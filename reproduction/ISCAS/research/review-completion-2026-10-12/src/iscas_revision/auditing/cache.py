"""Compare published BB predictions with the shared exact decision cache."""
from datetime import datetime, timezone
import json
import sqlite3

import numpy as np

from ..artifacts import WORK, commit_json, digest

LABELS = ("original", "strong", "single_fault_lookup")


def audit():
    matrices = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            db = WORK/"results/bb-decision-cache"/instance/(order+".sqlite")
            if not db.exists():
                continue
            folder = WORK/"results/bb"/instance/order
            with sqlite3.connect(db.as_uri()+"?mode=ro", uri=True, timeout=30) as connection:
                connection.execute("BEGIN")
                assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
                metadata = json.loads(connection.execute("SELECT identity FROM metadata WHERE id=1").fetchone()[0])
                assert metadata["labels"] == list(LABELS)
                assert all(digest(folder/name) == sha for name, sha in metadata["inputs"].items())
                stored = dict(connection.execute("SELECT record,prediction FROM decisions"))
            nd, nl = metadata["nd"], metadata["nl"]
            for key, value in stored.items():
                assert len(key) == (nd+7)//8 and len(value) == 3*nl
                assert set(value).issubset({0, 1})
            counters = dict(files=0, rows=0, cached_rows=0, later_or_uncached_rows=0)
            checked_files = {}

            def check(path, records):
                with np.load(path) as values:
                    assert set(values.files) == set(LABELS)
                    assert all(values[label].shape == (len(records), nl) for label in LABELS)
                    assert all(np.all((values[label] == 0) | (values[label] == 1)) for label in LABELS)
                    for i, record in enumerate(records):
                        key = np.packbits(record, bitorder="little").tobytes()
                        expected = b"".join(values[label][i].astype(np.uint8).tobytes() for label in LABELS)
                        if key in stored:
                            assert stored[key] == expected, f"Cache mismatch: {path}, row {i}"
                            counters["cached_rows"] += 1
                        else:
                            counters["later_or_uncached_rows"] += 1
                counters["files"] += 1
                counters["rows"] += len(records)
                checked_files[str(path)] = digest(path)

            with np.load(folder/"single-fault.npz") as single:
                records = single["detectors"]
                for path in sorted((folder/"single-fault-shards").glob("*.npz")):
                    start = int(path.stem)
                    check(path, records[start:start+256])
            for point in sorted(folder.glob("eps*")):
                if not (point/"physical.npz").exists():
                    continue
                with np.load(point/"physical.npz") as sample:
                    records = sample["detectors"]
                    for path in sorted((point/"decoded-shards").glob("*.npz")):
                        start = int(path.stem)
                        check(path, records[start:start+128])
            matrices.append(dict(instance=instance, order=order, cache_records=len(stored),
                metadata=metadata, **counters, checked_files_sha256=checked_files))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), matrices=matrices,
        published_files=sum(m["files"] for m in matrices),
        cached_rows_checked=sum(m["cached_rows"] for m in matrices),
        read_only=True, snapshot_scope="Published files visible during audit; cache read transaction per matrix")
    commit_json(WORK/"provenance/published-cache-audit.json", result)
    print({key: result[key] for key in ("published_files", "cached_rows_checked", "read_only")}, flush=True)


if __name__ == "__main__":
    audit()
