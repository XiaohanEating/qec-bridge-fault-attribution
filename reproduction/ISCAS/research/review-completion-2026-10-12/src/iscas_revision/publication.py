"""Publish completed arrays on Windows volumes without hard-link support."""
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def equal_arrays(path, arrays):
    with np.load(path) as saved:
        assert set(saved.files) == set(arrays), str(path)
        assert all(np.array_equal(saved[name], arrays[name]) for name in arrays), str(path)


def publish_arrays(source, target, arrays):
    """Windows rename commits a whole file and refuses an existing destination."""
    if os.name != "nt":
        raise RuntimeError("This publication protocol requires Windows rename semantics")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        equal_arrays(target, arrays)
        return "existing checkpoint verified elementwise"
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".publish", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output, Path(source).open("rb") as original:
            shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.rename(temporary, target)
            publication = "atomic Windows rename; existing files never replaced"
        except FileExistsError:
            publication = "existing concurrent checkpoint verified elementwise"
        equal_arrays(target, arrays)
        return publication
    finally:
        Path(temporary).unlink(missing_ok=True)
