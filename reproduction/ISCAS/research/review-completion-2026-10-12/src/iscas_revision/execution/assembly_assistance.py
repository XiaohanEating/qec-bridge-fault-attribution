"""Assist existing BB assembly using its unchanged registered source snapshot."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from ..artifacts import WORK, read_json, commit_json, digest


def launch():
    registration_path = WORK/'provenance/bb-assembly-v2/registration.json'
    registration = read_json(registration_path)
    assert sys.executable.casefold() == registration['python'].casefold()
    assert all(digest(path) == value for path, value in registration['source_sha256'].items())
    folder = WORK/'provenance/bb-assembly-assistance'
    folder.mkdir(parents=True, exist_ok=True)
    runner = (
        'import json,sys; '
        'from iscas_revision.artifacts import WORK,read_json,digest; '
        'from iscas_revision.experiments import dispatch; '
        'r=read_json(WORK/"provenance/bb-assembly-v2/registration.json"); '
        'assert all(digest(p)==h for p,h in r["source_sha256"].items()); '
        't=next(t for t in r["tasks"] if t["id"]==sys.argv[1]); '
        'print(json.dumps(dispatch(t)),flush=True)'
    )
    (folder/'runner-source.txt').write_text(runner, encoding='utf-8')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(WORK/'provenance/bb-assembly-v2/source/src')
    environment['PYTHONIOENCODING'] = 'utf-8'
    launched, skipped = [], []
    for task in registration['tasks']:
        identifier = task['id']
        if (WORK/'results/tasks'/(identifier+'.json')).exists() or (WORK/'results/locks'/(identifier+'.lock')).exists():
            skipped.append(identifier)
            continue
        with (folder/(identifier+'-stdout.log')).open('ab') as stdout, (folder/(identifier+'-stderr.log')).open('ab') as stderr:
            process = subprocess.Popen([sys.executable, '-c', runner, identifier],
                env=environment, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW)
        launched.append(dict(task=task, pid=process.pid))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__), runner_sha256=digest(folder/'runner-source.txt'),
        registration_sha256=digest(registration_path), launched=launched, skipped=skipped,
        unchanged_scientific_source=True, purpose='Parallel original task assembly from completed registered shards')
    commit_json(folder/'launch.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    launch()
