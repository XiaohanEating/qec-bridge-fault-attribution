"""Bind immutable BB execution sources to completed equivalence checks."""
from datetime import datetime, timezone
from pathlib import Path

from ..artifacts import WORK, read_json, digest, commit_json


def run():
    names = ('historical-execution-validation.json', 'optimized-execution-validation.json',
        'prefix-execution-validation.json', 'stored-execution-validation.json',
        'stable-execution-validation.json', 'packed-execution-validation.json')
    validated = {}
    sources = {}
    for name in names:
        path = WORK/'provenance'/name
        record = read_json(path)
        sources[str(path)] = digest(path)
        for value in [record, *record.get('matrices', [])]:
            for field in ('source_sha256', 'optimized_source_sha256'):
                if isinstance(value.get(field), dict):
                    for source, sha in value[field].items():
                        if Path(source).name in ('linear_osd.py', 'serial_bp.py', 'decoder.py'):
                            validated.setdefault((Path(source).name, sha), set()).add(name)
    scientific = ('decoding.py', 'baselines.py', 'experiments.py', 'stratification.py')
    rows = []
    for batch in ('bb-finite-parallel-v4', 'bb-finite-assistance-v6',
            'bb-finite-assistance-v8', 'bb-finite-assistance-v9', 'bb-finite-assistance-v10'):
        registration_path = WORK/'provenance'/batch/'registration.json'
        registration = read_json(registration_path)
        sources[str(registration_path)] = digest(registration_path)
        bindings = {}
        for source, sha in registration['source_sha256'].items():
            assert digest(source) == sha, source
            basename = Path(source).name
            if basename in scientific:
                assert digest(WORK/'provenance/bb-assembly-v2/source/src/iscas_revision'/basename) == sha
            if basename in ('linear_osd.py', 'serial_bp.py', 'decoder.py'):
                assert (basename, sha) in validated, (batch, basename)
                bindings[basename] = dict(sha256=sha, equivalence_checks=sorted(validated[basename, sha]))
        assert len(bindings) == 3
        rows.append(dict(batch=batch, bindings=bindings, scientific_modules_unchanged=True))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        batches=rows, input_sha256=sources, source_sha256=digest(__file__),
        scope='Registered immutable numeric modules appear in completed equivalence checks; original scientific modules match the assembly snapshot')
    commit_json(WORK/'provenance/execution-source-bindings.json', result)
    print(dict(passed=True, batches=len(rows)), flush=True)


if __name__ == '__main__':
    run()
