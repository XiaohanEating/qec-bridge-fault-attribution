"""Final acceptance requires complete experiments and the actual revised PDF."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import re
import shutil
import subprocess

from ..artifacts import ROOT, WORK, read_json, digest, commit_json


def command(*args):
    binary = shutil.which(args[0])
    assert binary, args[0]
    result = subprocess.run([binary, *args[1:]], capture_output=True, text=True,
        encoding='utf-8', errors='replace', check=True)
    return result.stdout


def verify():
    audited = read_json(WORK/'provenance/bb-results-audit.json')
    assert audited['passed'] and audited['complete']
    assert audited['counts'] == dict(single_fault_blocks=231, finite_blocks=12288, finite_points=48, final_tasks=6)
    results = read_json(WORK/'provenance/committed-results-audit.json')
    assert results['passed'] and results['counts']['conditional_shards'] == 6144
    assert results['counts']['bb-finite-parallel_published_partitions'] == 12288
    components = read_json(WORK/'results/bb-components.json')
    assert components['single_fault_complete'] and components['finite_complete']
    comparisons = read_json(WORK/'results/bb-comparisons.json')
    assert comparisons['complete'] and len(comparisons['comparisons']) == 96
    bindings = read_json(WORK/'provenance/execution-source-bindings.json')
    assert bindings['passed'] and len(bindings['batches']) == 5
    for path, expected in bindings['input_sha256'].items():
        assert digest(path) == expected, path
    figure = read_json(WORK/'figures/bb-finite-noise-validation.json')
    assert figure['complete'] and figure['plotted_points'] == 48
    assert figure['input_sha256'] == digest(WORK/'results/bb-components.json')
    report_check = read_json(WORK/'provenance/bb-report-verification.json')
    assert report_check['complete'] and report_check['finite_points'] == 48
    assert report_check['comparisons'] == 96
    assert report_check['report_sha256'] == digest(WORK/'reports/bb-complete-evidence.md')
    status = read_json(WORK/'results/status.json')
    assert not status['pending_tasks']
    assert status['counts']['bb']['complete'] == 6 and status['counts']['stratum']['complete'] == 96
    cache = read_json(WORK/'provenance/published-cache-audit.json')
    assert cache['published_files'] == 12519
    assert all(m['later_or_uncached_rows'] == 0 for m in cache['matrices'])
    for matrix in cache['matrices']:
        for path, expected in matrix['checked_files_sha256'].items():
            assert digest(path) == expected, path
    for path, expected in read_json(WORK/'results/historical-audit.json')['input_sha256'].items():
        if Path(path).as_posix() == 'ISCAS/paper/manuscript/main.tex':
            assert digest(WORK/'provenance/manuscript-before-review/main.tex') == expected, path
        else:
            assert digest(ROOT/path) == expected, path
    folder = ROOT/'ISCAS/paper/manuscript'
    source, pdf = folder/'main.tex', folder/'ISCAS2027-review-draft.pdf'
    tex = source.read_text(encoding='utf-8')
    assert 'Working draft' not in tex and 'running' not in tex
    assert tex.startswith(r'\documentclass[conference]{IEEEtran}')
    assert not re.search(r'\\(?:fontsize|textheight|baselinestretch|linespread|vspace|enlargethispage)', tex)
    log = (folder/'ISCAS2027-review-draft.log').read_text(encoding='utf-8')
    assert not re.search(r'Overfull|undefined|LaTeX Warning|Package .* Warning|^!', log, re.M)
    biblog = (folder/'ISCAS2027-review-draft.blg').read_text(encoding='utf-8')
    assert not re.search(r'Warning--|error message|I couldn.t open', biblog)
    info = command('pdfinfo', str(pdf))
    assert re.search(r'Pages:\s+5\b', info)
    fonts = command('pdffonts', str(pdf))
    font_rows = fonts.splitlines()[2:]
    assert font_rows and all('Type 3' not in row for row in font_rows)
    assert all(re.search(r'\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$', row) for row in font_rows)
    page5 = command('pdftotext', '-f', '5', '-l', '5', str(pdf), '-')
    normalized_page5 = re.sub(r'\s+', '', page5).upper()
    assert 'REFERENCES' in normalized_page5 and 'CONCLUSION' not in normalized_page5
    references = (folder/'ISCAS2027-review-draft.bbl').read_text(encoding='utf-8')
    assert len(re.findall(r'\\bibitem\{', references)) == 32
    visual = read_json(WORK/'provenance/manuscript-final-visual-review.json')
    assert visual['pdf_sha256'] == digest(pdf) and visual['pages_reviewed'] == [1, 2, 3, 4, 5]
    assert visual['passed']
    paths = [source, pdf, folder/'references.bib', WORK/'reports/reviewer-response.md',
        WORK/'reports/bb-complete-evidence.md', WORK/'reports/review-evidence-index.md',
        WORK/'provenance/bb-results-audit.json', WORK/'provenance/committed-results-audit.json',
        WORK/'provenance/published-cache-audit.json', WORK/'results/bb-components.json',
        WORK/'results/bb-comparisons.json', WORK/'results/multiple-fault-attribution.json',
        WORK/'results/low-noise-comparisons.json', WORK/'results/direct-comparisons.json',
        WORK/'results/order-space/comparisons.json', WORK/'results/deployment-comparisons.json',
        WORK/'results/sensitivity-comparisons.json', WORK/'results/decoder-comparisons.json',
        WORK/'results/bb-memory-distance-lower-bound.json',
        WORK/'provenance/execution-source-bindings.json',
        WORK/'provenance/manuscript-final-visual-review.json',
        WORK/'provenance/bb-report-verification.json',
        folder/'review-revision-notes.md', WORK/'README.md',
        WORK/'figures/bb-finite-noise.pdf', WORK/'figures/bb-finite-noise.png',
        WORK/'figures/bb-finite-noise-validation.json',
        WORK/'figures/bb-finite-noise-provenance.json']
    source_files = list((WORK/'src').rglob('*.py'))
    for path in source_files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    paths.extend(source_files)
    for path in paths:
        assert path.exists(), str(path)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        scope='All registered review experiments and revised manuscript; author scientific review and conference submission remain author decisions',
        pages=5, technical_pages=4, references_pages=1, cited_references=32,
        default_font_pt=10, all_fonts_embedded=True, type3_fonts=False,
        pdf_bytes=pdf.stat().st_size, python_syntax_files=len(source_files),
        input_sha256={str(path):digest(path) for path in paths},
        scientific_scope_preserved=True, historical_inputs_unchanged=True,
        bb_full_points=48, bb_comparisons=96, all_registered_tasks_complete=True,
        source_sha256=digest(__file__))
    commit_json(WORK/'provenance/final-review-acceptance.json', result)
    from ..reporting import collect, report
    final_status = collect()
    assert final_status['status'] == 'complete'
    report(final_status)
    result['input_sha256'][str(WORK/'reports/review-evidence-index.md')] = digest(WORK/'reports/review-evidence-index.md')
    result['input_sha256'][str(WORK/'results/status.json')] = digest(WORK/'results/status.json')
    commit_json(WORK/'provenance/final-review-acceptance.json', result)
    commit_json(WORK/'provenance/manuscript-working-draft-verification.json', dict(result,
        status='all registered calculations and manuscript verification complete',
        pdf_sha256=digest(pdf), source_sha256=digest(source), bib_sha256=digest(folder/'references.bib'), pending=[]))
    print({key:result[key] for key in ('passed','pages','cited_references','bb_full_points','bb_comparisons')}, flush=True)
    return result


if __name__ == '__main__':
    verify()
