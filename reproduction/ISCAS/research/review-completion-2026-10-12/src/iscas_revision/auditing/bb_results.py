"""Check BB published partitions against registered samples and final assemblies."""
import argparse
from datetime import datetime, timezone

import numpy as np

from ..artifacts import WORK, read_json, digest, commit_json
from ..audit import optimal_table
from ..publication import equal_arrays
from ..stratification import clopper_pearson

LABELS = ('original', 'strong', 'single_fault_lookup')
EPS = (1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def load_arrays(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def validate_predictions(values, rows, logical_bits, path):
    assert set(values) == set(LABELS), str(path)
    for value in values.values():
        assert value.shape == (rows, logical_bits), str(path)
        assert np.all((value == 0) | (value == 1)), str(path)


def run(require_complete=True):
    registration_path = WORK/'provenance/bb-assembly-v2/registration.json'
    registration = read_json(registration_path)
    for path, expected in registration['source_sha256'].items():
        assert digest(path) == expected, path
    tasks = registration['tasks']
    assert len(tasks) == 6 and all(t['shots'] == 32768 for t in tasks)
    identities = {str(registration_path): digest(registration_path)}
    matrices = []
    missing = []
    sf_total = finite_total = finite_points = final_tasks = 0
    for task in tasks:
        instance, order = task['instance'], task['order']
        folder = WORK/'results/bb'/instance/order
        inputs = {name: digest(folder/name) for name in ('single-fault.npz', 'prior.stim')}
        identities.update({str(folder/name): value for name, value in inputs.items()})
        faults = load_arrays(folder/'single-fault.npz')
        nd, nl = faults['detectors'].shape[1], faults['observables'].shape[1]
        table, optimum = optimal_table(faults['detectors'], faults['observables'], faults['physical_coefficients'])
        assert optimum == read_json(folder/'physical-risk.json')['R_star']
        sf_pieces = []
        sf_sources = {}
        for start in range(0, len(faults['detectors']), 256):
            path = folder/'single-fault-shards'/f'{start:06d}.npz'
            if not path.exists():
                missing.append(str(path)); continue
            stop = min(start+256, len(faults['detectors']))
            values = load_arrays(path)
            validate_predictions(values, stop-start, nl, path)
            for index, record in enumerate(faults['detectors'][start:stop]):
                assert np.array_equal(values['single_fault_lookup'][index], table[bytes(record)]), str(path)
            sf_pieces.append(values)
            sf_sources[str(path)] = digest(path)
            sf_total += 1
        exact = None
        if len(sf_pieces) == (len(faults['detectors'])+255)//256:
            prediction = {label: np.concatenate([p[label] for p in sf_pieces]) for label in LABELS}
            exact = {label: float(faults['physical_coefficients'] @ (prediction[label]^faults['observables']).any(axis=1)) for label in LABELS}
            assert exact['single_fault_lookup'] == optimum
            if (folder/'decoder-risk.json').exists():
                equal_arrays(folder/'single-fault-decoded.npz', prediction)
                actual = read_json(folder/'decoder-risk.json')
                assert actual['R_star'] == optimum
                for label in LABELS:
                    assert np.isclose(actual['decoder'][label]['R_delta'], exact[label], atol=1e-8, rtol=0)
                    assert np.isclose(actual['decoder'][label]['E'], exact[label]-optimum, atol=1e-8, rtol=0)
        rates = []
        for i, eps in enumerate(EPS):
            point = folder/f'eps{eps}'
            sample_path = point/'physical.npz'
            sample = load_arrays(sample_path)
            expected_seed = task['seed']+i*100
            assert int(sample['seed']) == expected_seed, str(sample_path)
            assert sample['detectors'].shape == (32768, nd)
            assert sample['observables'].shape == (32768, nl)
            sample_sha = digest(sample_path)
            identities[str(sample_path)] = sample_sha
            pieces = []
            sources = {}
            for start in range(0, 32768, 128):
                path = point/'decoded-shards'/f'{start:06d}.npz'
                if not path.exists():
                    missing.append(str(path)); continue
                values = load_arrays(path)
                validate_predictions(values, 128, nl, path)
                own = WORK/'results/bb-finite-parallel'/instance/order/f'eps{eps}'/f'{start:06d}'
                record_path = own/'result.json'
                if record_path.exists():
                    record = read_json(record_path)
                    assert (record['instance'], record['order'], record['eps'], record['start'], record['stop'], record['seed']) == (instance, order, eps, start, start+128, expected_seed), str(record_path)
                    assert record['sample_sha256'] == sample_sha, str(record_path)
                    assert record['inputs'] == inputs, str(record_path)
                    assert digest(own/'predictions.npz') == record['predictions_sha256'], str(record_path)
                    equal_arrays(own/'predictions.npz', values)
                else:
                    missing.append(str(record_path))
                pieces.append(values)
                sources[str(path)] = digest(path)
                finite_total += 1
            rate = dict(eps=eps, seed=expected_seed, blocks=len(pieces), sample_sha256=sample_sha, prediction_sha256=sources)
            if len(pieces) == 256:
                finite_points += 1
                pred = {label: np.concatenate([p[label] for p in pieces]) for label in LABELS}
                residual = {label: pred[label]^sample['observables'] for label in LABELS}
                rate['failures'] = {label: int(residual[label].any(axis=1).sum()) for label in LABELS}
                result_path = point/'result.json'
                if result_path.exists():
                    record = read_json(result_path)
                    assert record['sample_sha256'] == sample_sha
                    assert record['task'] == dict(task, eps=eps, seed=expected_seed)
                    assert digest(point/'decoded.npz') == record['decoded_sha256']
                    assembled = load_arrays(point/'decoded.npz')
                    for label in LABELS:
                        assert np.array_equal(assembled[label+'_prediction'], pred[label])
                        assert np.array_equal(assembled[label+'_residual'], residual[label])
                        row = record['decoder'][label]
                        assert row['shots'] == 32768 and row['failures'] == rate['failures'][label]
                        assert row['per_logical'] == residual[label].sum(axis=0).tolist()
                        assert np.allclose(row['interval'], clopper_pearson(row['failures'], row['shots']), atol=1e-13)
                else:
                    missing.append(str(result_path))
            rates.append(rate)
        complete_path = WORK/'results/tasks'/(task['id']+'.json')
        if complete_path.exists():
            assembled = read_json(complete_path)
            assert assembled['kind'] == 'bb' and assembled['status'] == 'complete'
            assert len(assembled['result']['samples']) == 8
            assert assembled['result']['risks'] == read_json(folder/'decoder-risk.json')
            for sample in assembled['result']['samples']:
                assert sample == read_json(folder/f"eps{sample['task']['eps']}"/'result.json')
            final_tasks += 1
        else:
            missing.append(str(complete_path))
        matrices.append(dict(instance=instance, order=order, exact_risks=exact, R_star=optimum, single_fault_sha256=sf_sources, finite_rates=rates))
    complete = not missing and final_tasks == 6 and finite_points == 48 and finite_total == 12288 and sf_total == 231
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, complete=complete,
        counts=dict(single_fault_blocks=sf_total, finite_blocks=finite_total, finite_points=finite_points, final_tasks=final_tasks),
        matrices=matrices, missing=missing, input_sha256=identities, source_sha256=digest(__file__),
        scientific_inputs_modified=False, scope='All logical bits, physical sample identities, seeds, partition metadata and final scientific assemblies')
    commit_json(WORK/'provenance/bb-results-audit.json', result)
    print(dict(passed=True, complete=complete, counts=result['counts'], missing=len(missing)), flush=True)
    if require_complete:
        assert complete, 'BB final acceptance requires every registered partition, point and scientific task'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    run(require_complete=not args.partial)
