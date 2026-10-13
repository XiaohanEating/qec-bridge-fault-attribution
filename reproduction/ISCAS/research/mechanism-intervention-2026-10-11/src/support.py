"""Fixed inputs, atomic records and predeclared decoder configurations."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import stim

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[2]
sys.path[:0] = [str(ROOT / 'ISCAS/src'), str(ROOT / 'ISCAS/experiments'),
               str(ROOT / 'ISCAS/research/conflict-fault-routing-2026-10-11/src'),
               str(ROOT / 'ISCAS/research/parallel-routing-2026-10-10/src')]
from quits import HgpCode
from quits.decoder.base import spacetime
from ldpc.bposd_decoder import BpOsdDecoder
from pl_validation import sd6
from decoder_aware import decode_lone_faults
from joint_router import inherited, synthesize

DECODERS = {
    'original': dict(window=1, max_iter=2, osd_order=0),
    'strong': dict(window=4, max_iter=50, osd_order=2),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda v: int(v)), encoding='utf-8')
    tmp.replace(path)

def code_for(name):
    h3 = np.array([[1, 1, 0], [0, 1, 1]])
    h4 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    return HgpCode(h3, h3 if name == 'tiny-hgp' else h4)

def decoder_state(code, circuit, label):
    config = DECODERS[label]
    window = config['window']
    h = code.hz
    num_rounds = circuit.num_detectors // h.shape[0] - 2
    num_cor = (2 + num_rounds - window + window - 1) // window
    checks, observations, priors, updates = spacetime(circuit, h, window, window, num_cor)
    decoders = [BpOsdDecoder(check, channel_probs=prior, bp_method='product_sum',
                            schedule='serial', max_iter=config['max_iter'],
                            osd_method='osd_cs', osd_order=config['osd_order'])
                for check, prior in zip(checks, priors)]
    return dict(decoders=decoders, obs_set=observations, update=updates,
                num_cor_rounds=num_cor, hz_rows=h.shape[0], W=window, F=window)

def predict(state, samples):
    """Same sliding-window recursion, keeping each observable instead of any()."""
    predictions = np.zeros((len(samples), state['obs_set'][0].shape[0]), dtype=np.uint8)
    rows, width, step = state['hz_rows'], state['W'], state['F']
    count = state['num_cor_rounds']
    for i, sample in enumerate(samples):
        accumulated = np.zeros(predictions.shape[1], dtype=int)
        update = np.zeros(rows, dtype=int)
        for k in range(count):
            diff = sample[step*k*rows:(step*k+width)*rows].copy() % 2
            diff[:rows] = (diff[:rows] + update) % 2
            errors = state['decoders'][k].decode(diff)
            m = state['obs_set'][k].shape[1]
            accumulated ^= np.asarray(state['obs_set'][k] @ errors[:m] % 2).ravel()
            update = np.asarray(state['update'][k] @ errors[:m] % 2).ravel()
        diff = sample[step*count*rows:].copy() % 2
        diff[:rows] = (diff[:rows] + update) % 2
        errors = state['decoders'][count].decode(diff)
        accumulated ^= np.asarray(state['obs_set'][count] @ errors % 2).ravel()
        predictions[i] = accumulated
    return predictions

def annotations(circuit):
    return [str(i) for i in circuit.flattened() if i.name in {'DETECTOR', 'OBSERVABLE_INCLUDE'}]

def signature(terms):
    ds, ls = set(), set()
    for target in terms:
        pool = ds if target.is_relative_detector_id() else ls if target.is_logical_observable_id() else None
        if pool is not None:
            if target.val in pool: pool.remove(target.val)
            else: pool.add(target.val)
    return tuple(sorted(ds)), tuple(sorted(ls))
