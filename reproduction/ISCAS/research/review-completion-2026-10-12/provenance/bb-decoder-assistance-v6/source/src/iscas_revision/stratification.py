"""Exact physical channel conditioning; mutually exclusive Pauli outcomes."""
from collections import defaultdict
import numpy as np
from scipy.stats import beta, binom

NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}


def bind_branch_locations(circuit, branch_data):
    """Old primary files omit target_start; bind only an unambiguous target slice."""
    flat=circuit.flattened()
    bound=[]
    for branch in branch_data["branches"]:
        if "target_start" in branch:
            bound.append(branch); continue
        inst=flat[branch["instruction_offset"]]
        targets=[t.qubit_value for t in inst.targets_copy()]
        width=2 if inst.name == "DEPOLARIZE2" else 1
        flipped={q for q,pauli in branch["paulis"]}
        candidates=[start for start in range(0,len(targets),width) if flipped.issubset(targets[start:start+width])]
        assert len(candidates)==1, "Ambiguous historical location; explicit re-enumeration required"
        bound.append(dict(branch,target_start=candidates[0]))
    return dict(branch_data,branches=bound)


class PhysicalResponses:
    def __init__(self, circuit, branch_data, single_fault):
        branch_data=bind_branch_locations(circuit,branch_data)
        grouped = defaultdict(list)
        for branch in branch_data["branches"]:
            grouped[(branch["instruction_offset"], branch["target_start"])].append(branch["signature"] + 1)
        self.channels, self.divisors = [], []
        count = 0
        for offset, inst in enumerate(circuit.flattened()):
            if inst.name not in NOISE:
                continue
            width = 2 if inst.name == "DEPOLARIZE2" else 1
            divisor = 15 if width == 2 else 3 if inst.name == "DEPOLARIZE1" else 1
            for start in range(0, len(inst.targets_copy()), width):
                visible = grouped.pop((offset,start), [])
                assert len(visible) <= divisor
                count += len(visible)
                self.channels.append(visible + [0] * (15-len(visible)))
                self.divisors.append(divisor)
        assert not grouped, "Source instruction coordinates failed to bind"
        assert count == len(branch_data["branches"])
        self.channels = np.asarray(self.channels, dtype=np.int32)
        self.divisors = np.asarray(self.divisors, dtype=np.int32)
        self.n = len(self.channels)
        self.nd = circuit.num_detectors
        self.nl = circuit.num_observables
        d, l = single_fault["detectors"], single_fault["observables"]
        raw = np.concatenate([d,l], axis=1)
        assert raw.shape[1] == self.nd+self.nl
        self.responses = np.concatenate([np.zeros((1,raw.shape[1]), dtype=np.uint8), raw], axis=0)
        self.packed = np.packbits(self.responses, axis=1, bitorder="little")
        # Padding outside each channel's actual divisor does not count.
        masses = np.zeros(len(self.responses))
        for row,divisor in zip(self.channels,self.divisors):
            np.add.at(masses, row[:divisor], 1/divisor)
        assert np.allclose(masses[1:], single_fault["physical_coefficients"], atol=1e-8)
        assert abs(masses.sum()-self.n) < 1e-7
        assert abs(masses[0]-branch_data["totals"]["invisible_coefficient"]) < 1e-6

    def sample(self, k, shots, seed):
        assert 0 <= k <= self.n
        rng = np.random.default_rng(seed)
        sites = np.empty((shots,k), dtype=np.int32)
        for column in range(k):
            drawn = rng.integers(self.n, size=shots, dtype=np.int32)
            if column:
                duplicate = (sites[:,:column] == drawn[:,None]).any(axis=1)
                while duplicate.any():
                    drawn[duplicate] = rng.integers(self.n, size=int(duplicate.sum()), dtype=np.int32)
                    duplicate = (sites[:,:column] == drawn[:,None]).any(axis=1)
            sites[:,column] = drawn
        outcome = (rng.random((shots,k))*self.divisors[sites]).astype(np.int32)
        signatures = self.channels[sites,outcome]
        packed = np.zeros((shots,self.packed.shape[1]), dtype=np.uint8)
        for column in range(k):
            packed ^= self.packed[signatures[:,column]]
        raw = np.unpackbits(packed, axis=1, count=self.nd+self.nl, bitorder="little")
        return raw[:,:self.nd], raw[:,self.nd:], sites, signatures


def clopper_pearson(failures, shots, alpha=.05):
    return (0.0 if failures == 0 else float(beta.ppf(alpha/2, failures, shots-failures+1)),
            1.0 if failures == shots else float(beta.ppf(1-alpha/2, failures+1, shots-failures)))


def integrate(n, eps, single_risk, strata, alpha=.05):
    """Simultaneous layer intervals plus an explicit unknown-tail bound."""
    kmax = max(strata, default=1)
    p1 = float(binom.pmf(1,n,eps))
    exact = p1*single_risk/n
    mean = low = high = exact
    contributions = {"single_fault":exact}
    adjusted = alpha/max(1,len(strata))
    for k, record in strata.items():
        weight = float(binom.pmf(k,n,eps))
        failures, shots = record["failures"], record["shots"]
        lo, hi = clopper_pearson(failures,shots,adjusted)
        value = weight*failures/shots
        mean += value; low += weight*lo; high += weight*hi
        contributions[str(k)] = value
    tail = float(binom.sf(kmax,n,eps))
    high += tail
    return dict(probability=mean, confidence_interval=[low,min(1.0,high)],
                confidence_method="Bonferroni Clopper-Pearson per conditional stratum plus exact unobserved-tail bound",
                confidence_level=1-alpha, tail_probability_bound=tail,
                exact_single_fault=exact, sampled_multiple_fault=mean-exact,
                linear_prediction=eps*single_risk, contributions=contributions)
