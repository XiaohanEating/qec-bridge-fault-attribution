"""Execute locked serial product-sum BP with cached identical tanh values.

Check products retain the original edge order and scalar multiplications.
Only repeated calls to tanh of an unchanged message are reused. UCRT log and
tanh retain the installed Windows binary's scalar transcendental functions.
"""
import ctypes

import numpy as np
from numba import njit

RUNTIME = ctypes.CDLL("ucrtbase")
LOG = RUNTIME.log
LOG.argtypes = (ctypes.c_double,)
LOG.restype = ctypes.c_double
TANH = RUNTIME.tanh
TANH.argtypes = (ctypes.c_double,)
TANH.restype = ctypes.c_double


@njit(fastmath=False, error_model="numpy")
def decode_serial(syndrome, initial, rowptr, colptr, coledges, edgecols, edgerows, max_iter):
    n = len(initial)
    edges = len(edgecols)
    llr = np.zeros(n, dtype=np.float64)
    messages = np.zeros(edges, dtype=np.float64)
    transformed = np.zeros(edges, dtype=np.float64)
    checks = np.zeros(edges, dtype=np.float64)
    decoded = np.zeros(n, dtype=np.uint8)
    if not np.any(syndrome):
        return decoded, llr, True, 0
    for e in range(edges):
        messages[e] = initial[edgecols[e]]
        transformed[e] = TANH(messages[e]/2)
    for iteration in range(1, max_iter+1):
        for bit in range(n):
            llr[bit] = initial[bit]
            for pos in range(colptr[bit], colptr[bit+1]):
                e = coledges[pos]
                row = edgerows[e]
                product = 1.0
                for g in range(rowptr[row], rowptr[row+1]):
                    if g != e:
                        product *= transformed[g]
                sign = -1.0 if syndrome[row] else 1.0
                checks[e] = sign*LOG((1+product)/(1-product))
                messages[e] = llr[bit]
                llr[bit] += checks[e]
            decoded[bit] = 1 if llr[bit] <= 0 else 0
            temp = 0.0
            for pos in range(colptr[bit+1]-1, colptr[bit]-1, -1):
                e = coledges[pos]
                messages[e] += temp
                temp += checks[e]
                transformed[e] = TANH(messages[e]/2)
        converged = True
        for row in range(len(syndrome)):
            value = 0
            for e in range(rowptr[row], rowptr[row+1]):
                value ^= decoded[edgecols[e]]
            if value != syndrome[row]:
                converged = False
                break
        if converged:
            return decoded, llr, True, iteration
    return decoded, llr, False, max_iter


class ExactSerialBp:
    def __init__(self, check, prior, max_iter):
        csr = check.tocsr(copy=True)
        csr.sort_indices()
        self.rowptr = csr.indptr.astype(np.int64)
        self.edgecols = csr.indices.astype(np.int64)
        self.edgerows = np.repeat(np.arange(check.shape[0], dtype=np.int64), np.diff(self.rowptr))
        # Stable edge order produces ascending rows in every column.
        self.coledges = np.argsort(self.edgecols, kind="stable").astype(np.int64)
        self.colptr = np.concatenate([[0], np.cumsum(np.bincount(self.edgecols, minlength=check.shape[1]))]).astype(np.int64)
        self.initial = np.array([LOG((1-float(p))/float(p)) for p in prior], dtype=np.float64)
        self.max_iter = max_iter
        self.converge = False
        self.log_prob_ratios = np.zeros(len(prior))
        self.iter = 0

    def decode(self, syndrome):
        value, llr, converge, iteration = decode_serial(np.asarray(syndrome, dtype=np.uint8), self.initial,
            self.rowptr, self.colptr, self.coledges, self.edgecols, self.edgerows, self.max_iter)
        self.log_prob_ratios = llr
        self.converge = converge
        self.iter = iteration
        return value
