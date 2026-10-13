"""Reuse linear solves while preserving ldpc 2.4.1 combination-sweep choices.

The candidate set, UCRT qsort, scalar weight summation order, and strict '<'
selection follow the locked native source. Native BP provides the same LLRs.
This module is not enabled in running batches before equivalence validation.
"""
import ctypes
import os

import numpy as np
from ldpc import BpDecoder
from numba import cfunc, njit, types
from .serial_bp import ExactSerialBp


class SortEntry(ctypes.Structure):
    _fields_ = [("value", ctypes.c_double), ("index", ctypes.c_int)]


COMPARE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


@cfunc(types.intc(types.CPointer(types.float64), types.CPointer(types.float64)))
def compiled_compare(a, b):
    first = a[0]
    second = b[0]
    return 1 if first>second else -1 if first<second else 0


compare = COMPARE(compiled_compare.address)


def native_order(llr):
    assert os.name == "nt", "Locked UCRT ordering is Windows-specific"
    entries = np.empty(len(llr), dtype=np.dtype([("value", np.float64), ("index", np.int32)], align=True))
    assert entries.dtype.itemsize == ctypes.sizeof(SortEntry)
    entries["value"] = llr
    entries["index"] = np.arange(len(llr), dtype=np.int32)
    runtime = ctypes.CDLL("ucrtbase")
    runtime.qsort.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, COMPARE)
    runtime.qsort(entries.ctypes.data, len(entries), ctypes.sizeof(SortEntry), compare)
    return entries["index"].astype(np.int64)


@njit(cache=True, fastmath=False)
def candidates(reduced, pivots, nonpivots, rhs, costs, order):
    n = len(costs)
    rank = len(pivots)
    base = np.zeros(n, dtype=np.uint8)
    for i in range(rank):
        base[pivots[i]] = rhs[i]
    best = base.copy()
    if order == 0:
        return base, best
    minimum = 0.0
    for column in range(n):
        if base[column]:
            minimum += costs[column]
    # Native COMBINATION_SWEEP enumerates each weight-one nonpivot first.
    for selected in range(len(nonpivots)):
        column = nonpivots[selected]
        value = base.copy()
        value[column] = 1
        for i in range(rank):
            value[pivots[i]] ^= reduced[i, column]
        weight = 0.0
        for j in range(n):
            if value[j]:
                weight += costs[j]
        if weight < minimum:
            minimum = weight
            best = value
    # The remaining candidates contain two of the first osd_order nonpivots.
    for i in range(min(order, len(nonpivots))):
        for j in range(i+1, min(order, len(nonpivots))):
            first, second = nonpivots[i], nonpivots[j]
            value = base.copy()
            value[first] = value[second] = 1
            for row in range(rank):
                value[pivots[row]] ^= reduced[row, first]^reduced[row, second]
            weight = 0.0
            for column in range(n):
                if value[column]:
                    weight += costs[column]
            if weight < minimum:
                minimum = weight
                best = value
    return base, best


@njit(cache=True, fastmath=False)
def sparse_candidates(reduced, pivots, nonpivots, rhs, costs, order):
    """Score only support bits, retaining ascending original-column addition."""
    n = len(costs)
    rank = len(pivots)
    base = np.zeros(n, dtype=np.uint8)
    for i in range(rank):
        base[pivots[i]] = rhs[i]
    best = base.copy()
    if order == 0:
        return base, best
    pivot_order = np.argsort(pivots)
    minimum = 0.0
    for column in range(n):
        if base[column]:
            minimum += costs[column]
    for selected in range(len(nonpivots)):
        column = nonpivots[selected]
        weight = 0.0
        inserted = False
        for pos in range(rank):
            i = pivot_order[pos]
            if not inserted and column < pivots[i]:
                weight += costs[column]
                inserted = True
            if rhs[i] ^ reduced[i, column]:
                weight += costs[pivots[i]]
        if not inserted:
            weight += costs[column]
        if weight < minimum:
            minimum = weight
            best = np.zeros(n, dtype=np.uint8)
            best[column] = 1
            for i in range(rank):
                best[pivots[i]] = rhs[i] ^ reduced[i, column]
    for i in range(min(order, len(nonpivots))):
        for j in range(i+1, min(order, len(nonpivots))):
            first, second = nonpivots[i], nonpivots[j]
            low, high = min(first, second), max(first, second)
            low_added = high_added = False
            weight = 0.0
            for pos in range(rank):
                row = pivot_order[pos]
                column = pivots[row]
                if not low_added and low < column:
                    weight += costs[low]
                    low_added = True
                if not high_added and high < column:
                    weight += costs[high]
                    high_added = True
                if rhs[row] ^ reduced[row, first] ^ reduced[row, second]:
                    weight += costs[column]
            if not low_added:
                weight += costs[low]
            if not high_added:
                weight += costs[high]
            if weight < minimum:
                minimum = weight
                best = np.zeros(n, dtype=np.uint8)
                best[first] = best[second] = 1
                for row in range(rank):
                    best[pivots[row]] = rhs[row] ^ reduced[row, first] ^ reduced[row, second]
    return base, best


def linear_sweep(checks, syndrome, llr, costs, order=2, sparse_support=False):
    column_order = native_order(llr)
    reduced = np.array(checks, dtype=np.uint8, order="C", copy=True)
    rhs = np.asarray(syndrome, dtype=np.uint8).copy()
    rank = 0
    pivots = []
    nonpivots = []
    for column in column_order:
        selected = np.flatnonzero(reduced[rank:, column])
        if not len(selected):
            nonpivots.append(column)
            continue
        row = rank+int(selected[0])
        if row != rank:
            reduced[[rank, row]] = reduced[[row, rank]]
            rhs[[rank, row]] = rhs[[row, rank]]
        other = np.flatnonzero(reduced[:, column])
        other = other[other != rank]
        reduced[other] ^= reduced[rank]
        rhs[other] ^= rhs[rank]
        pivots.append(column)
        rank += 1
        if rank == reduced.shape[0]:
            nonpivots.extend(column_order[len(pivots)+len(nonpivots):])
            break
    assert not rhs[rank:].any(), "Syndrome outside check image requires native fallback"
    score = sparse_candidates if sparse_support else candidates
    return score(reduced[:rank], np.asarray(pivots, dtype=np.int64),
        np.asarray(nonpivots, dtype=np.int64), rhs[:rank], costs, order)


class LinearOsd:
    def __init__(self, check, prior, max_iter=50, order=2, exact_bp=False, prefix_reuse=False, zero_shortcut=False, sparse_support=False):
        self.check = check.toarray().astype(np.uint8)
        self.bp = ExactSerialBp(check, prior, max_iter, prefix_reuse=prefix_reuse, zero_shortcut=zero_shortcut) if exact_bp else BpDecoder(check,
            channel_probs=prior, bp_method="product_sum", schedule="serial",
            max_iter=max_iter, input_vector_type="syndrome")
        self.order = order
        self.sparse_support = sparse_support
        runtime = ctypes.CDLL("ucrtbase")
        runtime.log.argtypes = (ctypes.c_double,)
        runtime.log.restype = ctypes.c_double
        self.costs = np.array([runtime.log(1/float(p)) for p in prior], dtype=np.float64)
        self.osd0_decoding = None
        self.osdw_decoding = None

    def decode(self, syndrome):
        value = self.bp.decode(syndrome)
        if self.bp.converge:
            return value
        self.osd0_decoding, self.osdw_decoding = linear_sweep(self.check, syndrome, self.bp.log_prob_ratios, self.costs, self.order, sparse_support=self.sparse_support)
        assert np.array_equal(self.check @ self.osdw_decoding % 2, syndrome)
        return self.osdw_decoding
