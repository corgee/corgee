# Adapted from Kunal Dahiya's pyxclib
cimport cython
import numpy as np
cimport numpy as np


np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def rank_data(b):
    if b.size == 0:
        return np.array([], dtype=int)
    sorter = np.argsort(b, kind='mergesort')
    inv = np.empty(b.size, dtype=int)
    inv[sorter] = np.arange(sorter.size, dtype=int)
    return inv+1


@cython.boundscheck(False)
@cython.wraparound(False)
def _rank(data, indices, indptr):
    cdef Py_ssize_t num_rows = indptr.size - 1
    cdef Py_ssize_t idx
    cdef np.ndarray[np.int_t, ndim=1] rank = np.empty(data.size, dtype=int)
    for idx in range(num_rows):
        rank[indptr[idx]:indptr[idx+1]] = rank_data(-1*data[indptr[idx]:indptr[idx+1]])
    return rank


@cython.boundscheck(False)
@cython.wraparound(False)
def _topk(data, indices, indptr, k, pad_ind, pad_val):
    cdef Py_ssize_t num_rows = indptr.size - 1
    cdef Py_ssize_t idx, num_el, start_idx, end_idx
    cdef np.ndarray[np.int_t, ndim=2] ind = np.full((num_rows, k), pad_ind, int, 'C')
    cdef np.ndarray[np.float64_t, ndim=2] val = np.full((num_rows, k), pad_val, float, 'C')
    for idx in range(num_rows):
        start_idx = indptr[idx]
        end_idx = indptr[idx+1]
        num_el = min(k, end_idx - start_idx)
        ind[idx, :num_el] = indices[start_idx:end_idx][np.argsort(-1*data[start_idx:end_idx])[:num_el]]
        val[idx, :num_el] = data[start_idx:end_idx][np.argsort(-1*data[start_idx:end_idx])[:num_el]]
    return ind, val
