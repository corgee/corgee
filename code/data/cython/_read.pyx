STUFF = "Hi"  # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
# cython: language_level=3
import array
from cpython cimport array
cimport cython
from libc.string cimport strchr
cimport numpy as np
from libcpp.string cimport string

cdef extern from "_read_utils.h":
    void read_emb_file_parallel(string, string, string, int)

from six import b

np.import_array()

cdef bytes COMMA = u','.encode('ascii')
cdef bytes COLON = u':'.encode('ascii')
cdef bytes TAB = u'\t'.encode('ascii')
cdef Py_UCS4 HASH = u'#'

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_double_file(fin):
    cdef bytes line
    
    cdef int idx, line_num
    cdef Py_ssize_t i
    
    ids = []
    
    data1 = array.array("f")
    indices1 = array.array("l")
    indptr1 = array.array("l")
    
    data2 = array.array("f")
    indices2 = array.array("l")    
    indptr2 = array.array("l")

    array.resize_smart(indptr1, len(indptr1) + 1)
    indptr1[len(indptr1) - 1] = 0

    array.resize_smart(indptr2, len(indptr2) + 1)
    indptr2[len(indptr2) - 1] = 0

    line_num = 0
    for line in fin:
        temp = line.split(TAB, -1)

        if(len(temp) != 3):
            continue

        ids.append(temp[0].decode())

        features = temp[1].split()
        for i in xrange(0, len(features)):
            idx_s, value = features[i].split(COLON, 1)
            idx = int(idx_s)
    
            array.resize_smart(data1, len(data1) + 1)
            array.resize_smart(indices1, len(indices1) + 1)
            data1[len(data1) - 1] = float(value)
            indices1[len(indices1) - 1] = idx

        labels = temp[2].split()
        for i in xrange(0, len(labels)):
            idx_s, value = labels[i].split(COLON, 1)
            idx = int(idx_s)
    
            array.resize_smart(data2, len(data2) + 1)
            array.resize_smart(indices2, len(indices2) + 1)
            data2[len(data2) - 1] = float(value)
            indices2[len(indices2) - 1] = idx

        array.resize_smart(indptr1, len(indptr1) + 1)
        indptr1[len(indptr1) - 1] = len(data1)

        array.resize_smart(indptr2, len(indptr2) + 1)
        indptr2[len(indptr2) - 1] = len(data2)

        line_num += 1

    return (ids, data1, indices1, indptr1, data2, indices2, indptr2)

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_sparse_file(fin):
    cdef bytes line
    
    cdef int idx, line_num
    cdef Py_ssize_t i
    
    ids = []
    
    data = array.array("f")
    indices = array.array("l")
    indptr = array.array("l")

    array.resize_smart(indptr, len(indptr) + 1)
    indptr[len(indptr) - 1] = 0

    line_num = 0
    for line in fin:
        temp = line.split(TAB, -1)

        if(len(temp) != 2):
            continue

        ids.append(temp[0].decode())

        sparse = temp[1].split()
        for i in xrange(0, len(sparse)):
            idx_s, value = sparse[i].split(COLON, 1)
            idx = int(idx_s)
    
            array.resize_smart(data, len(data) + 1)
            array.resize_smart(indices, len(indices) + 1)
            data[len(data) - 1] = float(value)
            indices[len(indices) - 1] = idx

        array.resize_smart(indptr, len(indptr) + 1)
        indptr[len(indptr) - 1] = len(data)

        line_num += 1
    return (ids, data, indices, indptr)

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_sparse_scores_file(fin, num_scores):
    cdef bytes line

    cdef int idx, line_num
    cdef Py_ssize_t i

    ids = []

    data = array.array("f")
    indices = array.array("l")
    indptr = array.array("l")

    array.resize_smart(indptr, len(indptr) + 1)
    indptr[len(indptr) - 1] = 0

    line_num = 0
    for line in fin:
        temp = line.split(TAB, -1)

        if(len(temp) != 2):
            continue

        ids.append(temp[0].decode())

        sparse = temp[1].split()
        for i in xrange(0, len(sparse)):
            split_idx_scores = sparse[i].split(COLON)
            if len(split_idx_scores) != (num_scores+1):
                continue
            idx = int(split_idx_scores[0])

            array.resize_smart(data, len(data) + num_scores)
            array.resize_smart(indices, len(indices) + 1)
            for j in xrange(0, num_scores):
                data[len(data) - num_scores + j] = float(split_idx_scores[1 + j])
            indices[len(indices) - 1] = idx

        array.resize_smart(indptr, len(indptr) + 1)
        indptr[len(indptr) - 1] = len(indices)

        line_num += 1
    return (ids, data, indices, indptr)

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_seller_file(fin):
    cdef bytes line
    
    cdef int idx, line_num
    cdef Py_ssize_t i
    
    ids = []
    
    indices = array.array("l")
    indptr = array.array("l")

    array.resize_smart(indptr, len(indptr) + 1)
    indptr[len(indptr) - 1] = 0

    line_num = 0
    for line in fin:
        temp = line.split(TAB, -1)
        line_data = temp[1].strip()
        
        if(len(temp) != 2):
            continue

        ids.append(temp[0].decode())
        
        if(len(line_data)>0):
            sparse = line_data.split(COMMA)
            for i in xrange(0, len(sparse)):
                idx = int(sparse[i])
                array.resize_smart(indices, len(indices) + 1)
                indices[len(indices) - 1] = idx

        array.resize_smart(indptr, len(indptr) + 1)
        indptr[len(indptr) - 1] = len(indices)

        line_num += 1

    return (ids, indices, indptr)

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_embeddings(fin, dim):
    cdef array.array data, rows, cols
    cdef bytes line
    
    cdef int idx, line_num
    cdef Py_ssize_t i
    
    ids = []
    emb = array.array("f")

    line_num = 0
    for line in fin:
        line_split = line.split(TAB, -1)
        if(len(line_split) != 2):
            continue
        embs = line_split[1].split()
        if(len(embs) != dim):
            continue
        
        ids.append(line_split[0].decode())
        array.resize_smart(emb, len(emb)+dim)
        for i in xrange(0, dim):
            emb[len(emb)-dim+i] = float(embs[i])
        line_num += 1
    return ids, emb

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_embeddings_cpp(input_fname, ids_out, embs_out, dim):
    cdef string c_input_fname = str.encode(input_fname)
    cdef string c_ids_out = str.encode(ids_out)
    cdef string c_embs_out = str.encode(embs_out)
    read_emb_file_parallel(c_input_fname, c_ids_out, c_embs_out, dim)

@cython.boundscheck(False)
@cython.wraparound(False)
def _read_embeddings_batched(fin, dim, batch_size):
    cdef array.array data, rows, cols
    cdef bytes line
    
    cdef int idx, line_num
    cdef Py_ssize_t i
    
    ids = []    
    emb = array.array("f")

    line_num = 0
    for line in fin:
        line_split = line.split(TAB, -1)

        if(len(line_split) != 2):
            continue
        embs = line_split[1].split()
        if(len(embs) != dim):
            continue
            
        ids.append(line_split[0].decode())
        array.resize_smart(emb, len(emb)+dim)
        for i in xrange(0, dim):
            emb[len(emb)-dim+i] = float(embs[i])
        line_num += 1
        if line_num == batch_size:
            return fin, ids, emb
    return fin, ids, emb