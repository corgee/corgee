from main.xclib._sparse import _rank, _topk
from scipy.sparse import csr_matrix


def binarize(X, copy=False):
    """Binarize a sparse matrix
    """
    if copy:
        X = X.copy()
    X.data.fill(1)
    return X


def rank(X):
    '''Rank of each element in decreasing order (per-row)
    Ranking will start from one (with zero at zero entries)
    '''
    ranks = _rank(X.data, X.indices, X.indptr)
    return csr_matrix((ranks, X.indices, X.indptr), shape=X.shape)


def topk(X, k, pad_ind, pad_val, return_values=False, dtype='float32'):
    """Get top-k indices and values for a sparse (csr) matrix
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    return_values: boolean, optional, default=False
        Return topk values or not
    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    ind, val = _topk(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    if return_values:
        return ind, val.astype(dtype)
    else:
        return ind


def retain_topk(X, copy=True, k=5):
    """Retain topk values of each row and make everything else zero
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    copy: boolean, optional, default=True
        copy data or change original array
    k: int, optional, default=5
        retain these many values

    Returns:
    --------
    X: csr_matrix
        sparse mat with only k entries in each row
    """
    X.sort_indices()
    ranks = rank(X)
    if copy:
        X = X.copy()
    X.data[ranks.data > k] = 0.0
    X.eliminate_zeros()
    return X
