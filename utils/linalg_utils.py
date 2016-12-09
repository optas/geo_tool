'''
Created on Jun 14, 2016

@author:    Panayotes Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import scipy.sparse as sps
import warnings
from scipy.stats import mode
from numpy.linalg import matrix_rank

l2_norm = np.linalg.norm


def is_close(a, b, atol=0):
    sp_a = sps.issparse(a)
    sp_b = sps.issparse(b)
    if sp_a and sp_b:
        return (abs(a - b) > atol).nnz == 0
    elif not sp_a and not sp_b:
        return (np.allclose(a, b, atol=atol))
    else:
        return (abs(a - b) <= atol).all()


def is_symmetric(array, tolerance=10e-6):
    if sps.issparse(array):
        return (abs(array - array.T) > tolerance).nnz == 0
    else:
        return np.allclose(array, array.T, atol=tolerance, rtol=0)


def is_finite(array):
    if sps.issparse(array):
        array = array.tocoo().data
    return np.isfinite(array).all()


def is_orthogonal(array, axis=1, tolerance=10e-6):
    ''' axis - (optional, 0 or 1, default=1). If 0 checks for orthogonality of rows, else of columns.
    '''
    if axis == 0:
        gram_matrix = array.dot(array.T)
    else:
        gram_matrix = array.T.dot(array)

    if sps.issparse(gram_matrix):
        gram_matrix = gram_matrix.todense()
    np.fill_diagonal(gram_matrix, 0)
    return np.allclose(gram_matrix, np.zeros_like(gram_matrix), atol=tolerance, rtol=0)


def is_square_matrix(array):
    return array.ndim == 2 and array.shape[0] == array.shape[1]


def unique_rows(array):
    if array.ndim != 2:
        raise ValueError('Unique rows works with 2D arrays only.')
    array = np.ascontiguousarray(array)
    unique_a = np.unique(array.view([('', array.dtype)] * array.shape[1]))
    return unique_a.view(array.dtype).reshape((unique_a.shape[0], array.shape[1]))


def scale(array, vmin=0, vmax=1):
    if vmin >= vmax:
        raise ValueError('vmax must be strictly bigger than vmin.')
    amax = np.max(array)
    amin = np.min(array)
    if amax == amin:
        warnings.warn('Constant array cannot be scaled')
        return array
    res = vmax - (((vmax - vmin) * (amax - array)) / (amax - amin))

    cond_1 = np.all(abs(vmax - res) < 10e-5) and np.all(abs(res - vmin) > 10e-5)
    cond_2 = abs(np.max(res) - vmax) < 10e-5 and abs(np.min(res) - vmin) < 10e-5

    if not (cond_1 or cond_2):
            warnings.warn('Scaling failed in the accuracy of 10e-5.')

    return res


def is_increasing(l):
    return all(l[i] <= l[i + 1] for i in xrange(len(l) - 1))


def is_decreasing(l):
    return all(l[i] >= l[i + 1] for i in xrange(len(l) - 1))


def order_of_elements_after_deletion(num_elements, delete_index):
    '''
    Assuming a sequence of num_elements index in [0-num_elements-1] and a list of indices to be deleted from the sequence,
    creates the mapping from the remaining elements to their position in the new list created after the deletion takes place.
    '''
    delete_index = np.unique(delete_index)
    init_list = np.arange(num_elements)
    after_del = np.delete(init_list, delete_index)
    return {key: i for i, key in enumerate(after_del)}


def non_homogeneous_vectors(array, thres):
    index = []
    n = float(array.shape[1])
    for i, vec in enumerate(array):
        frac = mode(vec)[1][0] / n
        if frac < thres:
            index.append(i)
    return index


def smooth_normal_outliers(array, dev):
    '''In each row of array, finds outlier elements transforms their values. An outlier in row[i], is any
    element of that row that is in magnitude bigger than `dev` times \sigma[i] + \mu[i], where ...
    '''
    stds = np.std(array, axis=1)
    means = np.mean(array, axis=1)
    for i in xrange(array.shape[0]):
        outliers = abs(array[i]) > dev * stds[i] + means[i]
        array[i, outliers] = means[i]


def are_coplanar(points):
    return matrix_rank(points) > 2


def accumarray(subs, val):
    ''' Matlab inspired function. 
    A = accumarray(subs,val) returns array A by accumulating elements of vector
    val using the subscripts subs. If subs is a column vector, then each element 
    defines a corresponding subscript in the output, which is also a column vector.
    The accumarray function collects all elements of val that have identical subscripts
    in subs and stores their sum in the location of A corresponding to that 
    subscript (for index i, A(i)=sum(val(subs(:)==i))).
    '''
    return np.bincount(subs, weights=val)


def sort_spectra(evals, evecs, transformer=None):
    if transformer is None:
        index = np.argsort(evals)  # Sort evals from smallest to largest
    else:
        index = np.argsort(transformer(evals))  # Sort evals from smallest to largest
    evals = evals[index]
    evecs = evecs[:, index]
    assert(is_increasing(evals))
    return evals, evecs
