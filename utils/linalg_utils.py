'''
Created on June 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import scipy.sparse as sps
import warnings
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
