'''
Created on Jun 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import math
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

import utils.linalg_utils as utils


def fiedler_of_component_spectra(in_mesh, in_lb, thres):
    spectra, multi_cc = in_lb.multi_component_spectra(2, thres)
    n_cc = len(multi_cc)
    aggregate_color = np.zeros((in_mesh.num_vertices, 1))
    for i in xrange(n_cc):
        nodes = multi_cc[i]
        if spectra[i]:
            magic_color = utils.scale(spectra[i][1][:, -1]**2)
            aggregate_color[nodes] = magic_color.reshape(len(nodes), 1)
    return aggregate_color[:, 0]


def hks_of_component_spectra(in_mesh, in_lb, area_type, percent_of_eigs, time_horizon, min_nodes=None, min_eigs=None, max_eigs=None):
    spectra, multi_cc = in_lb.multi_component_spectra(in_mesh, area_type, percent_of_eigs,
                                                      min_nodes=min_nodes, min_eigs=min_eigs, max_eigs=max_eigs)
    n_cc = len(multi_cc)
    aggregate_color = np.zeros((in_mesh.num_vertices, 1))
    for i in xrange(n_cc):
        nodes = multi_cc[i]
        if spectra[i]:
            evals = spectra[i][0]
            evecs = spectra[i][1].T
            pos_index = evals > 0
            if np.sum(pos_index) == 0:
                continue
            evals = evals[pos_index]
            evecs = evecs[pos_index, :]
            evecs = np.around(evecs, 2)
            utils.smooth_normal_outliers(evecs, 3)
            index = utils.non_homogeneous_vectors(evecs, 0.95)
            if len(index) >= 2:
                evecs = evecs[index, :]
                evals = evals[index] + 1        # Add 1 to make the division on time_samples strictly decreasing
                ts = hks_time_sample_generator(evals[0], evals[-1], time_horizon)
                sig = heat_kernel_signature(evals, evecs, ts)
                sig = sig / utils.l2_norm(sig, axis=0)
                sig = np.sum(sig, 1)
                magic_color = utils.scale(sig)
                aggregate_color[nodes] = magic_color.reshape(len(nodes), 1)

    return aggregate_color[:, 0]


def gaussian_curvature(in_mesh):
    # TODO: Use Gauss-Bonne theorem to compute the Genus.
    acc_map = in_mesh.triangles.ravel()
    angles = in_mesh.angles_of_triangles().ravel()
    acc_array = np.bincount(acc_map, weights=angles)
    gauss_curv = (2 * np.pi - acc_array)
    gauss_curv = gauss_curv.reshape(len(gauss_curv), 1)
    gauss_curv /= in_mesh.area_of_vertices()
    return gauss_curv


def mean_curvature(in_mesh, laplace_beltrami):
    N = in_mesh.normals_of_vertices()
    mean_curv = 0.5 * np.sum(N * (laplace_beltrami.W * in_mesh.vertices), 1)
    return mean_curv


def heat_kernel_signature(evals, evecs, time_horizon, verbose=False):
    if len(evals) != evecs.shape[0]:
        raise ValueError('Eigenvectors must have dimension = #eigen-vectors x nodes.')
    if verbose:
        print "Computing Heat Kernel Signature with %d eigen-pairs." % (len(evals),)

    n = evecs.shape[1]  # Number of nodes.
    e = np.e
    signatures = np.empty((n, len(time_horizon)))
    squared_evecs = np.square(evecs)
    squared_evecs = np.transpose(squared_evecs)

    for t, tp in enumerate(time_horizon):
        interm = e**(-tp * evals)
#         if interm < 10e-5:
#             signatures[:, t] = 0
#             continue
#         normFactor = 1 / np.sum(interm)
#         print normFactor
        for i in xrange(n):
            # signatures[i, t] = np.dot(interm, squared_evecs[i]) * normFactor
            signatures[i, t] = np.dot(interm, squared_evecs[i])
    return signatures


def hks_time_sample_generator(min_eval, max_eval, time_points):
    if max_eval <= min_eval or min_eval <= 0:
        raise ValueError('Two non-negative and sorted eigen-values are expected as input.')

    tmin = math.log(10) / max_eval
    tmax = math.log(10) / min_eval
    assert(tmax > tmin)
    stepsize = (math.log(tmax) - math.log(tmin)) / time_points
    logts = [math.log(tmin)]

    for _ in xrange(time_points - 1):
        logts.append(logts[-1] + stepsize)

    return [math.exp(i) for i in logts]


def wave_kernel_signature(evals, evecs, energies, sigma=1, verbose=False):
    if len(evals) != evecs.shape[0]:
        raise ValueError('Eigenvectors must have dimension = #eigen-vectors x nodes.')
    if verbose:
        print "Computing Wave Kernel Signature with %d eigen-pairs." % (len(evals),)
     
    n = evecs.shape[1]  # Number of nodes.
    e = math.exp(1)
    log = np.log
    signatures        = np.empty((n, len(energies)))
            
    squared_evecs = np.square(evecs)
    squared_evecs = np.transpose(squared_evecs)
    sigma = 2 * (sigma**2)
    for t, en in enumerate(energies):
        interm = e ** (-1 * ( ( (en - log(evals))**2) / sigma))
        norm_factor = 1 /np.sum(interm)
        for i in xrange (n):
            signatures[i, t] = np.dot(interm, squared_evecs[i]) * norm_factor

    assert(np.alltrue(signatures >= 0))
    return signatures


def wks_energy_generator(minEval, maxEval, timePoints, shrink=1):
    emin = math.log(minEval)   
    if shrink != 1:
        emax = math.log(maxEval) / float(shrink)
    else:
        emax = math.log(maxEval)

    emin = abs(emin)
    emax = abs(emax)

    if emax <= emin:
        print "Warning: too much shrink. - Will be set manually."
        emax = emin + 0.05 * emin

    delta = (emax - emin) / timePoints
    sigma = 10 * delta

    res = [emin]
    for _ in xrange(1, timePoints):
        res.append(res[-1] + delta)
    assert(utils.is_increasing(res))
    return res, sigma


def merge_component_spectra(in_mesh, in_lb, percent_of_eigs, merger=np.sum):
        spectra, multi_cc =  in_lb.multi_component_spectra(in_mesh, percent=percent_of_eigs)            
        n_cc = len(multi_cc)
        signature = np.zeros((in_mesh.num_vertices, 1))        
        for i in xrange(n_cc):
            nodes         = multi_cc[i]            
            if spectra[i]:               
                magic_color = merger(spectra[i][1]**2, axis=1)                                                
                signature[nodes] = magic_color.reshape(len(nodes), 1)
        return signature[:,0]


def extrinsic_laplacian(in_mesh, num_eigs):
    V = in_mesh.vertices
    E = in_mesh.undirected_edges()
    vals = V[E[:,1]]
    Wx = sparse.csr_matrix((vals[:,0], (E[:,0], E[:,1])), shape=(in_mesh.num_vertices, in_mesh.num_vertices))
    Wy = sparse.csr_matrix((vals[:,1], (E[:,0], E[:,1])), shape=(in_mesh.num_vertices, in_mesh.num_vertices))
    Wz = sparse.csr_matrix((vals[:,2], (E[:,0], E[:,1])), shape=(in_mesh.num_vertices, in_mesh.num_vertices))
    _, evecsx = eigs(Wx, num_eigs, which='LM')
    _, evecsy = eigs(Wy, num_eigs, which='LM')
    _, evecsz = eigs(Wz, num_eigs, which='LM')
    evecsx = np.sum(evecsx.real, axis=1)
    evecsy = np.sum(evecsy.real, axis=1)
    evecsz = np.sum(evecsz.real, axis=1)
    return np.vstack((evecsx, evecsy, evecsz))
