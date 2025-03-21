"""This file contains re-usable helper functions that are called throughout the ib-annealing process. These are heavily
influenced by the embo repository (https://gitlab.com/epiasini/embo)

@authors Lindsay Skinner (skinnel@uw.edu)
"""

import numpy as np
from scipy.stats import entropy
from typing import Tuple

def get_entropy(p: np.array, axis: int = 0, eps: float = 0.00001) -> float:
    """Computes the entropy of the provided probability distribution.

    Parameters
    ----------
    p
        The distribution for which the entropy is being calculated. Represented empirically as a numpy array.
    axis
        In cases where p is multidimensional, indicates which axis the entropy should be computed along.
    eps
        The equality threshold used to determine if a distribution is properly normalized.

    Returns
    -------
    entropy
        The value of the computed entropy (in bits).
    """

    # # Normalize p, if necessary
    # if np.abs(p.sum(axis=axis) - 1.0) > eps:
    #     p = p / np.sum(p, axis=axis, keepdims=True)

    # Calculate the entropy
    entropy_val = entropy(pk=p, base=2, axis=axis)

    return entropy_val

def get_kl_divergence(p: np.array, q: np.array, axis: int = 0, eps: float = 0.00001) -> float:
    """Computes the KL divergence of the provided distributions.

    Notes: i.e. D[p||q]

    Parameters
    ----------
    p
        The distribution against which q is compared to calculate the KL divergence. Represented empirically as a
        numpy array.
    q
        The distribution being compared to p in order to calculate the KL divergence. Represented empirically as a
        numpy array.
    axis
        In cases where p and q are multidimensional, indicates which axis the KL divergence should be computed along.
    eps
        The equality threshold used to determine if a distribution is properly normalized.

    Returns
    -------
    kl_div
        The value of the computed KL divergence (in bits).
    """

    # # Normalize p, if necessary
    # if np.abs(p.sum(axis=axis) - 1.0) > eps:
    #     p = p / np.sum(p, axis=axis, keepdims=True)
    #
    # # Normalize q, if necessary
    # if np.abs(q.sum(axis=axis) - 1.0) > eps:
    #     q = q / np.sum(q, axis=axis, keepdims=True)

    # Calculate the KL divergence
    kl_div = entropy(pk=p, qk=q, base=2, axis=axis)

    return kl_div

def get_mutual_information(p: np.array, q: np.array, p_q: np.array, axis: int = 0, eps: float = 0.0001) -> float:
    """Computes the mutual information between the probability distributions p and q. 
    
    Parameters
    ----------
    p
        One of the distributions for which mutual information is to be calculated. Represented empirically as a numpy 
        array.
    q
        The other distribution for which mutual information is to be calculated. Represented empirically as a numpy 
        array.
    p_q
        The distribution of variable one, whose probability distribution is defined by p, conditioned on variable 2,
        whose probability distributio is defined by q.
    axis
        In cases where p and q are multidimensional, indicates which axis the mutual information should be computed
        along.
    eps
        The equality threshold used to determine if a distribution is properly normalized.

    Returns
    -------
    mutual_information
        The value of the computed mutual information between p and q.
    """

    # Normalize p, if necessary
    if np.abs(p.sum(axis=axis) - 1.0) > eps:
        p = p / np.sum(p, axis=axis, keepdims=True)

    # Normalize q, if necessary
    if np.abs(q.sum(axis=axis) - 1.0) > eps:
        q = q / np.sum(q, axis=axis, keepdims=True)

    # Calculate the conditional distribution
    joint_dist = p_q * q[np.newaxis, :]

    # Calculate the joint distribution
    outer_prod = p[:, np.newaxis] * q[np.newaxis, :]

    # Calculate the mutual information
    mutual_information = get_kl_divergence(p=joint_dist, q=outer_prod, axis=None)

    return mutual_information

# Taken from Embo, modified to match format
def compute_upper_bound(IX: np.array, IY: np.array) -> Tuple[np.array, np.array]:
    """Remove all points in an IB sequence that would make it nonmonotonic.

    This is a post-processing step that is needed after computing an
    IB sequence (defined as a sequence of (IX, IY) pairs),
    to remove the random fluctuations in the result induced by the AB
    algorithm getting stuck in local minima.

    Parameters
    ----------
    IX
        I(M:X) values
    IY
        I(M:Y) values

    Returns
    -------
    ib_bound
        (I(M:X), I(M:Y)) coordinates of the IB bound after ensuring monotonic progression (with increasing beta) in both
        coordinates.
    selected_ids
        The indices of the points selected to ensure monotonic progression

    """
    points = np.vstack((IX, IY)).T
    selected_ids = [0]

    for idx in range(1, points.shape[0]):
        if points[idx, 0] > points[selected_ids[-1], 0] and points[idx, 1] >= points[selected_ids[-1], 1]:
            selected_ids.append(idx)

    ib_bound = points[selected_ids, :]

    return ib_bound, selected_ids

def get_joint_dist(x: np.array, y: np.array, axis: int = None) -> np.array:
    """Calculates the joint distribution between provided distributions x and y. Assumes x and y are ordered so that
    the probability at location i in each distribution correspond to the same observation.

    Parameters
    ----------
    x
        The first array, defining a distribution over which we will calculate the joint distribution with y.
    y
        The second array, defining a distribution over which we will calculate the joint distribution with x.

    Returns
    -------
    pxy
        The joint distribution of x and y.
    """

    assert x.shape[0] == y.shape[0], 'x and y must contain same number of observations'

    # Get joint distribution via counts
    x_unique, x = np.unique(x, return_inverse=True, axis=axis)
    y_unique, y = np.unique(y, return_inverse=True, axis=axis)
    n_x = x_unique.size
    n_y = y_unique.size
    pxy = np.zeros((n_x, n_y))
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        pxy[xi, yi] += 1
    pxy = pxy / pxy.sum()

    return pxy

# def get_accuracy(pu, pm_u):
#
#     # pm = np.zeros[pm_u.shape[1]]
#     # for ui in range(len(pu)):
#     #     pm += pu[ui] * pm_u[:, ui]
#     pm = pm_u @ pu
#
#     # for ui in range(len(pu)):
#     #     for mi in range(pm_u.shape[1]):
#     #         pu[ui] * pm_u[mi, ui] * np.log((pm_u[mi, ui] / pm[mi]))
#     acc = (pu[np.newaxis, :] * pm_u) * np.log(pm_u / pm[:, np.newaxis])
#     return acc

# def get_accuracy2(pm, pu_m):
#
#     pu = np.zeros(pu_m.shape[0])
#     for mi in range(len(pm)):
#         val = pm[mi] * pu_m[:, mi]
#         pu = pu + val
#
#     # pm = np.zeros(pm_u.shape[1])
#     # for ui in range(len(pu)):
#     #     pm += pu[ui] * pm_u[:, ui]
#
#     #pu = pu_m @ pm
#
#     acc = 0.0
#     for mi in range(len(pm)):
#         for ui in range(pu_m.shape[1]):
#             acc += pm[mi] * pu_m[ui, mi] * np.log((pu_m[ui, mi] / pu[ui]))
#
#     # acc = 0.0
#     # for ui in range(len(pu)):
#     #     for mi in range(pm_u.shape[1]):
#     #         acc += pu[ui] * pm_u[mi, ui] * np.log((pm_u[mi, ui] / pm[mi]))
#
#     #acc = (pm[np.newaxis, :] * pu_m) * np.log(pu_m / pu[:, np.newaxis])
#
#     return acc

PRECISION = 1e-16

def marginal(pXY, axis=1):
    """:return pY (axis = 0) or pX (default, axis = 1)"""
    return pXY.sum(axis)

def conditional(pXY):
    """:return  pY_X """
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])

def joint(pY_X, pX):
    """:return  pXY """
    return pY_X * pX[:, None]

def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)

def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)

def MI(pXY):
    """ mutual information, I(X;Y) """
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)
