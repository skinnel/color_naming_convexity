""" Test to see if the MI, KL, Accuracy, Complexity, etc. definitions from Zaslavsky's repo cause different results."""

import logging
import os
from urllib.request import urlretrieve
import numpy as np
from scipy.special import logsumexp

PRECISION = 1e-16

# DISTRIBUTIONS

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


def marginalize(pY_X, pX):
    """:return  pY """
    return pY_X.T @ pX


def bayes(pY_X, pX):
    """:return pX_Y """
    pXY = joint(pY_X, pX)
    pY = marginalize(pY_X, pX)
    return np.where(pY > PRECISION, pXY.T / pY, 1 / pXY.shape[0])


def softmax(dxy, beta=1, axis=None):
    """:return
        axis = None: pXY propto exp(-beta * dxy)
        axis = 1: pY_X propto exp(-beta * dxy)
        axis = 0: pX_Y propto exp(-beta * dxy)
    """
    log_z = logsumexp(-beta * dxy, axis, keepdims=True)
    return np.exp(-beta * dxy - log_z)


# INFORMATIONAL MEASURES

def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """ mutual information, I(X;Y) """
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def DKL(p, q, axis=None):
    """ KL divergences, D[p||q] """
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(axis=axis)


def gNID(pW_X, pV_X, pX):
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)
    score = 1 - MI(pWV) / (np.max([MI(pWW), MI(pVV)]))
    return score


# IB UPDATE FUNCTIONS


def update_qw(pm: np.array, encoder: np.array) -> np.array:
    """Updates q_beta(w) in the IB iteration process. This is p_m in embo.

    Parameters
    ----------
    pm
        p(m), the marginal distribution of meaning m. This is px in embo.
    encoder
        q_beta(w|m), the distribution of words (w) conditioned on meaning m. This is pmx_c in embo.

    Returns
    -------
    qw
        The updated distribution of words, w. .
    """
    qw = encoder @ pm

    return qw


def update_pu(mwu, qw, encoder, meanings):

    # Get denominator
    denom = encoder @ meanings

    # Get numerator
    # num = mwu * qw[:, np.newaxis]
    num = mwu * qw[np.newaxis, :]

    pu = num / denom.T
    pu = pu.sum(axis=1)

    return pu


def update_mwu(qw: np.array, pm: np.array, pu_m: np.array, encoder: np.array) -> np.array:
    """Updates m^_w(u) in the IB iteration process. This is p_ym_c in embo.

    Parameters
    ----------
    qw
        q_beta(w), the marginal distribution of word w. This is pm in embo.
    pm
        p(m), the marginal distribution of meaning m. This is px in embo.
    pu_m
        p(u|m), the distribution of u (colors) conditioned on meaning m. This is pyx_c in embo.
    encoder
        q_beta(w|m), the distribution of w (words) conditioned on meaning m. This is pmx_c in embo.

    Returns
    -------
     mwu
        The beta-specific distribution, m^_w(w), updated during the most recent iteration.
    """
    # TODO: try switching this out so its sum_m decoder * m(u) (SI pg 3)
    # note that to get m(u) we do p(u|m)*p(m)
    mu = pu_m * pm[np.newaxis, :]
    puw = mu @ encoder.T
    #puw = pu_m * pm[np.newaxis, :] @ encoder.T
    mwu = puw / qw[np.newaxis, :]


    return mwu