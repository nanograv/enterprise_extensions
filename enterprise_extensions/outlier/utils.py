#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities module containing standalone functions used in pulsar data
preparation and likelihood calculations.

Requirements:
    numpy
    enterprise
"""

import numpy as np

import enterprise.constants as const
from enterprise.signals.utils import create_quantization_matrix


def d_powerlaw(lAmp, Si, Tmax, freqs, ntotfreqs=None):
    """Returns the derivative of power spectral density given power-law signal
    parameters

    :param lAmp: Log10 of the power-law amplitude
    :param Si: Spectral index
    :param Tmax: max(TOA) - min(TOA) for data range
    :param freqs: Frequencies of all bins
    :param ntotfreqs: Total number of frequency bins, defaults to None
    :return: (len(freqs) x 3) array containing derivatives of PSD wrt signal
        parameters
    """
    if ntotfreqs is None:
        ntotfreqs = len(freqs)

    freqpy = freqs * const.yr
    d_mat = np.zeros((ntotfreqs, 3))

    d_mat[0:len(freqs), 0] = (2*np.log(10)*10**(2*lAmp) * const.yr**3 / \
                                             (12*np.pi*np.pi * Tmax)) * \
                                             freqpy ** (-Si)
    d_mat[0:len(freqs), 1] = -np.log(freqpy)*(10**(2*lAmp) * const.yr**3 / \
                                             (12*np.pi*np.pi * Tmax)) * \
                                             freqpy ** (-Si)
    d_mat[0:len(freqs), 2] = 0.0

    return d_mat


def argsortTOAs(toas, flags):
    """Sort TOA vector by absolute TOA and backend flags

    :param array toas: Array of pulsar TOAS
    :param array flags: N TOA length array of backend flags
    :return: N TOA length array of indices to sort TOA vector
    """
    U, _ = create_quantization_matrix(toas, nmin=1)
    isort = np.argsort(toas, kind='mergesort')
    flagvals = list(set(flags))

    for _, col in enumerate(U.T):
        for flag in flagvals:
            flagmask = (flags[isort] == flag)
            if np.sum(col[isort][flagmask]) > 1:
                colmask = col[isort].astype(np.bool)
                epmsk = flagmask[colmask]
                epinds = np.flatnonzero(epmsk)
                if len(epinds) == epinds[-1] - epinds[0] + 1:
                    pass
                else:
                    episort = np.argsort(flagmask[colmask], kind='mergesort')
                    isort[colmask] = isort[colmask][episort]
            else:
                pass

    return isort


def set_Uindslc(Umat, Uind):
    """Filter quantization matrix by removing observing epochs containing <4
    TOAs. This is to avoid issues when computing gradients.

    :param Umat: Quantization matrix mapping TOAs to observing epochs
    :param Uind: List of slice objects for non-zero elements of `Umat`
    :return: Quantization matrix with short observing epochs removed and array
        containing start and stop indices of corresponding slice objects
    """
    Uinds = []
    for ind in Uind:
        Uinds.append((ind.start, ind.stop))

    smallepochs = []
    for ii, elem in enumerate(Uind):
        if elem.stop - elem.start < 4:
            smallepochs.append(ii)

    Uindslc = np.array(Uinds, dtype=np.int)
    Umatslc = np.delete(Umat, smallepochs, axis=1)

    Uindslc = np.delete(Uindslc, smallepochs, axis=0)
    Umat = Umatslc

    return Umat, Uindslc
