# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.interpolate as interp
from enterprise.signals import signal_base
from enterprise import constants as const


@signal_base.function
def param_hd_orf(pos1, pos2, a=1.5, b=-0.25, c=0.5):
    '''Pre-factor parametrized Hellings & Downs spatial correlation function.

    :param: a, b, c:
        coefficients of H&D-like curve [default=1.5,-0.25,0.5].

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        params = [a,b,c]
        return params[0] * omc2 * np.log(omc2) + params[1] * omc2 + params[2]


@signal_base.function
def spline_orf(pos1, pos2, params):
    '''Agnostic spline-interpolated spatial correlation function. Spline knots
    are placed at edges, zeros, and minimum of H&D curve. Changing locations
    will require manual intervention to create new function.

    :param: params
        spline knot amplitudes.

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        # spline knots placed at edges, zeros, and minimum of H&D
        spl_knts = np.array([1e-3, 25.0, 49.3, 82.5,
                             121.8, 150.0, 180.0]) * np.pi/180.0
        omc2_knts = (1 - np.cos(spl_knts)) / 2
        finterp = interp.interp1d(omc2_knts, params, kind='cubic')

        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return finterp(omc2)


@signal_base.function
def bin_orf(pos1, pos2, params):
    '''Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.

    :param: params
        inter-pulsar correlation bin amplitudes.

    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        # bins in angsep space
        bins = np.array([1e-3, 30.0, 50.0, 80.0, 100.0,
                         120.0, 150.0, 180.0]) * np.pi/180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        idx = np.digitize(angsep, bins)
        return params[idx-1]


@signal_base.function
def zero_diag_bin_orf(pos1, pos2, params):
    '''Agnostic binned spatial correlation function. To be
    used in a "split likelihood" model with an additional common
    uncorrelated red process. The latter is necessary to regularize
    the overall Phi covariance matrix.

    :param: params
        inter-pulsar correlation bin amplitudes.

    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1e-20
    else:
        # bins in angsep space
        bins = np.array([1e-3, 30.0, 50.0, 80.0, 100.0,
                         120.0, 150.0, 180.0]) * np.pi/180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        idx = np.digitize(angsep, bins)
        return params[idx-1]


@signal_base.function
def zero_diag_hd(pos1, pos2):
    '''Off-diagonal Hellings & Downs spatial correlation function. To be
    used in a "split likelihood" model with an additional common uncorrelated
    red process. The latter is necessary to regularize the overall Phi
    covariance matrix.

    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1e-20
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5


@signal_base.function
def freq_hd(pos1, pos2, params):
    '''Frequency-dependent Hellings & Downs spatial correlation function.
    Implemented as a model that only enforces H&D inter-pulsar correlations
    after a certain number of frequencies in the spectrum. The first set of
    frequencies are uncorrelated.

    :param: params
        params[0] is the number of components in the stochastic process.
        params[1] is the frequency at which to start the H&D inter-pulsar
        correlations (indexing from 0).

    Reference: Taylor et al. (2017), https://arxiv.org/abs/1606.09180
    Author: S. R. Taylor (2020)
    '''
    nfreq = params[0]
    orf_ifreq = params[1]
    if np.all(pos1 == pos2):
        return np.ones(2*nfreq)
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        hd_coeff = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
        hd_coeff *= np.ones(2*nfreq)
        hd_coeff[:2*orf_ifreq] = 0.0
        return hd_coeff


@signal_base.function
def legendre_orf(pos1, pos2, params):
    '''Legendre polynomial spatial correlation function. Assumes process
    normalization such that autocorrelation signature is 1. A separate function
    is needed to use a "split likelihood" model with this Legendre process
    decoupled from the autocorrelation signature ("zero_diag_legendre_orf").

    :param: params
        Legendre polynomial amplitudes describing the Legendre series approximation
        to the inter-pulsar correlation signature.
        H&D coefficients are a_0=0, a_1=0, a_2=0.3125, a_3=0.0875, ...

    Reference: Gair et al. (2014), https://arxiv.org/abs/1406.4664
    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        costheta = np.dot(pos1, pos2)
        orf = np.polynomial.legendre.legval(costheta, params)
        return orf


@signal_base.function
def zero_diag_legendre_orf(pos1, pos2, params):
    '''Legendre polynomial spatial correlation function. To be
    used in a "split likelihood" model with an additional common uncorrelated
    red process. The latter is necessary to regularize the overall Phi
    covariance matrix.

    :param: params
        Legendre polynomial amplitudes describing the Legendre series approximation
        to the inter-pulsar correlation signature.
        H&D coefficients are a_0=0, a_1=0, a_2=0.3125, a_3=0.0875, ...

    Reference: Gair et al. (2014), https://arxiv.org/abs/1406.4664
    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1e-20
    else:
        costheta = np.dot(pos1, pos2)
        orf = np.polynomial.legendre.legval(costheta, params)
        return orf

@signal_base.function
def hd_orf(pos1, pos2):
    """Hellings & Downs spatial correlation function."""
    print(pos1,pos2)
    if np.all(pos1 == pos2):
        return 1
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

@signal_base.function
def dipole_orf(pos1, pos2):
    """Dipole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1 + 1e-5
    else:
        return np.dot(pos1, pos2)

@signal_base.function
def monopole_orf(pos1, pos2):
    """Monopole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1.0 + 1e-5
    else:
        return 1.0

@signal_base.function
def anis_orf(pos1, pos2, params, **kwargs):
    """Anisotropic GWB spatial correlation function."""

    anis_basis = kwargs["anis_basis"]
    psrs_pos = kwargs["psrs_pos"]
    lmax = kwargs["lmax"]

    psr1_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos1)][0]
    psr2_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos2)][0]

    clm = np.zeros((lmax + 1) ** 2)
    clm[0] = 2.0 * np.sqrt(np.pi)
    if lmax > 0:
        clm[1:] = params

    return sum(clm[ii] * basis for ii, basis in enumerate(anis_basis[: (lmax + 1) ** 2, psr1_index, psr2_index]))
    
@signal_base.function
def gw_monopole_orf(pos1, pos2):
    """
    GW-monopole Correlations. This phenomenological correlation pattern can be
    used in Bayesian runs as the simplest type of correlations.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1/2

@signal_base.function
def gw_dipole_orf(pos1, pos2):
    """
    GW-dipole Correlations.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1/2*np.dot(pos1, pos2)

@signal_base.function
def st_orf(pos1, pos2):
    """
    Scalar tensor correlations as induced by the breathing polarization mode of gravity.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1/8 * (3.0 + np.dot(pos1, pos2) )

@signal_base.function
def gt_orf(pos1, pos2, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        k = 1/2*(1-np.dot(pos1,pos2))
        return 1/8 * (3+np.dot(pos1,pos2)) + (1-tau)*3/4*k*np.log(k)
