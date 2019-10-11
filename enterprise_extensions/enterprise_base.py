# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.stats
from collections import OrderedDict

import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const

@signal_base.function
def createfourierdesignmatrix_chromatic(toas, freqs, nmodes=30, Tspan=None,
                                        logf=False, fmin=None, fmax=None,
                                        idx=4):

    """
    Construct Scattering-variation fourier design matrix.

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param idx: Index of chromatic effects

    :return: F: Chromatic-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = utils.createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf,
        fmin=fmin, fmax=fmax)

    # compute the DM-variation vectors
    Dm = (1400/freqs) ** idx

    return F * Dm[:, None], Ffreqs


@signal_base.function
def powerlaw_genmodes(f, log10_A=-16, gamma=5, components=2, wgts=None):
    if wgts is not None:
        df = wgts**2
    else:
        df = np.diff(np.concatenate((np.array([0]), f[::components])))
    return ((10**log10_A)**2 / 12.0 / np.pi**2 *
            const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, components))


@signal_base.function
def free_spectrum(f, log10_rho=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10**(2*np.array(log10_rho)), 2)

@signal_base.function
def t_process(f, log10_A=-15, gamma=4.33, alphas=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    alphas = np.ones_like(f) if alphas is None else np.repeat(alphas, 2)
    return utils.powerlaw(f, log10_A=log10_A, gamma=gamma) * alphas

@signal_base.function
def t_process_adapt(f, log10_A=-15, gamma=4.33, alphas_adapt=None, nfreq=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    if alphas_adapt is None:
        alpha_model = np.ones_like(f)
    else:
        if nfreq is None:
            alpha_model = np.repeat(alphas_adapt, 2)
        else:
            alpha_model = np.ones_like(f)
            alpha_model[2*int(np.rint(nfreq))] = alphas_adapt
            alpha_model[2*int(np.rint(nfreq))+1] = alphas_adapt

    return utils.powerlaw(f, log10_A=log10_A, gamma=gamma) * alpha_model

def InvGammaPrior(value, alpha=1, gamma=1):
    """Prior function for InvGamma parameters."""
    return scipy.stats.invgamma.pdf(value, alpha, scale=gamma)

def InvGammaSampler(alpha=1, gamma=1, size=None):
    """Sampling function for Uniform parameters."""
    return scipy.stats.invgamma.rvs(alpha, scale=gamma, size=size)

def InvGamma(alpha=1, gamma=1, size=None):
    """Class factory for Inverse Gamma parameters."""
    class InvGamma(parameter.Parameter):
        _size = size
        _prior = parameter.Function(InvGammaPrior, alpha=alpha, gamma=gamma)
        _sampler = staticmethod(InvGammaSampler)
        _alpha = alpha
        _gamma = gamma

        def __repr__(self):
            return '"{}": InvGamma({},{})'.format(self.name, alpha, gamma) \
                + ('' if self._size is None else '[{}]'.format(self._size))

    return InvGamma

@signal_base.function
def turnover_knee(f, log10_A, gamma, lfb, lfk, kappa, delta):
    """
    Generic turnover spectrum with a high-frequency knee.
    :param f: sampling frequencies of GWB
    :param A: characteristic strain amplitude at f=1/yr
    :param gamma: negative slope of PSD around f=1/yr (usually 13/3)
    :param lfb: log10 transition frequency at which environment dominates GWs
    :param lfk: log10 knee frequency due to population finiteness
    :param kappa: smoothness of turnover (10/3 for 3-body stellar scattering)
    :param delta: slope at higher frequencies
    """
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    hcf = (10**log10_A * (f / const.fyr) ** ((3-gamma) / 2) *
            (1.0 + (f / 10**lfk)) ** delta / np.sqrt(1 + (10**lfb / f) ** kappa))
    return hcf**2 / 12 / np.pi**2 / f**3 * np.repeat(df, 2)

@signal_base.function
def broken_powerlaw(f, log10_A, gamma, delta, log10_fb, kappa=0.1):
    """
    Generic broken powerlaw spectrum.
    :param f: sampling frequencies
    :param A: characteristic strain amplitude [set for gamma at f=1/yr]
    :param gamma: negative slope of PSD for f > f_break [set for comparison at f=1/yr (default 13/3)]
    :param delta: slope for frequencies < f_break
    :param log10_fb: log10 transition frequency at which slope switches from gamma to delta
    :param kappa: smoothness of transition (Default = 0.1)
    """
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    hcf = (10**log10_A * (f / const.fyr) ** ((3-gamma) / 2) *
          (1 + (f / 10**log10_fb) ** (1/kappa)) ** (kappa * (gamma - delta) / 2))
    return hcf**2 / 12 / np.pi**2 / f**3 * np.repeat(df, 2)
