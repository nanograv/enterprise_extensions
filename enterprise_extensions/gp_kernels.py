# -*- coding: utf-8 -*-

import numpy as np
from enterprise.signals import signal_base, utils

__all__ = ['linear_interp_basis_dm',
           'linear_interp_basis_freq',
           'dmx_ridge_prior',
           'periodic_kernel',
           'se_kernel',
           'se_dm_kernel',
           'get_tf_quantization_matrix',
           'tf_kernel',
           'sf_kernel',
           ]


# linear interpolation basis in time with nu^-2 scaling
@signal_base.function
def linear_interp_basis_dm(toas, freqs, dt=30*86400):

    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**2

    return U * Dm[:, None], avetoas


@signal_base.function
def linear_interp_basis_chromatic(toas, freqs, dt=30*86400, idx=4):
    """Linear interpolation basis in time with nu^-4 scaling"""
    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**idx

    return U * Dm[:, None], avetoas


@signal_base.function
def linear_interp_basis_freq(freqs, df=64):
    """Linear interpolation in radio frequency"""
    return utils.linear_interp_basis(freqs, dt=df)


@signal_base.function
def dmx_ridge_prior(avetoas, log10_sigma=-7):
    """DMX-like signal with Gaussian prior"""
    sigma = 10**log10_sigma
    return sigma**2 * np.ones_like(avetoas)


@signal_base.function
def periodic_kernel(avetoas, log10_sigma=-7, log10_ell=2,
                    log10_gam_p=0, log10_p=0):
    """Quasi-periodic kernel for DM"""
    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    p = 10**log10_p * 3.16e7
    gam_p = 10**log10_gam_p
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2) + d
    return K


@signal_base.function
def se_kernel(avefreqs, log10_sigma=-7, log10_lam=3):
    """Squared-exponential kernel for FD"""
    tm = np.abs(avefreqs[None, :] - avefreqs[:, None])

    lam = 10**log10_lam
    sigma = 10**log10_sigma
    d = np.eye(tm.shape[0]) * (sigma/500)**2
    return sigma**2 * np.exp(-tm**2/2/lam) + d


@signal_base.function
def se_dm_kernel(avetoas, log10_sigma=-7, log10_ell=2):
    """Squared-exponential kernel for DM"""
    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # Convert everything into seconds
    l = 10**log10_ell * 86400
    sigma = 10**log10_sigma
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2) + d
    return K


@signal_base.function
def get_tf_quantization_matrix(toas, freqs, dt=30*86400, df=None, dm=False, dm_idx=2):
    """
    Quantization matrix in time and radio frequency to cut down on the kernel
    size.
    """
    if df is None:
        dfs = [(600, 1000), (1000, 1900), (1900, 3000), (3000, 5000)]
    else:
        fmin = freqs.min()
        fmax = freqs.max()
        fs = np.arange(fmin, fmax+df, df)
        dfs = [(fs[ii], fs[ii+1]) for ii in range(len(fs)-1)]

    Us, avetoas, avefreqs, masks = [], [], [], []
    for rng in dfs:
        mask = np.logical_and(freqs>=rng[0], freqs<rng[1])
        if any(mask):
            masks.append(mask)
            U, _ = utils.create_quantization_matrix(toas[mask],
                                                    dt=dt, nmin=1)
            avetoa = np.array([toas[mask][idx.astype(bool)].mean()
                               for idx in U.T])
            avefreq = np.array([freqs[mask][idx.astype(bool)].mean()
                                for idx in U.T])
            Us.append(U)
            avetoas.append(avetoa)
            avefreqs.append(avefreq)

    nc = np.sum(U.shape[1] for U in Us)
    U = np.zeros((len(toas), nc))
    avetoas = np.concatenate(avetoas)
    idx = np.argsort(avetoas)
    avefreqs = np.concatenate(avefreqs)
    nctot = 0
    for ct, mask in enumerate(masks):
        Umat = Us[ct]
        nn = Umat.shape[1]
        U[mask, nctot:nn+nctot] = Umat
        nctot += nn

    if dm:
        weights = (1400/freqs)**dm_idx
    else:
        weights = np.ones_like(freqs)

    return U[:, idx] * weights[:, None], {'avetoas': avetoas[idx],
                                          'avefreqs': avefreqs[idx]}


@signal_base.function
def tf_kernel(labels, log10_sigma=-7, log10_ell=2, log10_gam_p=0,
              log10_p=0, log10_ell2=4, log10_alpha_wgt=0):
    """
    The product of a quasi-periodic time kernel and
    a rational-quadratic frequency kernel.
    """
    avetoas = labels['avetoas']
    avefreqs = labels['avefreqs']

    r = np.abs(avetoas[None, :] - avetoas[:, None])
    r2 = np.abs(avefreqs[None, :] - avefreqs[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    l2 = 10**log10_ell2
    p = 10**log10_p * 3.16e7
    gam_p = 10**log10_gam_p
    alpha_wgt = 10**log10_alpha_wgt

    d = np.eye(r.shape[0]) * (sigma/500)**2
    Kt = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2)
    Kv = (1+r2**2/2/alpha_wgt/l2**2)**(-alpha_wgt)

    return Kt * Kv + d


@signal_base.function
def sf_kernel(labels, log10_sigma=-7, log10_ell=2,
              log10_ell2=4, log10_alpha_wgt=0):
    """
    The product of a squared-exponential time kernel and
    a rational-quadratic frequency kernel.
    """
    avetoas = labels['avetoas']
    avefreqs = labels['avefreqs']

    r = np.abs(avetoas[None, :] - avetoas[:, None])
    r2 = np.abs(avefreqs[None, :] - avefreqs[:, None])

    # Convert everything into seconds
    l = 10**log10_ell * 86400
    sigma = 10**log10_sigma
    l2 = 10**log10_ell2
    alpha_wgt = 10**log10_alpha_wgt

    d = np.eye(r.shape[0]) * (sigma/500)**2
    Kt = sigma**2 * np.exp(-r**2/2/l**2)
    Kv = (1+r2**2/2/alpha_wgt/l2**2)**(-alpha_wgt)

    return Kt * Kv + d
