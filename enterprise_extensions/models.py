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

from enterprise_extensions import model_utils
import enterprise_extensions.enterprise_base as eb

#### Extra model components not part of base enterprise ####

# timing model delay
@signal_base.function
def tm_delay(residuals, t2pulsar, tmparams_orig, tmparams, which='all'):
    """
    Compute difference in residuals due to perturbed timing model.

    :param residuals: original pulsar residuals from Pulsar object
    :param t2pulsar: libstempo pulsar object
    :param tmparams_orig: dictionary of TM parameter tuples, (val, err)
    :param tmparams: new timing model parameters, rescaled to be in sigmas
    :param which: option to have all or only named TM parameters varied

    :return: difference between new and old residuals in seconds
    """

    if which == 'all': keys = tmparams_orig.keys()
    else: keys = which

    # grab original timing model parameters and errors in dictionary
    orig_params = np.array([tmparams_orig[key] for key in keys])

    # put varying parameters into dictionary
    tmparams_rescaled = np.atleast_1d(np.double(orig_params[:,0] +
                                                tmparams * orig_params[:,1]))
    tmparams_vary = OrderedDict(zip(keys, tmparams_rescaled))

    # set to new values
    t2pulsar.vals(tmparams_vary)
    new_res = np.double(t2pulsar.residuals().copy())

    # remmeber to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params[:,0])))))

    # Return the time-series for the pulsar
    return new_res - residuals

# linear interpolation basis in time with nu^-2 scaling
@signal_base.function
def linear_interp_basis_dm(toas, freqs, dt=30*86400):

    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**2

    return U * Dm[:, None], avetoas

# linear interpolation basis in time with nu^-4 scaling
@signal_base.function
def linear_interp_basis_scattering(toas, freqs, dt=30*86400):

    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**4

    return U * Dm[:, None], avetoas

# linear interpolation in radio frequcny
@signal_base.function
def linear_interp_basis_freq(freqs, df=64):

    return utils.linear_interp_basis(freqs, dt=df)

# DMX-like signal with Gaussian prior
@signal_base.function
def dmx_ridge_prior(avetoas, log10_sigma=-7):
    sigma = 10**log10_sigma
    return sigma**2 * np.ones_like(avetoas)

# quasi-periodic kernel for DM
@signal_base.function
def periodic_kernel(avetoas, log10_sigma=-7, log10_ell=2, log10_gam_p=0, log10_p=0):

    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    p = 10**log10_p * 3.16e7
    gam_p = 10**log10_gam_p
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2) + d
    return K

# squared-exponential kernel for FD
@signal_base.function
def se_kernel(avefreqs, log10_sigma=-7, log10_lam=3):

    tm = np.abs(avefreqs[None, :] - avefreqs[:, None])

    lam = 10**log10_lam
    sigma = 10**log10_sigma
    d = np.eye(tm.shape[0]) * (sigma/500)**2
    return sigma**2 * np.exp(-tm**2/2/lam) + d

# squared-exponential kernel for DM
@signal_base.function
def se_dm_kernel(avetoas, log10_sigma=-7, log10_ell=2):

    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # Convert everything into seconds
    l = 10**log10_ell * 86400
    sigma = 10**log10_sigma
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2) + d
    return K

# quantization matrix in time and radio frequency to cut down on the kernel size.
@signal_base.function
def get_tf_quantization_matrix(toas, freqs, dt=30*86400, df=None, dm=False, dm_idx=2):
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

# kernel is the product of a quasi-periodic time kernel and
# a rational-quadratic frequency kernel.
@signal_base.function
def tf_kernel(labels, log10_sigma=-7, log10_ell=2, log10_gam_p=0,
              log10_p=0, log10_ell2=4, log10_alpha_wgt=0):

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

# kernel is the product of a squared-exponential time kernel and
# a rational-quadratic frequency kernel
@signal_base.function
def sf_kernel(labels, log10_sigma=-7, log10_ell=2,
              log10_ell2=4, log10_alpha_wgt=0):

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

@signal_base.function
def chrom_exp_decay(toas, freqs, log10_Amp=-7, sign_param=-1.0,
                    t0=54000, log10_tau=1.7, idx=2):
    """
    Chromatic exponential-dip delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau: 1/e time of exponential [s]
    :param log10_Amp: amplitude of dip
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0 *= const.day
    tau = 10**log10_tau * const.day
    ind = np.where(toas > t0)[0]
    wf = 10**log10_Amp * np.heaviside(toas - t0, 1)
    wf[ind] *= np.exp(- (toas[ind] - t0) / tau)

    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

@signal_base.function
def chrom_exp_cusp(toas, freqs, log10_Amp=-7, sign_param=-1.0,
                   t0=54000, log10_tau_pre=1.7, log10_tau_post=1.7,
                   symmetric=False, idx=2):
    """
    Chromatic exponential-cusp delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau_pre: 1/e time of exponential before peak [s]
    :param tau_post: 1/e time of exponential after peak[s]
    :param symmetric: whether or not tau_pre = tau_post
    :param log10_Amp: amplitude of cusp
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0 *= const.day
    if symmetric:
        tau = 10**log10_tau_pre * const.day
        ind_pre = np.where(toas < t0)[0]
        ind_post = np.where(toas > t0)[0]
        wf_pre = 10**log10_Amp * (1 - np.heaviside(toas - t0, 1))
        wf_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau)
        wf_post = 10**log10_Amp * np.heaviside(toas - t0, 1)
        wf_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau)
        wf = wf_pre + wf_post

    else:
        tau_pre = 10**log10_tau_pre * const.day
        tau_post = 10**log10_tau_post * const.day
        ind_pre = np.where(toas < t0)[0]
        ind_post = np.where(toas > t0)[0]
        wf_pre = 10**log10_Amp * (1 - np.heaviside(toas - t0, 1))
        wf_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau_pre)
        wf_post = 10**log10_Amp * np.heaviside(toas - t0, 1)
        wf_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau_post)
        wf = wf_pre + wf_post

    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

@signal_base.function
def chrom_dual_exp_cusp(toas, freqs, t0=54000, sign_param=-1.0,
                        log10_Amp_1=-7, log10_tau_pre_1=1.7, log10_tau_post_1=1.7,
                        log10_Amp_2=-7, log10_tau_pre_2=1.7, log10_tau_post_2=1.7,
                        symmetric=False, idx1=2, idx2=4):
    """
    Chromatic exponential-cusp delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau_pre: 1/e time of exponential before peak [s]
    :param tau_post: 1/e time of exponential after peak[s]
    :param symmetric: whether or not tau_pre = tau_post
    :param log10_Amp: amplitude of cusp
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0 *= const.day
    ind_pre = np.where(toas < t0)[0]
    ind_post = np.where(toas > t0)[0]
    if symmetric:
        tau_1 = 10**log10_tau_pre_1 * const.day
        wf_1_pre = 10**log10_Amp_1 * (1 - np.heaviside(toas - t0, 1))
        wf_1_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau_1)
        wf_1_post = 10**log10_Amp_1 * np.heaviside(toas - t0, 1)
        wf_1_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau_1)
        wf_1 = wf_1_pre + wf_1_post

        tau_2 = 10**log10_tau_pre_2 * const.day
        wf_2_pre = 10**log10_Amp_2 * (1 - np.heaviside(toas - t0, 1))
        wf_2_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau_2)
        wf_2_post = 10**log10_Amp_2 * np.heaviside(toas - t0, 1)
        wf_2_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau_2)
        wf_2 = wf_2_pre + wf_2_post

    else:
        tau_1_pre = 10**log10_tau_pre_1 * const.day
        tau_1_post = 10**log10_tau_post_1 * const.day
        wf_1_pre = 10**log10_Amp_1 * (1 - np.heaviside(toas - t0, 1))
        wf_1_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau_1_pre)
        wf_1_post = 10**log10_Amp_1 * np.heaviside(toas - t0, 1)
        wf_1_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau_1_post)
        wf_1 = wf_1_pre + wf_1_post

        tau_2_pre = 10**log10_tau_pre_2 * const.day
        tau_2_post = 10**log10_tau_post_2 * const.day
        wf_2_pre = 10**log10_Amp_2 * (1 - np.heaviside(toas - t0, 1))
        wf_2_pre[ind_pre] *= np.exp(- (t0 - toas[ind_pre]) / tau_2_pre)
        wf_2_post = 10**log10_Amp_2 * np.heaviside(toas - t0, 1)
        wf_2_post[ind_post] *= np.exp(- (toas[ind_post] - t0) / tau_2_post)
        wf_2 = wf_2_pre + wf_2_post

    return np.sign(sign_param) * ( wf_1 * (1400 / freqs) ** idx1 + wf_2 * (1400 / freqs) ** idx2)

@signal_base.function
def chrom_yearly_sinusoid(toas, freqs, log10_Amp=-7, phase=0, idx=2):
    """
    Chromatic annual sinusoid.

    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """

    wf = 10**log10_Amp * np.sin( 2 * np.pi * const.fyr * toas + phase)
    return wf * (1400 / freqs) ** idx

@signal_base.function
def chromatic_quad_basis(toas, freqs, idx=4):
    """
    Basis for chromatic quadratic function.

    :param idx: index of chromatic dependence

    :return ret: normalized quadratic basis matrix [Ntoa, 3]
    """
    ret = np.zeros((len(toas), 3))
    t0 = (toas.max() + toas.min()) / 2
    for ii in range(3):
        ret[:, ii] = (toas-t0) ** (ii) * (1400/freqs) ** idx
    norm = np.sqrt(np.sum(ret**2, axis=0))
    return ret/norm, np.ones(3)

@signal_base.function
def chromatic_quad_prior(toas):
    """
    Prior for chromatic quadratic function.

    :return prior: prior-range for quadratic coefficients
    """
    return np.ones(3) * 1e80

@signal_base.function
def dmx_delay(toas, freqs, dmx_ids, **kwargs):
    """
    Delay in DMX model of DM variations.

    :param dmx_ids: dictionary of DMX data for each pulsar from parfile
    :param kwargs: dictionary of enterprise DMX parameters

    :return wf: DMX signal
    """
    wf = np.zeros(len(toas))
    dmx = kwargs
    for dmx_id in dmx_ids:
        mask = np.logical_and(toas >= (dmx_ids[dmx_id]['DMX_R1'] - 0.01) * 86400.,
                              toas <= (dmx_ids[dmx_id]['DMX_R2'] + 0.01) * 86400.)
        wf[mask] += dmx[dmx_id] / freqs[mask]**2 / const.DM_K / 1e12
    return wf

#### Model component building blocks ####

def timing_block(tmparam_list=['RAJ', 'DECJ', 'F0', 'F1',
                               'PMRA', 'PMDEC', 'PX']):
    """
    Returns the timing model block of the model
    :param tmparam_list: a list of parameters to vary in the model
    """
    # default 5-sigma prior above and below the parfile mean
    tm_params = parameter.Uniform(-5.0, 5.0, size=len(tmparam_list))

    # timing model
    tm_func = tm_delay(tmparams=tm_params, which=tmparam_list)
    tm = deterministic_signals.Deterministic(tm_func, name='timing model')

    return tm

def white_noise_block(vary=False, inc_ecorr=False, gp_ecorr=False,
                      efac1=False, select='backend', name=None):
    """
    Returns the white noise block of the model:

        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system

    :param vary:
        If set to true we vary these parameters
        with uniform priors. Otherwise they are set to constants
        with values to be set later.
    :param inc_ecorr:
        include ECORR, needed for NANOGrav channelized TOAs
    :param gp_ecorr:
        whether to use the Gaussian process model for ECORR
    :param efac1:
        use a strong prior on EFAC = Normal(mu=1, stdev=0.1)
    """

    if select == 'backend':
        # define selection by observing backend
        backend = selections.Selection(selections.by_backend)
        # define selection by nanograv backends
        backend_ng = selections.Selection(selections.nanograv_backends)
    else:
        # define no selection
        backend = selections.Selection(selections.no_selection)


    # white noise parameters
    if vary:
        if efac1:
            efac = parameter.Normal(1.0, 0.1)
        else:
            efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5)
        if inc_ecorr:
            ecorr = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant()
        equad = parameter.Constant()
        if inc_ecorr:
            ecorr = parameter.Constant()

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac,
                                        selection=backend, name=name)
    eq = white_signals.EquadNoise(log10_equad=equad,
                                  selection=backend, name=name)
    if inc_ecorr:
        if gp_ecorr:
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                            selection=backend_ng, name=name)
        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,
                                                selection=backend_ng, name=name)

    # combine signals
    if inc_ecorr:
        s = ef + eq + ec
    elif not inc_ecorr:
        s = ef + eq

    return s

def red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=None,
                    components=30, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None,
                    break_flat=False, break_flat_fq=None):
    """
    Returns red noise model:
        1. Red noise modeled as a power-law with 30 sampling frequencies
    :param psd:
        PSD function [e.g. powerlaw (default), turnover, spectrum, tprocess]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of red noise
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess red noise
    :param coefficients: include latent coefficients in GP model?
    """
    # red noise parameters that are common
    if psd in ['powerlaw', 'powerlaw_genmodes', 'turnover',
               'tprocess', 'tprocess_adapt']:
        # parameters shared by PSD functions
        if prior == 'uniform':
            log10_A = parameter.LinearExp(-20, -11)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_A = parameter.Uniform(-20, -11)
            else:
                log10_A = parameter.Uniform(-20, -11)
        else:
            log10_A = parameter.Uniform(-20, -11)

        if gamma_val is not None:
            gamma = parameter.Constant(gamma_val)
        else:
            gamma = parameter.Uniform(0, 7)

        # different PSD function parameters
        if psd == 'powerlaw':
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        elif psd == 'powerlaw_genmodes':
            pl = powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=wgts)
        elif psd == 'turnover':
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            pl = utils.turnover(log10_A=log10_A, gamma=gamma,
                                 lf0=lf0, kappa=kappa)
        elif psd == 'tprocess':
            df = 2
            alphas = eb.InvGamma(df/2, df/2, size=components)
            pl = eb.t_process(log10_A=log10_A, gamma=gamma, alphas=alphas)
        elif psd == 'tprocess_adapt':
            df = 2
            alpha_adapt = eb.InvGamma(df/2, df/2, size=1)
            nfreq = parameter.Uniform(-0.5, 10-0.5)
            pl = eb.t_process_adapt(log10_A=log10_A, gamma=gamma,
                                    alphas_adapt=alpha_adapt, nfreq=nfreq)

    if psd == 'spectrum':
        if prior == 'uniform':
            log10_rho = parameter.LinearExp(-10, -4, size=components)
        elif prior == 'log-uniform':
            log10_rho = parameter.Uniform(-10, -4, size=components)

        pl = eb.free_spectrum(log10_rho=log10_rho)

    if select == 'backend':
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == 'band' or select == 'band+':
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    if break_flat:
        log10_A_flat = parameter.Uniform(-20, -11)
        gamma_flat = parameter.Constant(0)
        pl_flat = utils.powerlaw(log10_A=log10_A_flat, gamma=gamma_flat)

        freqs = 1.0 * np.arange(1, components+1) / Tspan
        components_low = sum(f < break_flat_fq for f in freqs)
        if components_low < 1.5:
            components_low = 2

        rn = gp_signals.FourierBasisGP(pl, components=components_low, Tspan=Tspan,
                                       coefficients=coefficients, selection=selection)

        rn_flat = gp_signals.FourierBasisGP(pl_flat, modes=freqs[components_low:],
                                            coefficients=coefficients, selection=selection,
                                            name='red_noise_hf')
        rn = rn + rn_flat
    else:
        rn = gp_signals.FourierBasisGP(pl, components=components, Tspan=Tspan,
                                       coefficients=coefficients, selection=selection,
                                       modes=modes)

    if select == 'band+': # Add the common component as well
        rn = rn + gp_signals.FourierBasisGP(pl, components=components, Tspan=Tspan,
                                   coefficients=coefficients)

    return rn


def dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', Tspan=None, components=30, gamma_val=None,
                   coefficients=False):
    """
    Returns DM noise model:

        1. DM noise modeled as a power-law with 30 sampling frequencies

    :param psd:
        PSD function [e.g. powerlaw (default), spectrum, tprocess]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of DM-variations.
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess DM-variations
    """
    # dm noise parameters that are common
    if gp_kernel == 'diag':
        if psd in ['powerlaw', 'turnover', 'tprocess', 'tprocess_adapt']:
            # parameters shared by PSD functions
            if prior == 'uniform':
                log10_A_dm = parameter.LinearExp(-20, -11)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_A_dm = parameter.Uniform(-20, -11)
                else:
                    log10_A_dm = parameter.Uniform(-20, -11)
            else:
                log10_A_dm = parameter.Uniform(-20, -11)

            if gamma_val is not None:
                gamma_dm = parameter.Constant(gamma_val)
            else:
                gamma_dm = parameter.Uniform(0, 7)

            # different PSD function parameters
            if psd == 'powerlaw':
                dm_prior = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
            elif psd == 'turnover':
                kappa_dm = parameter.Uniform(0, 7)
                lf0_dm = parameter.Uniform(-9, -7)
                dm_prior = utils.turnover(log10_A=log10_A_dm, gamma=gamma_dm,
                                          lf0=lf0_dm, kappa=kappa_dm)
            elif psd == 'tprocess':
                df = 2
                alphas_dm = eb.InvGamma(df/2, df/2, size=components)
                dm_prior = eb.t_process(log10_A=log10_A_dm, gamma=gamma_dm,
                                        alphas=alphas_dm)
            elif psd == 'tprocess_adapt':
                df = 2
                alpha_adapt_dm = eb.InvGamma(df/2, df/2, size=1)
                nfreq_dm = parameter.Uniform(-0.5, 10-0.5)
                dm_prior = eb.t_process_adapt(log10_A=log10_A_dm,
                                              gamma=gamma_dm,
                                              alphas_adapt=alpha_adapt_dm,
                                              nfreq=nfreq_dm)

        if psd == 'spectrum':
            if prior == 'uniform':
                log10_rho_dm = parameter.LinearExp(-10, -4, size=components)
            elif prior == 'log-uniform':
                log10_rho_dm = parameter.Uniform(-10, -4, size=components)

            dm_prior = eb.free_spectrum(log10_rho=log10_rho_dm)

        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=components,
                                                      Tspan=Tspan)

    elif gp_kernel == 'nondiag':
        if nondiag_kernel == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            dm_basis = linear_interp_basis_dm(dt=15*86400)
            dm_prior = periodic_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                       log10_gam_p=log10_gam_p, log10_p=log10_p)
        elif nondiag_kernel == 'periodic_rfband':
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            dm_basis = get_tf_quantization_matrix(df=200, dt=15*86400, dm=True)
            dm_prior = tf_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                 log10_gam_p=log10_gam_p, log10_p=log10_p,
                                 log10_alpha_wgt=log10_alpha_wgt, log10_ell2=log10_ell2)
        elif nondiag_kernel == 'sq_exp':
            # squared-exponential GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            dm_basis = linear_interp_basis_dm(dt=15*86400)
            dm_prior = se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)
        elif nondiag_kernel == 'sq_exp_rfband':
            # Sq-Exp GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            dm_basis = get_tf_quantization_matrix(df=200, dt=15*86400, dm=True)
            dm_prior = sf_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                 log10_alpha_wgt=log10_alpha_wgt, log10_ell2=log10_ell2)
        elif nondiag_kernel == 'dmx_like':
            # DMX-like signal
            log10_sigma = parameter.Uniform(-10, -4)

            dm_basis = linear_interp_basis_dm(dt=30*86400)
            dm_prior = dmx_ridge_prior(log10_sigma=log10_sigma)

    dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp',
                              coefficients=coefficients)

    return dmgp

def scattering_noise_block(kernel='periodic', coefficients=False):
    """
    Returns Scattering noise model:

        1. Scattering noise modeled as a power-law with 30 sampling frequencies

    :param psd:
        PSD function [e.g. powerlaw (default), spectrum, tprocess]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of DM-variations.
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess DM-variations
    """
    #gp_kernel == 'nondiag':
    if kernel == 'periodic':
        # Periodic GP kernel for DM
        log10_sigma = parameter.Uniform(-10, -4)
        log10_ell = parameter.Uniform(1, 4)
        log10_p = parameter.Uniform(-4, 1)
        log10_gam_p = parameter.Uniform(-3, 2)

        dm_basis = linear_interp_basis_scattering(dt=15*86400)
        dm_prior = periodic_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                   log10_gam_p=log10_gam_p, log10_p=log10_p)
    elif kernel == 'periodic_rfband':
        # Periodic GP kernel for DM with RQ radio-frequency dependence
        log10_sigma = parameter.Uniform(-10, -4)
        log10_ell = parameter.Uniform(1, 4)
        log10_ell2 = parameter.Uniform(2, 7)
        log10_alpha_wgt = parameter.Uniform(-4, 1)
        log10_p = parameter.Uniform(-4, 1)
        log10_gam_p = parameter.Uniform(-3, 2)

        dm_basis = get_tf_quantization_matrix(df=200, dt=15*86400, dm=True, idx=4)
        dm_prior = tf_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                             log10_gam_p=log10_gam_p, log10_p=log10_p,
                             log10_alpha_wgt=log10_alpha_wgt, log10_ell2=log10_ell2)
    elif kernel == 'sq_exp':
        # squared-exponential kernel for DM
        log10_sigma = parameter.Uniform(-10, -4)
        log10_ell = parameter.Uniform(1, 4)

        dm_basis = linear_interp_basis_scattering(dt=15*86400)
        dm_prior = se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)

    dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='scattering_gp',
                              coefficients=coefficients)

    return dmgp

def dm_annual_signal(idx=2, name='dm_s1yr'):
    """
    Returns chromatic annual signal (i.e. TOA advance):

    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param name: Name of signal

    :return dm1yr:
        chromatic annual waveform.
    """
    log10_Amp_dm1yr = parameter.Uniform(-10, -2)
    phase_dm1yr = parameter.Uniform(0, 2*np.pi)

    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_dm1yr,
                               phase=phase_dm1yr, idx=idx)
    dm1yr = deterministic_signals.Deterministic(wf, name=name)

    return dm1yr

def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(-10, -2)
    log10_tau_dmexp = parameter.Uniform(0, 2.5)
    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0
    wf = chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp

def dm_exponential_cusp(tmin, tmax, idx=2, sign='negative', symmetric=False, name='dm_cusp'):
    """
    Returns chromatic exponential cusp (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dm_cusp = parameter.Uniform(tmin,tmax)
    log10_Amp_dm_cusp = parameter.Uniform(-10, -2)
    log10_tau_dm_cusp_pre = parameter.Uniform(0, 2.5)

    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0

    if symmetric:
        log10_tau_dm_cusp_post = 1
    else:
        log10_tau_dm_cusp_post = parameter.Uniform(0, 2.5)

    wf = chrom_exp_cusp(log10_Amp=log10_Amp_dm_cusp, sign_param=sign_param,
                        t0=t0_dm_cusp, log10_tau_pre=log10_tau_dm_cusp_pre,
                        log10_tau_post=log10_tau_dm_cusp_post, symmetric=symmetric,
                        idx=idx)
    dm_cusp = deterministic_signals.Deterministic(wf, name=name)

    return dm_cusp

def dm_dual_exp_cusp(tmin, tmax, idx1=2, idx2=4, sign='negative', symmetric=False, name='dual_dm_cusp'):
    """
    Returns chromatic exponential cusp (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dual_cusp = parameter.Uniform(tmin,tmax)
    log10_Amp_dual_cusp_1 = parameter.Uniform(-10, -2)
    log10_Amp_dual_cusp_2 = parameter.Uniform(-10, -2)
    log10_tau_dual_cusp_pre_1 = parameter.Uniform(0, 2.5)
    log10_tau_dual_cusp_pre_2 = parameter.Uniform(0, 2.5)

    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0

    if symmetric:
        log10_tau_dual_cusp_post_1 = 1
        log10_tau_dual_cusp_post_2 = 1
    else:
        log10_tau_dual_cusp_post_1 = parameter.Uniform(0, 2.5)
        log10_tau_dual_cusp_post_2 = parameter.Uniform(0, 2.5)

    wf = chrom_dual_exp_cusp(t0=t0_dual_cusp, sign_param=sign_param, symmetric=symmetric,
                        log10_Amp_1=log10_Amp_dual_cusp_1, log10_tau_pre_1=log10_tau_dual_cusp_pre_1,
                        log10_tau_post_1=log10_tau_dual_cusp_post_1, log10_Amp_2=log10_Amp_dual_cusp_2,
                        log10_tau_pre_2=log10_tau_dual_cusp_pre_2, log10_tau_post_2=log10_tau_dual_cusp_post_2,
                        idx1=idx1, idx2=idx2)
    dm_cusp = deterministic_signals.Deterministic(wf, name=name)

    return dm_cusp

def dmx_signal(dmx_data, name='dmx_signal'):
    """
    Returns DMX signal:

    :param dmx_data: dictionary of DMX data for each pulsar from parfile.
    :param name: Name of signal.

    :return dmx_sig:
        dmx signal waveform.
    """
    dmx = {}
    for dmx_id in sorted(dmx_data):
        dmx_data_tmp = dmx_data[dmx_id]
        dmx.update({dmx_id : parameter.Normal(mu=dmx_data_tmp['DMX_VAL'],
                                              sigma=dmx_data_tmp['DMX_ERR'])})
    wf = dmx_delay(dmx_ids=dmx_data, **dmx)
    dmx_sig = deterministic_signals.Deterministic(wf, name=name)

    return dmx_sig

def chromatic_noise_block(psd='powerlaw', prior='log-uniform', idx=4,
                          Tspan=None, name='chromatic', components=30,
                          coefficients=False):
    """
    Returns GP chromatic noise model :

        1. Chromatic modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
    :param idx:
        Index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param name: Name of signal

    """
    if psd in ['powerlaw', 'turnover']:
        if prior == 'uniform':
            log10_A = parameter.LinearExp(-18, -11)
        elif prior == 'log-uniform':
            log10_A = parameter.Uniform(-18, -11)
        gamma = parameter.Uniform(0, 7)

        # PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        elif psd == 'turnover':
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            cpl = utils.turnover(log10_A=log10_A, gamma=gamma,
                                 lf0=lf0, kappa=kappa)

    if psd == 'spectrum':
        if prior == 'uniform':
            log10_rho = parameter.LinearExp(-10, -4, size=components)
        elif prior == 'log-uniform':
            log10_rho = parameter.Uniform(-10, -4, size=components)
        cpl = eb.free_spectrum(log10_rho=log10_rho)

    # set up signal
    # JS: This does not work with basis_quad function below
    #if idx == 'vary':
    #    c_idx = parameter.Uniform(0, 6)

    # quadratic piece
    basis_quad = chromatic_quad_basis(idx=idx)
    prior_quad = chromatic_quad_prior()
    cquad = gp_signals.BasisGP(prior_quad, basis_quad, name=name+'_quad')

    # Fourier piece
    basis_gp = eb.createfourierdesignmatrix_chromatic(nmodes=components,
                                                     Tspan=Tspan)
    cgp = gp_signals.BasisGP(cpl, basis_gp, name=name+'_gp',
                             coefficients=coefficients)

    return cquad + cgp

def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, components=30, gamma_val=None,
                           orf=None, name='gw', coefficients=False):
    """
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param name: Name of common red process

    """

    orfs = {'hd': utils.hd_orf(), 'dipole': utils.dipole_orf(),
            'monopole': utils.monopole_orf()}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover', 'turnover_knee']:
        amp_name = '{}_log10_A'.format(name)
        if prior == 'uniform':
            log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_Agw = parameter.Uniform(-18, -14)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = eb.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                   lfb=lfb_gw, lfk=lfk_gw,
                                   kappa=kappa_gw, delta=delta_gw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)
        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(-9, -4, size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(-9, -4, size=components)(rho_name)

        cpl = eb.free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients,
                                        components=components, Tspan=Tspan,
                                        name=name)
    elif orf in orfs.keys():
        crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf], #coefficients=coefficients,
                                              components=components, Tspan=Tspan,
                                              name=name)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn


###############################
###  PTA models from paper  ###
###############################

def model_singlepsr_noise(psr, tm_var=False, tm_linear=False, tmparam_list=None,
                          red_var=True, psd='powerlaw', red_select=None,
                          noisedict=None, tm_svd=False, tm_norm=True,
                          white_vary=True, components=30, upper_limit=False,
                          wideband=False, gamma_val=None, dm_var=False,
                          dm_type='gp', dmgp_kernel='diag', dm_psd='powerlaw',
                          dm_nondiag_kernel='periodic', dmx_data=None,
                          dm_annual=False, gamma_dm_val=None, dm_chrom=False,
                          dmchrom_psd='powerlaw', dmchrom_idx=4,
                          dm_expdip=False, dmexp_sign='negative', dm_expdip_idx=2,
                          dm_expdip_tmin=None, dm_expdip_tmax=None,
                          num_dmdips=1, dmdip_seqname=None,
                          dm_cusp=False, dm_cusp_sign='negative', dm_cusp_idx=2,
                          dm_cusp_tmin=None, dm_cusp_tmax=None, dm_cusp_sym=False,
                          num_dm_cusps=1, dm_cusp_seqname=None,
                          dm_dual_cusp=False, dm_dual_cusp_tmin=None, dm_dual_cusp_tmax=None,
                          dm_dual_cusp_idx1=2, dm_dual_cusp_idx2=4, dm_dual_cusp_sym=False,
                          dm_dual_cusp_sign='negative', num_dm_dual_cusps=1, dm_dual_cusp_seqname=None,
                          dm_scattering=False, dm_sc_kernel='sq_exp',
                          coefficients=False):
    """
    Single pulsar noise model
    :param psr: enterprise pulsar object
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tmparam_list: an explicit list of timing model parameters to vary
    :param red var: include red noise in the model
    :param psd: red noise psd model
    :param noisedict: dictionary of noise parameters
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    :param tm_norm: normalize the timing model, or provide custom normalization
    :param white_vary: boolean for varying white noise or keeping fixed
    :param components: number of modes in Fourier domain processes
    :param upper_limit: whether to do an upper-limit analysis
    :param wideband: whether to include ecorr in the white noise model
    :param gamma_val: red noise spectral index to fix
    :param dm_var: whether to explicitly model DM-variations
    :param dm_type: gaussian process ('gp') or dmx ('dmx')
    :param dmgp_kernel: diagonal in frequency or non-diagonal
    :param dm_psd: power-spectral density of DM variations
    :param dm_nondiag_kernel: type of time-domain DM GP kernel
    :param dmx_data: supply the DMX data from par files
    :param dm_annual: include an annual DM signal
    :param gamma_dm_val: spectral index of power-law DM variations
    :param dm_chrom: include general chromatic noise
    :param dmchrom_psd: power-spectral density of chromatic noise
    :param dmchrom_idx: frequency scaling of chromatic noise
    :param dm_expdip: inclue a DM exponential dip
    :param dmexp_sign: set the sign parameter for dip
    :param dm_expdip_idx: chromatic index of exponential dip
    :param dm_expdip_tmin: sampling minimum of DM dip epoch
    :param dm_expdip_tmax: sampling maximum of DM dip epoch
    :param num_dmdips: number of dm exponential dips
    :param dmdip_seqname: name of dip sequence
    :param dm_cusp: inclue a DM exponential cusp
    :param dm_cusp_sign: set the sign parameter for cusp
    :param dm_cusp_idx: chromatic index of exponential cusp
    :param dm_cusp_tmin: sampling minimum of DM cusp epoch
    :param dm_cusp_tmax: sampling maximum of DM cusp epoch
    :param dm_cusp_sym: make exponential cusp symmetric
    :param num_dm_cusps: number of dm exponential cusps
    :param dm_cusp_seqname: name of cusp sequence
    :param dm_dual_cusp: include a DM cusp with two chromatic indices
    :param dm_dual_cusp_tmin: sampling minimum of DM dual cusp epoch
    :param dm_dual_cusp_tmax: sampling maximum of DM dual cusp epoch
    :param dm_dual_cusp_idx1: first chromatic index of DM dual cusp
    :param dm_dual_cusp_idx2: second chromatic index of DM dual cusp
    :param dm_dual_cusp_sym: make dual cusp symmetric
    :param dm_dual_cusp_sign: set the sign parameter for dual cusp
    :param num_dm_dual_cusps: number of DM dual cusps
    :param dm_dual_cusp_seqname: name of dual cusp sequence
    :param dm_scattering: whether to explicitly model DM scattering variations
    :param dm_sc_kernel: type of time-domain DM GP kernel for the scattering variations
    :param coefficients: explicitly include latent coefficients in model

    :return s: single pulsar noise model
    """
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # timing model
    if not tm_var:
        s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                   coefficients=coefficients)
    else:
        # create new attribute for enterprise pulsar object
        psr.tmparams_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
        for key in psr.tmparams_orig:
            psr.tmparams_orig[key] = (psr.t2pulsar[key].val,
                                      psr.t2pulsar[key].err)
        if not tm_linear:
            s = timing_block(tmparam_list=tmparam_list)
        else:
            pass

    # red noise
    if red_var:
        s += red_noise_block(psd=psd, prior=amp_prior,
                             components=components, gamma_val=gamma_val,
                             coefficients=coefficients, select=red_select)

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            if dmgp_kernel == 'diag':
                s += dm_noise_block(gp_kernel=dmgp_kernel, psd=dm_psd,
                                    prior=amp_prior, components=components,
                                    gamma_val=gamma_dm_val,
                                    coefficients=coefficients)
            elif dmgp_kernel == 'nondiag':
                s += dm_noise_block(gp_kernel=dmgp_kernel,
                                    nondiag_kernel=dm_nondiag_kernel,
                                    coefficients=coefficients)
        elif dm_type == 'dmx':
            s += dmx_signal(dmx_data=dmx_data[psr.name])
        if dm_annual:
            s += dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic', components=components,
                                       coefficients=coefficients)
        if dm_scattering:
            s += scattering_noise_block(kernel=dm_sc_kernel, coefficients=coefficients)
        if dm_expdip:
            if dm_expdip_tmin is None and dm_expdip_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = dm_expdip_tmin
                tmax = dm_expdip_tmax
            if dmdip_seqname is not None:
                dmdipname_base = 'dmexp_'+dmdip_seqname+'_'
            else:
                dmdipname_base = 'dmexp_'
            for dd in range(1,num_dmdips+1):
                s += dm_exponential_dip(tmin=tmin, tmax=tmax,
                                        idx=dm_expdip_idx,
                                        sign=dmexp_sign,
                                        name=dmdipname_base+str(dd))
        if dm_cusp:
            if dm_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = dm_cusp_tmin
                tmax = dm_cusp_tmax
            if dm_cusp_seqname is not None:
                cusp_name_base = 'dm_cusp_'+dm_cusp_seqname+'_'
            else:
                cusp_name_base = 'dm_cusp_'
            for dd in range(1,num_dm_cusps+1):
                s += dm_exponential_cusp(tmin=tmin, tmax=tmax,
                                         idx=dm_cusp_idx,
                                         sign=dm_cusp_sign,
                                         symmetric=dm_cusp_sym,
                                         name=cusp_name_base+str(dd))
        if dm_dual_cusp:
            if dm_dual_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = dm_dual_cusp_tmin
                tmax = dm_dual_cusp_tmax
            if dm_dual_cusp_seqname is not None:
                dual_cusp_name_base = 'dm_dual_cusp_'+dm_cusp_seqname+'_'
            else:
                dual_cusp_name_base = 'dm_dual_cusp_'
            for dd in range(1,num_dm_dual_cusps+1):
                s += dm_dual_exp_cusp(tmin=tmin, tmax=tmax,
                                      idx1=dm_dual_cusp_idx1,
                                      idx2=dm_dual_cusp_idx2,
                                      sign=dm_dual_cusp_sign,
                                      symmetric=dm_dual_cusp_sym,
                                      name=dual_cusp_name_base+str(dd))

    # adding white-noise, and acting on psr objects
    if 'NANOGrav' in psr.flags['pta'] and not wideband:
        s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
        model = s2(psr)
    else:
        s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
        model = s3(psr)

    # set up PTA
    pta = signal_base.PTA([model])

    # set white noise parameters
    if not white_vary:
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_1(psrs, psd='powerlaw', noisedict=None, components=30,
            upper_limit=False, bayesephem=False, wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with only white and red noise:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Optional physical ephemeris modeling.


    :param psd:
        Choice of PSD function [e.g. powerlaw (default), turnover, tprocess]
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(psd=psd, prior=amp_prior,
                         Tspan=Tspan, components=components)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_2a(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False, select='backend'):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:
    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False, select=select)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_general(psrs, tm_var=False, tm_linear=False, tmparam_list=None,
                  common_psd='powerlaw', red_psd='powerlaw', orf=None,
                  common_components=30, red_components=30, dm_components=30,
                  modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  noisedict=None,
                  tm_svd=False, tm_norm=True, gamma_common=None,
                  upper_limit=False, upper_limit_red=None, upper_limit_dm=None,
                  upper_limit_common=None,
                  bayesephem=False, wideband=False,
                  dm_var=False, dm_type='gp', dm_psd='powerlaw', dm_annual=False,
                  white_vary=False, gequad=False, dm_chrom=False,
                  dmchrom_psd='powerlaw', dmchrom_idx=4,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  coefficients=False,):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tmparam_list: an explicit list of timing model parameters to vary
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'
    gp_priors = [upper_limit_red, upper_limit_dm, upper_limit_common]
    if all(ii is None for ii in gp_priors):
        amp_prior_red = amp_prior
        amp_prior_dm = amp_prior
        amp_prior_common = amp_prior
    else:
        amp_prior_red = 'uniform' if upper_limit_red else 'log-uniform'
        amp_prior_dm = 'uniform' if upper_limit_dm else 'log-uniform'
        amp_prior_common = 'uniform' if upper_limit_common else 'log-uniform'

    # timing model
    if not tm_var:
        s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                   coefficients=coefficients)
    else:
        # create new attribute for enterprise pulsar object
        for p in psrs:
            p.tmparams_orig = OrderedDict.fromkeys(p.t2pulsar.pars())
            for key in p.tmparams_orig:
                p.tmparams_orig[key] = (p.t2pulsar[key].val,
                                        p.t2pulsar[key].err)
        if not tm_linear:
            s = timing_block(tmparam_list=tmparam_list)
        else:
            pass

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(Tspan, nmodes_log, 1.0 / fmin / Tspan,
                                             common_components, nmodes_log)
        wgts = wgts**2.0

    # red noise
    s += red_noise_block(psd=red_psd, prior=amp_prior_red, Tspan=Tspan,
                        components=red_components, modes=modes, wgts=wgts,
                        coefficients=coefficients,
                        select=red_select, break_flat=red_breakflat,
                        break_flat_fq=red_breakflat_fq)

    # common red noise block
    if orf is None:
        s += common_red_noise_block(psd=common_psd, prior=amp_prior_common, Tspan=Tspan,
                                    components=common_components, coefficients=coefficients,
                                    gamma_val=gamma_common, name='gw')
    elif orf == 'hd':
        s += common_red_noise_block(psd=common_psd, prior=amp_prior_common, Tspan=Tspan,
                                    components=common_components, coefficients=coefficients,
                                    gamma_val=gamma_common, orf='hd', name='gw')

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd, prior=amp_prior_dm,
                                components=dm_components, gamma_val=None,
                                coefficients=coefficients)
        if dm_annual:
            s += dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic', components=components,
                                       coefficients=coefficients)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
            if gequad:
                s2 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / 86400
                tmax = p.toas.max() / 86400
                s3 = s2 + dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                             sign=False, name='dmexp')
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            if gequad:
                s4 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / 86400
                tmax = p.toas.max() / 86400
                s5 = s4 + dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                             sign=False, name='dmexp')
                models.append(s5(p))
            else:
                models.append(s4(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary:
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2b(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_2c(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_2d(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_3a(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_3b(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_3c(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        4. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_3d(psrs, psd='powerlaw', noisedict=None, components=30,
             gamma_common=None, upper_limit=False, bayesephem=False,
             wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_2a_drop_be(psrs, psd='powerlaw', noisedict=None, components=30,
                     gamma_common=None, upper_limit=False, wideband=False,
                     k_threshold=0.5):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    :param k_threshold:
        Define threshold for dropout parameter 'k'.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw')

    # ephemeris model
    s += Dropout_PhysicalEphemerisSignal(use_epoch_toas=True,
                                         k_threshold=k_threshold)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_2a_drop_crn(psrs, psd='powerlaw', noisedict=None, components=30,
                      gamma_common=None, upper_limit=False, bayesephem=False,
                      wideband=False, k_threshold=0.5):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    amp_name = '{}_log10_A'.format('gw')
    if amp_prior == 'uniform':
        log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
    elif amp_prior == 'log-uniform' and gamma_common is not None:
        if np.abs(gamma_common - 4.33) < 0.1:
            log10_Agw = parameter.Uniform(-18, -14)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)
    else:
        log10_Agw = parameter.Uniform(-18, -11)(amp_name)

    gam_name = '{}_gamma'.format('gw')
    if gamma_common is not None:
        gamma_gw = parameter.Constant(gamma_common)(gam_name)
    else:
        gamma_gw = parameter.Uniform(0, 7)(gam_name)

    k_drop = parameter.Uniform(0.0, 1.0) # per-pulsar

    drop_pl = dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                               k_drop=k_drop, k_threshold=k_threshold)
    crn = gp_signals.FourierBasisGP(drop_pl, components=components,
                                    Tspan=Tspan, name='gw')
    s += crn

    # ephemeris model
    s += Dropout_PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


## Does not yet work with IPTA datasets due to white-noise modeling issues.
def model_chromatic(psrs, psd='powerlaw', noisedict=None, components=30,
                    gamma_common=None, upper_limit=False, bayesephem=False,
                    wideband=False,
                    idx=4, chromatic_psd='powerlaw', c_psrs=['J1713+0747']):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper + additional
    chromatic noise for given pulsars

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
        6. Chromatic noise for given pulsar list

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    :param idx:
        Index of chromatic process (i.e DM is 2, scattering would be 4). If
        set to `vary` then will vary from 0 - 6 (This will be VERY slow!)
    :param chromatic_psd:
        PSD to use for chromatic noise. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param c_psrs:
        List of pulsars to use chromatic noise. 'all' will use all pulsars
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # white noise
    s = white_noise_block(vary=False, wideband=wideband)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # chromatic noise
    sc = chromatic_noise_block(psd=chromatic_psd, idx=idx)
    if c_psrs == 'all':
        s += sc
        models = [s(psr) for psr in psrs]
    elif len(c_psrs) > 0:
        models = []
        for psr in psrs:
            if psr.name in c_psrs:
                print('Adding chromatic model to PSR {}'.format(psr.name))
                snew = s + sc
                models.append(snew(psr))
            else:
                models.append(s(psr))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_bwm(psrs, noisedict=None, tm_svd=False,
              Tmin_bwm=None, Tmax_bwm=None, skyloc=None,
              red_psd='powerlaw', components=30,
              dm_var=False, dm_psd='powerlaw', dm_annual=False,
              upper_limit=False, bayesephem=False, wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with BWM model:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
    global:
        1. Deterministic GW burst with memory signal.
        2. Optional physical ephemeris modeling.

    :param psrs:
        list of enterprise.Pulsar objects for PTA
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param red_psd:
        PSD to use for per pulsar red noise. Available options
        are ['powerlaw', 'turnover', tprocess, 'spectrum'].
    :param components:
        number of modes in Fourier domain processes (red noise, DM
        variations, etc)
    :param dm_var:
        include gaussian process DM variations
    :param dm_psd:
        power-spectral density for gp DM variations
    :param dm_annual:
        include a yearly period DM variation
    :param upper_limit:
        Perform upper limit on BWM amplitude. By default this is
        set to False for a 'detection' run.
    :param bayesephem:
        Include BayesEphem model.
    :return: instantiated enterprise.PTA object
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm == None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm == None:
        Tmax_bwm = tmax/const.day

    # red noise
    s = red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior,
                   skyloc=skyloc, name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel(use_svd=tm_svd)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True)
            if dm_var and 'J1713+0747' == p.name:
                s2 += dmexp
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            if dm_var and 'J1713+0747' == p.name:
                s3 += dmexp
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_cw(psrs, upper_limit=False,
             noisedict=None, rn_psd='powerlaw', components=30,
             bayesephem=False, skyloc=None, log10_F=None, ecc=False,
             psrTerm=False, wideband=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with CW model:
    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1. Deterministic CW signal.
        2. Optional physical ephemeris modeling.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param rn_psd:
        psd to use in red_noise_block()
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_F:
        Fixed frequency of CW signal search.
        Search over frequency if ``None`` given.
    :param ecc:
        boolean or float
        if boolean: include/exclude eccentricity in search
        if float: use fixed eccentricity with eccentric model
    :psrTerm:
        boolean, include/exclude pulsar term in search
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # red noise
    s = red_noise_block(prior=amp_prior,
                        psd=rn_psd, Tspan=Tspan, components=components)

    # GW CW signal block
    if not ecc:
        s += cw_block_circ(amp_prior=amp_prior,
                           skyloc=skyloc, log10_fgw=log10_F,
                           psrTerm=psrTerm, tref=tmin, name='cw')
    else:
        if type(ecc) is not float:
            ecc = None
        s += cw_block_ecc(amp_prior=amp_prior,
                          skyloc=skyloc, log10_F=log10_F, ecc=ecc,
                          psrTerm=psrTerm, tref=tmin, name='cw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True, gp_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta
