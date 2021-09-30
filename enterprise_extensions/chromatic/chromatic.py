# -*- coding: utf-8 -*-

import numpy as np
from enterprise import constants as const
from enterprise.signals import deterministic_signals, parameter, signal_base

__all__ = ['chrom_exp_decay',
           'chrom_exp_cusp',
           'chrom_dual_exp_cusp',
           'chrom_yearly_sinusoid',
           'chromatic_quad_basis',
           'chromatic_quad_prior',
           'dmx_delay',
           'dm_exponential_dip',
           'dm_exponential_cusp',
           'dm_dual_exp_cusp',
           'dmx_signal',
           'dm_annual_signal',
           ]


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
                        log10_Amp_1=-7, log10_tau_pre_1=1.7,
                        log10_tau_post_1=1.7,
                        log10_Amp_2=-7, log10_tau_pre_2=1.7,
                        log10_tau_post_2=1.7,
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

    return np.sign(sign_param) * (wf_1 * (1400 / freqs) ** idx1 + wf_2 * (1400 / freqs) ** idx2)


@signal_base.function
def chrom_yearly_sinusoid(toas, freqs, log10_Amp=-7, phase=0, idx=2):
    """
    Chromatic annual sinusoid.

    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """

    wf = 10**log10_Amp * np.sin(2 * np.pi * const.fyr * toas + phase)
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
    t0_dmexp = parameter.Uniform(tmin, tmax)
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


def dm_exponential_cusp(tmin, tmax, idx=2, sign='negative',
                        symmetric=False, name='dm_cusp'):
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
    t0_dm_cusp = parameter.Uniform(tmin, tmax)
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
                        log10_tau_post=log10_tau_dm_cusp_post,
                        symmetric=symmetric, idx=idx)
    dm_cusp = deterministic_signals.Deterministic(wf, name=name)

    return dm_cusp


def dm_dual_exp_cusp(tmin, tmax, idx1=2, idx2=4, sign='negative',
                     symmetric=False, name='dual_dm_cusp'):
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
    t0_dual_cusp = parameter.Uniform(tmin, tmax)
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

    wf = chrom_dual_exp_cusp(t0=t0_dual_cusp, sign_param=sign_param,
                             symmetric=symmetric,
                             log10_Amp_1=log10_Amp_dual_cusp_1,
                             log10_tau_pre_1=log10_tau_dual_cusp_pre_1,
                             log10_tau_post_1=log10_tau_dual_cusp_post_1,
                             log10_Amp_2=log10_Amp_dual_cusp_2,
                             log10_tau_pre_2=log10_tau_dual_cusp_pre_2,
                             log10_tau_post_2=log10_tau_dual_cusp_post_2,
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
        dmx.update({dmx_id: parameter.Normal(mu=dmx_data_tmp['DMX_VAL'],
                                             sigma=dmx_data_tmp['DMX_ERR'])})
    wf = dmx_delay(dmx_ids=dmx_data, **dmx)
    dmx_sig = deterministic_signals.Deterministic(wf, name=name)

    return dmx_sig


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
