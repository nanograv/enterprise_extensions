# -*- coding: utf-8 -*-

import functools
from collections import OrderedDict

import numpy as np
from enterprise import constants as const
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                selections, signal_base, white_signals)
from enterprise.signals.signal_base import LogLikelihood

from enterprise_extensions import chromatic as chrom
from enterprise_extensions import deterministic
from enterprise_extensions import dropout as do
from enterprise_extensions import model_utils
from enterprise_extensions.blocks import (bwm_block, bwm_sglpsr_block,
                                          chromatic_noise_block,
                                          common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)
from enterprise_extensions.chromatic.solar_wind import solar_wind_block
from enterprise_extensions.timing import timing_block

# from enterprise.signals.signal_base import LookupLikelihood


def model_singlepsr_noise(psr, tm_var=False, tm_linear=False,
                          tmparam_list=None,
                          red_var=True, psd='powerlaw', red_select=None,
                          noisedict=None, tm_svd=False, tm_norm=True,
                          white_vary=True, components=30, upper_limit=False,
                          is_wideband=False, use_dmdata=False,
                          dmjump_var=False, gamma_val=None, dm_var=False,
                          dm_type='gp', dmgp_kernel='diag', dm_psd='powerlaw',
                          dm_nondiag_kernel='periodic', dmx_data=None,
                          dm_annual=False, gamma_dm_val=None,
                          dm_dt=15, dm_df=200,
                          chrom_gp=False, chrom_gp_kernel='nondiag',
                          chrom_psd='powerlaw', chrom_idx=4, chrom_quad=False,
                          chrom_kernel='periodic',
                          chrom_dt=15, chrom_df=200,
                          dm_expdip=False, dmexp_sign='negative',
                          dm_expdip_idx=2,
                          dm_expdip_tmin=None, dm_expdip_tmax=None,
                          num_dmdips=1, dmdip_seqname=None,
                          dm_cusp=False, dm_cusp_sign='negative',
                          dm_cusp_idx=2, dm_cusp_sym=False,
                          dm_cusp_tmin=None, dm_cusp_tmax=None,
                          num_dm_cusps=1, dm_cusp_seqname=None,
                          dm_dual_cusp=False, dm_dual_cusp_tmin=None,
                          dm_dual_cusp_tmax=None, dm_dual_cusp_sym=False,
                          dm_dual_cusp_idx1=2, dm_dual_cusp_idx2=4,
                          dm_dual_cusp_sign='negative', num_dm_dual_cusps=1,
                          dm_dual_cusp_seqname=None,
                          dm_sw_deter=False, dm_sw_gp=False,
                          swgp_prior=None, swgp_basis=None,
                          coefficients=False, extra_sigs=None,
                          psr_model=False, factorized_like=False,
                          Tspan=None, fact_like_gamma=13./3, gw_components=10,
                          select='backend', tm_marg=False, dense_like=False):
    """
    Single pulsar noise model
    :param psr: enterprise pulsar object
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tmparam_list: an explicit list of timing model parameters to vary
    :param red_var: include red noise in the model
    :param psd: red noise psd model
    :param noisedict: dictionary of noise parameters
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    :param tm_norm: normalize the timing model, or provide custom normalization
    :param white_vary: boolean for varying white noise or keeping fixed
    :param components: number of modes in Fourier domain processes
    :param dm_components: number of modes in Fourier domain DM processes
    :param upper_limit: whether to do an upper-limit analysis
    :param is_wideband: whether input TOAs are wideband TOAs; will exclude
           ecorr from the white noise model
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
           is_wideband
    :param gamma_val: red noise spectral index to fix
    :param dm_var: whether to explicitly model DM-variations
    :param dm_type: gaussian process ('gp') or dmx ('dmx')
    :param dmgp_kernel: diagonal in frequency or non-diagonal
    :param dm_psd: power-spectral density of DM variations
    :param dm_nondiag_kernel: type of time-domain DM GP kernel
    :param dmx_data: supply the DMX data from par files
    :param dm_annual: include an annual DM signal
    :param gamma_dm_val: spectral index of power-law DM variations
    :param dm_dt: time-scale for DM linear interpolation basis (days)
    :param dm_df: frequency-scale for DM linear interpolation basis (MHz)
    :param chrom_gp: include general chromatic noise
    :param chrom_gp_kernel: GP kernel type to use in chrom ['diag','nondiag']
    :param chrom_psd: power-spectral density of chromatic noise
        ['powerlaw','tprocess','free_spectrum']
    :param chrom_idx: frequency scaling of chromatic noise
    :param chrom_kernel: Type of 'nondiag' time-domain chrom GP kernel to use
        ['periodic', 'sq_exp','periodic_rfband', 'sq_exp_rfband']
    :param chrom_quad: Whether to add a quadratic chromatic term. Boolean
    :param chrom_dt: time-scale for chromatic linear interpolation basis (days)
    :param chrom_df: frequency-scale for chromatic linear interpolation basis (MHz)
    :param dm_expdip: inclue a DM exponential dip
    :param dmexp_sign: set the sign parameter for dip
    :param dm_expdip_idx: chromatic index of exponential dip
    :param dm_expdip_tmin: sampling minimum of DM dip epoch
    :param dm_expdip_tmax: sampling maximum of DM dip epoch
    :param num_dmdips: number of dm exponential dips
    :param dmdip_seqname: name of dip sequence
    :param dm_cusp: include a DM exponential cusp
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
    :param dm_sw_deter: use the deterministic solar wind model
    :param dm_sw_gp: add a Gaussian process perturbation to the deterministic
        solar wind model.
    :param swgp_prior: prior is currently set automatically
    :param swgp_basis: ['powerlaw', 'periodic', 'sq_exp']
    :param coefficients: explicitly include latent coefficients in model
    :param psr_model: Return the enterprise model instantiated on the pulsar
        rather than an instantiated PTA object, i.e. model(psr) rather than
        PTA(model(psr)).
    :param factorized_like: Whether to run a factorized likelihood analyis Boolean
    :param gw_components: number of modes in Fourier domain for a common
           process in a factorized likelihood calculation.
    :param fact_like_gamma: fixed common process spectral index
    :param Tspan: time baseline used to determine Fourier GP frequencies
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return s: single pulsar noise model
    """
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # timing model
    if not tm_var:
        if (is_wideband and use_dmdata):
            if dmjump_var:
                dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmjump = parameter.Constant()
            if white_vary:
                dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
                log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
                # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmefac = parameter.Constant()
                log10_dmequad = parameter.Constant()
                # dmjump = parameter.Constant()
            s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                               log10_dmequad=log10_dmequad, dmjump=dmjump,
                                               dmefac_selection=selections.Selection(
                                                   selections.by_backend),
                                               log10_dmequad_selection=selections.Selection(
                                                   selections.by_backend),
                                               dmjump_selection=selections.Selection(
                                                   selections.by_frontend))
        else:
            if tm_marg:
                s = gp_signals.MarginalizingTimingModel()
            else:
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

    # red noise and common process
    if factorized_like:
        if Tspan is None:
            msg = 'Must Timespan to match amongst all pulsars when doing '
            msg += 'a factorized likelihood analysis.'
            raise ValueError(msg)

        s += common_red_noise_block(psd=psd, prior=amp_prior,
                                    Tspan=Tspan, components=gw_components,
                                    gamma_val=fact_like_gamma, delta_val=None,
                                    orf=None, name='gw',
                                    coefficients=coefficients,
                                    pshift=False, pseed=None)

    if red_var:
        s += red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
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
                                    dt=dm_dt, df=dm_df,
                                    coefficients=coefficients)
        elif dm_type == 'dmx':
            s += chrom.dmx_signal(dmx_data=dmx_data[psr.name])
        if dm_annual:
            s += chrom.dm_annual_signal()
        if chrom_gp:
            s += chromatic_noise_block(gp_kernel=chrom_gp_kernel,
                                       psd=chrom_psd, idx=chrom_idx,
                                       components=components,
                                       nondiag_kernel=chrom_kernel,
                                       dt=chrom_dt, df=chrom_df,
                                       include_quadratic=chrom_quad,
                                       coefficients=coefficients)

        if dm_expdip:
            if dm_expdip_tmin is None and dm_expdip_tmax is None:
                tmin = [psr.toas.min() / const.day for ii in range(num_dmdips)]
                tmax = [psr.toas.max() / const.day for ii in range(num_dmdips)]
            else:
                tmin = (dm_expdip_tmin if isinstance(dm_expdip_tmin, list)
                        else [dm_expdip_tmin])
                tmax = (dm_expdip_tmax if isinstance(dm_expdip_tmax, list)
                        else [dm_expdip_tmax])
            if dmdip_seqname is not None:
                dmdipname_base = (['dmexp_' + nm for nm in dmdip_seqname]
                                  if isinstance(dmdip_seqname, list)
                                  else ['dmexp_' + dmdip_seqname])
            else:
                dmdipname_base = ['dmexp_{0}'.format(ii+1)
                                  for ii in range(num_dmdips)]

            dm_expdip_idx = (dm_expdip_idx if isinstance(dm_expdip_idx, list)
                             else [dm_expdip_idx])
            for dd in range(num_dmdips):
                s += chrom.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                              idx=dm_expdip_idx[dd],
                                              sign=dmexp_sign,
                                              name=dmdipname_base[dd])
        if dm_cusp:
            if dm_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = [psr.toas.min() / const.day for ii in range(num_dm_cusps)]
                tmax = [psr.toas.max() / const.day for ii in range(num_dm_cusps)]
            else:
                tmin = (dm_cusp_tmin if isinstance(dm_cusp_tmin, list)
                        else [dm_cusp_tmin])
                tmax = (dm_cusp_tmax if isinstance(dm_cusp_tmax, list)
                        else [dm_cusp_tmax])
            if dm_cusp_seqname is not None:
                cusp_name_base = 'dm_cusp_'+dm_cusp_seqname+'_'
            else:
                cusp_name_base = 'dm_cusp_'
            dm_cusp_idx = (dm_cusp_idx if isinstance(dm_cusp_idx, list)
                           else [dm_cusp_idx])
            dm_cusp_sign = (dm_cusp_sign if isinstance(dm_cusp_sign, list)
                            else [dm_cusp_sign])
            for dd in range(1, num_dm_cusps+1):
                s += chrom.dm_exponential_cusp(tmin=tmin[dd-1],
                                               tmax=tmax[dd-1],
                                               idx=dm_cusp_idx,
                                               sign=dm_cusp_sign[dd-1],
                                               symmetric=dm_cusp_sym,
                                               name=cusp_name_base+str(dd))
        if dm_dual_cusp:
            if dm_dual_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = psr.toas.min() / const.day
                tmax = psr.toas.max() / const.day
            else:
                tmin = dm_dual_cusp_tmin
                tmax = dm_dual_cusp_tmax
            if dm_dual_cusp_seqname is not None:
                dual_cusp_name_base = 'dm_dual_cusp_'+dm_cusp_seqname+'_'
            else:
                dual_cusp_name_base = 'dm_dual_cusp_'
            for dd in range(1, num_dm_dual_cusps+1):
                s += chrom.dm_dual_exp_cusp(tmin=tmin, tmax=tmax,
                                            idx1=dm_dual_cusp_idx1,
                                            idx2=dm_dual_cusp_idx2,
                                            sign=dm_dual_cusp_sign,
                                            symmetric=dm_dual_cusp_sym,
                                            name=dual_cusp_name_base+str(dd))
        if dm_sw_deter:
            Tspan = psr.toas.max() - psr.toas.min()
            s+=solar_wind_block(ACE_prior=True, include_swgp=dm_sw_gp,
                                swgp_prior=swgp_prior, swgp_basis=swgp_basis,
                                Tspan=Tspan)

    if extra_sigs is not None:
        s += extra_sigs
    # adding white-noise, and acting on psr objects
    if 'NANOGrav' in psr.flags['pta'] and not is_wideband:
        s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                   select=select)
        model = s2(psr)
        if psr_model:
            Model = s2
    else:
        s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                   select=select)
        model = s3(psr)
        if psr_model:
            Model = s3

    if psr_model:
        return Model
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA([model], lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA([model])

        # set white noise parameters
        if not white_vary or (is_wideband and use_dmdata):
            if noisedict is None:
                print('No noise dictionary provided!...')
            else:
                noisedict = noisedict
                pta.set_default_params(noisedict)

        return pta


def model_1(psrs, psd='powerlaw', noisedict=None, white_vary=False,
            components=30, upper_limit=False, bayesephem=False,
            be_type='orbel', is_wideband=False, use_dmdata=False,
            select='backend', tm_marg=False, dense_like=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(psd=psd, prior=amp_prior,
                         Tspan=Tspan, components=components)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a(psrs, psd='powerlaw', noisedict=None, components=30,
             n_rnfreqs=None, n_gwbfreqs=None, gamma_common=None,
             delta_common=None, upper_limit=False, bayesephem=False,
             be_type='orbel', white_vary=False, is_wideband=False,
             use_dmdata=False, select='backend',
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param psr_models:
        Return list of psr models rather than signal_base.PTA object.
    :param n_rnfreqs:
        Number of frequencies to use in achromatic rednoise model.
    :param n_gwbfreqs:
        Number of frequencies to use in the GWB model.
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common, name='gw',
                                pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA(models)

        # set white noise parameters
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

        return pta


def model_general(psrs, tm_var=False, tm_linear=False, tmparam_list=None,
                  tm_svd=False, tm_norm=True, noisedict=None, white_vary=False,
                  Tspan=None, modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  common_psd='powerlaw', common_components=30,
                  log10_A_common=None, gamma_common=None,
                  orf='crn', orf_names=None, orf_ifreq=0, leg_lmax=5,
                  upper_limit_common=None, upper_limit=False,
                  red_var=True, red_psd='powerlaw', red_components=30, upper_limit_red=None,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  bayesephem=False, be_type='setIII_1980', is_wideband=False, use_dmdata=False,
                  dm_var=False, dm_type='gp', dm_psd='powerlaw', dm_components=30,
                  upper_limit_dm=None, dm_annual=False, dm_chrom=False, dmchrom_psd='powerlaw',
                  dmchrom_idx=4, gequad=False, coefficients=False, pshift=False,
                  select='backend', tm_marg=False, dense_like=False):
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.

    :param tm_var: boolean to vary timing model coefficients.
        [default = False]
    :param tm_linear: boolean to vary timing model under linear approximation.
        [default = False]
    :param tmparam_list: list of timing model parameters to vary.
        [default = None]
    :param tm_svd: stabilize timing model designmatrix with SVD.
        [default = False]
    :param tm_norm: normalize the timing model design matrix, or provide custom
        normalization. Alternative to 'tm_svd'.
        [default = True]
    :param noisedict: Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
        [default = None]
    :param white_vary: boolean for varying white noise or keeping fixed.
        [default = False]
    :param Tspan: timespan assumed for describing stochastic processes,
        in units of seconds. If None provided will find span of pulsars.
        [default = None]
    :param modes: list of frequencies on which to describe red processes.
        [default = None]
    :param wgts: sqrt summation weights for each frequency bin, i.e. sqrt(delta f).
        [default = None]
    :param logfreq: boolean for including log-spaced bins.
        [default = False]
    :param nmodes_log: number of log-spaced bins below 1/T.
        [default = 10]
    :param common_psd: psd of common process.
        ['powerlaw', 'spectrum', 'turnover', 'turnover_knee,', 'broken_powerlaw']
        [default = 'powerlaw']
    :param common_components: number of frequencies starting at 1/T for common process.
        [default = 30]
    :param log10_A_common: value of fixed log10_A_common parameter for
        fixed amplitude analyses.
        [default = None]
    :param gamma_common: fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
        [default = None]
    :param orf: comma de-limited string of multiple common processes with different orfs.
        [default = crn]
    :param orf_names: comma de-limited string of process names for different orfs. Manual
        control of these names is useful for embedding model_general within a hypermodel
        analysis for a process with and without hd correlations where we want to avoid
        parameter duplication.
        [default = None]
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
        [default = 0]
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function.
        [default = 5]
    :param upper_limit_common: perform upper limit on common red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param upper_limit: apply upper limit priors to all red processes.
        [default = False]
    :param red_var: boolean to switch on/off intrinsic red noise.
        [default = True]
    :param red_psd: psd of intrinsic red process.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt', 'infinitepower']
        [default = 'powerlaw']
    :param red_components: number of frequencies starting at 1/T for intrinsic red process.
        [default = 30]
    :param upper_limit_red: perform upper limit on intrinsic red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param red_select: selection properties for intrinsic red noise.
        ['backend', 'band', 'band+', None]
        [default = None]
    :param red_breakflat: break red noise spectrum and make flat above certain frequency.
        [default = False]
    :param red_breakflat_fq: break frequency for 'red_breakflat'.
        [default = None]
    :param bayesephem: boolean to include BayesEphem model.
        [default = False]
    :param be_type: flavor of bayesephem model based on how partials are computed.
        ['orbel', 'orbel-v2', 'setIII', 'setIII_1980']
        [default = 'setIII_1980']
    :param is_wideband: boolean for whether input TOAs are wideband TOAs. Will exclude
        ecorr from the white noise model.
        [default = False]
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if is_wideband.
        [default = False]
    :param dm_var: boolean for explicitly searching for DM variations.
        [default = False]
    :param dm_type: type of DM variations.
        ['gp', other choices selected with additional options; see below]
        [default = 'gp']
    :param dm_psd: psd of DM GP.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param dm_components: number of frequencies starting at 1/T for DM GP.
        [default = 30]
    :param upper_limit_dm: perform upper limit on DM GP. Note that when perfoming
        upper limits it is recommended that the spectral index also be
        fixed to a specific value.
        [default = False]
    :param dm_annual: boolean to search for an annual DM trend.
        [default = False]
    :param dm_chrom: boolean to search for a generic chromatic GP.
        [default = False]
    :param dmchrom_psd: psd of generic chromatic GP.
        ['powerlaw', 'spectrum', 'turnover']
        [default = 'powerlaw']
    :param dmchrom_idx: spectral index of generic chromatic GP.
        [default = 4]
    :param gequad: boolean to search for a global EQUAD.
        [default = False]
    :param coefficients: boolean to form full hierarchical PTA object;
        (no analytic latent-coefficient marginalization)
        [default = False]
    :param pshift: boolean to add random phase shift to red noise Fourier design
        matrices for false alarm rate studies.
        [default = False]
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    Default PTA object composition:
        1. fixed EFAC per backend/receiver system (per pulsar)
        2. fixed EQUAD per backend/receiver system (per pulsar)
        3. fixed ECORR per backend/receiver system (per pulsar)
        4. Red noise modeled as a power-law with 30 sampling frequencies
           (per pulsar)
        5. Linear timing model (per pulsar)
        6. Common-spectrum uncorrelated process modeled as a power-law with
           30 sampling frequencies. (global)
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
    if not tm_var and not use_dmdata:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                       coefficients=coefficients)
    elif not tm_var and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
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
    if Tspan is not None:
        Tspan = Tspan
    else:
        Tspan = model_utils.get_tspan(psrs)

    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(Tspan, nmodes_log,
                                             1.0 / fmin / Tspan,
                                             common_components, nmodes_log)
        wgts = wgts**2.0

    # red noise
    if red_var:
        s += red_noise_block(psd=red_psd, prior=amp_prior_red, Tspan=Tspan,
                             components=red_components, modes=modes, wgts=wgts,
                             coefficients=coefficients,
                             select=red_select, break_flat=red_breakflat,
                             break_flat_fq=red_breakflat_fq)

    # common red noise block
    crn = []
    if orf_names is None:
        orf_names = orf
    for elem, elem_name in zip(orf.split(','), orf_names.split(',')):
        if elem == 'zero_diag_bin_orf' or elem == 'zero_diag_legendre_orf':
            log10_A_val = log10_A_common
        else:
            log10_A_val = None
        crn.append(common_red_noise_block(psd=common_psd, prior=amp_prior_common, Tspan=Tspan,
                                          components=common_components,
                                          log10_A_val=log10_A_val, gamma_val=gamma_common,
                                          delta_val=None, orf=elem, name='gw_{}'.format(elem_name),
                                          orf_ifreq=orf_ifreq, leg_lmax=leg_lmax,
                                          coefficients=coefficients, pshift=pshift, pseed=None))
        # orf_ifreq only affects freq_hd model.
        # leg_lmax only affects (zero_diag_)legendre_orf model.
    crn = functools.reduce((lambda x, y: x+y), crn)
    s += crn

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd,
                                prior=amp_prior_dm,
                                components=dm_components, gamma_val=None,
                                coefficients=coefficients)
        if dm_annual:
            s += chrom.dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic',
                                       components=dm_components,
                                       coefficients=coefficients)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            if gequad:
                s2 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s3 = s2 + chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                                   sign=False, name='dmexp')
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            if gequad:
                s4 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s5 = s4 + chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                                   sign=False, name='dmexp')
                models.append(s5(p))
            else:
                models.append(s4(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', pshift=False,
             tm_marg=False, dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)
    # set white noise parameters

    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', pshift=False,
             tm_marg=False, dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3a(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, n_rnfreqs=None, n_gwbfreqs=None,
             gamma_common=None, delta_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend',
             correlationsonly=False,
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param gamma_common:
        Fixed common red process spectral index value for higher frequencies in
        broken power law model.
        By default we vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param correlationsonly:
        Give infinite power (well, 1e40) to pulsar red noise, effectively
        canceling out also GW diagonal terms
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param psr_models:
        Return list of psr models rather than signal_base.PTA object.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(psd='infinitepower' if correlationsonly else 'powerlaw',
                         prior=amp_prior,
                         Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common,
                                orf='hd', name='gw', pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA(models)

        # set white noise parameters
        if not white_vary or (is_wideband and use_dmdata):
            if noisedict is None:
                print('No noise dictionary provided!...')
            else:
                noisedict = noisedict
                pta.set_default_params(noisedict)

        return pta


def model_3b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if is_wideband and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a_drop_be(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                     components=30, gamma_common=None, upper_limit=False,
                     is_wideband=False, use_dmdata=False, k_threshold=0.5,
                     pshift=False, tm_marg=False, dense_like=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param k_threshold:
        Define threshold for dropout parameter 'k'.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True,
                                            k_threshold=k_threshold)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a_drop_crn(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                      components=30, gamma_common=None, upper_limit=False,
                      bayesephem=False, is_wideband=False, use_dmdata=False,
                      k_threshold=0.5, pshift=False, tm_marg=False,
                      dense_like=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

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

    k_drop = parameter.Uniform(0.0, 1.0)  # per-pulsar

    drop_pl = do.dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                  k_drop=k_drop, k_threshold=k_threshold)
    crn = gp_signals.FourierBasisGP(drop_pl, components=components,
                                    Tspan=Tspan, name='gw', pshift=pshift)
    s += crn

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


# Does not yet work with IPTA datasets due to white-noise modeling issues.
def model_chromatic(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                    components=30, gamma_common=None, upper_limit=False,
                    bayesephem=False, is_wideband=False, use_dmdata=False,
                    pshift=False, idx=4, chromatic_psd='powerlaw',
                    c_psrs=['J1713+0747'], tm_marg=False, dense_like=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param idx:
        Index of chromatic process (i.e DM is 2, scattering would be 4). If
        set to `vary` then will vary from 0 - 6 (This will be VERY slow!)
    :param chromatic_psd:
        PSD to use for chromatic noise. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param c_psrs:
        List of pulsars to use chromatic noise. 'all' will use all pulsars
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # white noise
    s += white_noise_block(vary=white_vary, inc_ecorr=not is_wideband)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

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
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_bwm(psrs, likelihood=LogLikelihood, lookupdir=None, noisedict=None, tm_svd=False,
              Tmin_bwm=None, Tmax_bwm=None, skyloc=None, logmin=None, logmax=None,
              burst_logmin=-17, burst_logmax=-12, red_psd='powerlaw', components=30,
              dm_var=False, dm_psd='powerlaw', dm_annual=False,
              upper_limit=False, bayesephem=False, wideband=False, tm_marg=False, dense_like=False):
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
    :param logmin:
        Lower bound on log10_A of the red noise process in each pulsar`
    :param logmax:
        Upper bound on log10_A of the red noise process in each pulsar
    :param burst_logmin:
        Lower bound on the log10_A of the burst amplitude in each pulsar
    :param burst_logmax:
        Upper boudn on the log10_A of the burst amplitude in each pulsar
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
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm is None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm is None:
        Tmax_bwm = tmax/const.day

    if tm_marg:
        s = gp_signals.MarginalizingTimingModel()
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components, logmin=logmin, logmax=logmax)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += chrom.dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = chrom.dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, logmin=burst_logmin, logmax=burst_logmax,
                   amp_prior=amp_prior,
                   skyloc=skyloc, name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

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
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_bwm_sglpsr(psr, likelihood=LogLikelihood, lookupdir=None,
                     noisedict=None, tm_svd=False,
                     Tmin_bwm=None, Tmax_bwm=None,
                     burst_logmin=-17, burst_logmax=-12, fixed_sign=None,
                     red_psd='powerlaw', logmin=None,
                     logmax=None, components=30,
                     dm_var=False, dm_psd='powerlaw', dm_annual=False,
                     upper_limit=False, bayesephem=False,
                     wideband=False, tm_marg=False, dense_like=False):
    """
    Burst-With-Memory model for single pulsar runs
    Because all of the geometric parameters (pulsar_position, source_position, gw_pol) are all degenerate with each other in a single pulsar BWM search,
    this model can only search over burst epoch and residual-space ramp amplitude (t0, ramp_amplitude)

    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with single-pulsar BWM model (called a ramp):

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
        7. Deterministic GW burst with memory signal for this pulsar

    :param psr:
        enterprise.Pulsar objects for PTA. This model is only for one pulsar at a time.
    :param likelihood:
        The likelihood function to use. The options are [enterprise.signals.signal_base.LogLikelihood, enterprise.signals.signal_base.LookupLikelihood]
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
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
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object


    """
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set frequency sampling
    tmin = psr.toas.min()
    tmax = psr.toas.max()
    Tspan = tmax - tmin

    if Tmin_bwm is None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm is None:
        Tmax_bwm = tmax/const.day

    if tm_marg:
        s = gp_signals.MarginalizingTimingModel()
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components, logmin=logmin, logmax=logmax)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += chrom.dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = chrom.dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += bwm_sglpsr_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior, name='ramp',
                          logmin=burst_logmin, logmax=burst_logmax, fixed_sign=fixed_sign)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []

    if 'NANOGrav' in psr.flags['pta'] and not wideband:
        s2 = s + white_noise_block(vary=False, inc_ecorr=True)
        if dm_var and 'J1713+0747' == psr.name:
            s2 += dmexp
        models.append(s2(psr))
    else:
        s3 = s + white_noise_block(vary=False, inc_ecorr=False)
        if dm_var and 'J1713+0747' == psr.name:
            s3 += dmexp
        models.append(s3(psr))

    # set up PTA
    # TODO: decide on a way to handle likelihood
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_fdm(psrs, noisedict=None, white_vary=False, tm_svd=False,
              Tmin_fdm=None, Tmax_fdm=None, gw_psd='powerlaw',
              red_psd='powerlaw', components=30, n_rnfreqs=None,
              n_gwbfreqs=None, gamma_common=None, delta_common=None,
              dm_var=False, dm_psd='powerlaw', dm_annual=False,
              upper_limit=False, bayesephem=False, wideband=False,
              pshift=False, pseed=None, model_CRN=False,
              amp_upper=-11, amp_lower=-18,
              freq_upper=-7, freq_lower=-9,
              use_fixed_freq=False, fixed_freq=-8, tm_marg=False,
              dense_like=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with FDM model:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
        7. The pulsar phase term.
    global:
        1. Deterministic GW FDM signal.
        2. Optional physical ephemeris modeling.

    :param psrs:
        list of enterprise.Pulsar objects for PTA
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_fdm:
        Min time to search for FDM (MJD). If omitted, uses first TOA.
    :param Tmax_fdm:
        Max time to search for FDM (MJD). If omitted, uses last TOA.
    :param gw_psd:
        PSD to use for the per pulsar GWB.
    :param red_psd:
        PSD to use for per pulsar red noise. Available options
        are ['powerlaw', 'turnover', tprocess, 'spectrum'].
    :param components:
        number of modes in Fourier domain processes (red noise, DM
        variations, etc)
    :param n_rnfreqs:
        Number of frequencies to use in achromatic rednoise model.
    :param n_gwbfreqs:
        Number of frequencies to use in the GWB model.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param dm_var:
        include gaussian process DM variations
    :param dm_psd:
        power-spectral density for gp DM variations
    :param dm_annual:
        include a yearly period DM variation
    :param upper_limit:
        Perform upper limit on FDM amplitude. By default this is
        set to False for a 'detection' run.
    :param bayesephem:
        Include BayesEphem model.
    :param wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param model_CRN:
        Option to model the common red process in addition to the
        FDM signal.
    :param amp_upper, amp_lower, freq_upper, freq_lower:
        The log-space bounds on the amplitude and frequency priors.
    :param use_fixed_freq:
        Whether to do a fixed-frequency run and not search over the frequency.
    :param fixed_freq:
        The frequency value to do a fixed-frequency run with.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # find the maximum time span to set frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_fdm is None:
        Tmin_fdm = tmin/const.day
    if Tmax_fdm is None:
        Tmax_fdm = tmax/const.day

    # timing model
    if tm_marg:
        s = gp_signals.MarginalizingTimingModel()
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=n_rnfreqs)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += chrom.dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = chrom.dm_exponential_dip(tmin=54500, tmax=54900)

    if model_CRN is True:
        # common red noise block
        s += common_red_noise_block(psd=gw_psd, prior=amp_prior, Tspan=Tspan,
                                    components=n_gwbfreqs, gamma_val=gamma_common,
                                    delta_val=delta_common, name='gw',
                                    pshift=pshift, pseed=pseed)

    # GW FDM signal block
    s += deterministic.fdm_block(Tmin_fdm, Tmax_fdm,
                                 amp_prior=amp_prior, name='fdm',
                                 amp_lower=amp_lower, amp_upper=amp_upper,
                                 freq_lower=freq_lower, freq_upper=freq_upper,
                                 use_fixed_freq=use_fixed_freq, fixed_freq=fixed_freq)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

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
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_cw(psrs, upper_limit=False, rn_psd='powerlaw', noisedict=None,
             white_vary=False, components=30, bayesephem=False, skyloc=None,
             log10_F=None, ecc=False, psrTerm=False, is_wideband=False,
             use_dmdata=False, tm_marg=False, dense_like=False):
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
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           dmefac_selection=selections.Selection(selections.by_backend),
                                           log10_dmequad_selection=selections.Selection(
                                               selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel()

    # red noise
    s += red_noise_block(prior=amp_prior,
                         psd=rn_psd, Tspan=Tspan, components=components)

    # GW CW signal block
    if not ecc:
        s += deterministic.cw_block_circ(amp_prior=amp_prior,
                                         skyloc=skyloc,
                                         log10_fgw=log10_F,
                                         psrTerm=psrTerm, tref=tmin,
                                         name='cw')
    else:
        if type(ecc) is not float:
            ecc = None
        s += deterministic.cw_block_ecc(amp_prior=amp_prior,
                                        skyloc=skyloc, log10_F=log10_F,
                                        ecc=ecc, psrTerm=psrTerm,
                                        tref=tmin, name='cw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       gp_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta
