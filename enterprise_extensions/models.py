# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from collections import OrderedDict

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise import constants as const

from enterprise_extensions import model_utils
from enterprise_extensions import deterministic
from enterprise_extensions.timing import timing_block
from enterprise_extensions.blocks import (white_noise_block, red_noise_block,
                                          dm_noise_block,
                                          chromatic_noise_block,
                                          common_red_noise_block)
from enterprise_extensions.chromatic.solar_wind import solar_wind_block
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import dropout as do
"""
PTA models from paper
"""


def model_singlepsr_noise(psr, tm_var=False, tm_linear=False,
                          tmparam_list=None, tm_svd=False, tm_norm=True,
                          red_var=True, psd='powerlaw', red_select=None,
                          noisedict=None, white_vary=True,
                          components=30, modes=None, wgts=None,
                          logfreq=False, nmodes_log=10, tnfreq=False,
                          upper_limit=False, is_wideband=False, use_dmdata=False,
                          dmjump_var=False, gamma_val=None, delta_val=None,
                          dm_var=False, dm_type='gp', dmgp_kernel='diag',
                          dm_psd='powerlaw', dm_nondiag_kernel='periodic',
                          dmx_data=None, dm_annual=False, gamma_dm_val=None,
                          chrom_select=None, chrom_gp=False,
                          chrom_gp_kernel='nondiag', chrom_psd='powerlaw',
                          chrom_idx=4, chrom_kernel='periodic',
                          gamma_chrom_val=None, dm_select=None, tndm=False,
                          dm_expdip=False, dmexp_sign='negative', dm_expdip_idx=2,
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
                          coefficients=False, extra_sigs=None, select='backend'):
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
    :param chrom_gp: include general chromatic noise
    :param chrom_gp_kernel: GP kernel type to use in chrom ['diag','nondiag']
    :param chrom_psd: power-spectral density of chromatic noise
        ['powerlaw','tprocess','free_spectrum']
    :param chrom_idx: frequency scaling of chromatic noise
    :param chrom_kernel: Type of 'nondiag' time-domain chrom GP kernel to use
        ['periodic', 'sq_exp','periodic_rfband', 'sq_exp_rfband']
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
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.

    :return s: single pulsar noise model
    """
    if isinstance(upper_limit, str):
        amp_prior = upper_limit
    elif upper_limit:
        amp_prior = 'uniform'
    else:
        amp_prior = 'log-uniform'

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
                #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmefac = parameter.Constant()
                log10_dmequad = parameter.Constant()
                #dmjump = parameter.Constant()
            s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                    log10_dmequad=log10_dmequad, dmjump=dmjump,
                    dmefac_selection=selections.Selection(
                        selections.by_backend),
                    log10_dmequad_selection=selections.Selection(
                        selections.by_backend),
                    dmjump_selection=selections.Selection(
                        selections.by_frontend))
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

    Tspan = model_utils.get_tspan([psr])
    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(Tspan, nmodes_log,
                                             1.0 / fmin / Tspan,
                                             components, nmodes_log)
        wgts = wgts**2.0

    if tnfreq:
        components = model_utils.get_tncoeff(Tspan, components)

    # red noise
    red_select = np.atleast_1d(red_select)
    for i in range(red_var):
        s += red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                             components=components, modes=modes, wgts=wgts,
                             gamma_val=gamma_val, delta_val=delta_val,
                             coefficients=coefficients, select=red_select[i])

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            if dmgp_kernel == 'diag':
                s += dm_noise_block(gp_kernel=dmgp_kernel, psd=dm_psd,
                                    prior=amp_prior, Tspan=Tspan,
                                    components=components,
                                    gamma_val=gamma_dm_val,
                                    coefficients=coefficients,
                                    tndm=tndm, select=dm_select)
            elif dmgp_kernel == 'nondiag':
                s += dm_noise_block(gp_kernel=dmgp_kernel, Tspan=Tspan,
                                    nondiag_kernel=dm_nondiag_kernel,
                                    coefficients=coefficients,
                                    tndm=tndm, select=dm_select)
        elif dm_type == 'dmx':
            s += chrom.dmx_signal(dmx_data=dmx_data[psr.name])
        if dm_annual:
            s += chrom.dm_annual_signal()
        if chrom_gp:
            s += chromatic_noise_block(gp_kernel=chrom_gp_kernel,
                                       psd=chrom_psd, idx=chrom_idx,
                                       Tspan=Tspan, components=components,
                                       gamma_val=gamma_chrom_val,
                                       nondiag_kernel=chrom_kernel,
                                       coefficients=coefficients,
                                       select=chrom_select)

        if dm_expdip:
            if dm_expdip_tmin is None and dm_expdip_tmax is None:
                tmin = [psr.toas.min() / 86400 for ii in range(num_dmdips)]
                tmax = [psr.toas.max() / 86400 for ii in range(num_dmdips)]
            else:
                tmin = (dm_expdip_tmin if isinstance(dm_expdip_tmin,list)
                                     else [dm_expdip_tmin])
                tmax = (dm_expdip_tmax if isinstance(dm_expdip_tmax,list)
                                     else [dm_expdip_tmax])
            if dmdip_seqname is not None:
                dmdipname_base = (['dmexp_' + nm for nm in dmdip_seqname]
                                   if isinstance(dmdip_seqname,list)
                                   else ['dmexp_' + dmdip_seqname])
            else:
                dmdipname_base = ['dmexp_{0}'.format(ii+1)
                                  for ii in range(num_dmdips)]

            dm_expdip_idx = (dm_expdip_idx if isinstance(dm_expdip_idx,list)
                                           else [dm_expdip_idx])
            for dd in range(num_dmdips):
                s += chrom.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                              idx=dm_expdip_idx[dd],
                                              sign=dmexp_sign,
                                              name=dmdipname_base[dd])
        if dm_cusp:
            if dm_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = [psr.toas.min() / 86400 for ii in range(num_dm_cusps)]
                tmax = [psr.toas.max() / 86400 for ii in range(num_dm_cusps)]
            else:
                tmin = (dm_cusp_tmin if isinstance(dm_cusp_tmin,list)
                                     else [dm_cusp_tmin])
                tmax = (dm_cusp_tmax if isinstance(dm_cusp_tmmax,list)
                                     else [dm_cusp_tmax])
            if dm_cusp_seqname is not None:
                cusp_name_base = 'dm_cusp_'+dm_cusp_seqname+'_'
            else:
                cusp_name_base = 'dm_cusp_'
            dm_cusp_idx = (dm_cusp_idx if isinstance(dm_cusp_idx,list)
                                           else [dm_cusp_idx])
            for dd in range(1,num_dm_cusps+1):
                s += chrom.dm_exponential_cusp(tmin=tmin[dd-1],
                                               tmax=tmax[dd-1],
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
    else:
        s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                select=select)
        model = s3(psr)

    # set up PTA
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
            select='backend'):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(psd=psd, prior=amp_prior,
                        Tspan=Tspan, components=components)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             n_rnfreqs = None, n_gwbfreqs=None, gamma_common=None,
             delta_common=None, upper_limit=False, bayesephem=False,
             be_type='setIII', sat_orb_elements=False,
             white_vary=False, is_wideband=False, use_dmdata=False,
             select='backend', pshift=False, pseed=None, psr_models=False):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common, name='gw',
                                pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type,
                                                           sat_orb_elements=sat_orb_elements)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
    pta = signal_base.PTA(models)

    if psr_models:
        return models
    else:
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
                  Tspan=None, common_psd='powerlaw', red_psd='powerlaw',
                  common_components=30, red_components=30, dm_components=30,
                  modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  tnfreq=False, noisedict=None, orfs=None, tm_svd=False,
                  tm_norm=True, gamma_common=None, delta_common=None,
                  red_var=True, upper_limit=False, upper_limit_red=None,
                  upper_limit_dm=None, upper_limit_common=None, bayesephem=False,
                  be_type='setIII', sat_orb_elements=False, is_wideband=False,
                  use_dmdata=False, tndm=False, dm_var=False, dm_select=None,
                  dm_type='gp', dm_psd='powerlaw', dm_annual=False, num_dmdips=1,
                  white_vary=False, gequad=False, dm_chrom=False,
                  dmchrom_psd='powerlaw', dmchrom_idx=4, chrom_select=None,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  select='backend', coefficients=False, pshift=False, pseed=None):
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
    :param Tspan:
        timespan assumed for describing stochastic processes,
        in units of seconds.
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tmparam_list: an explicit list of timing model parameters to vary
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param orfs:
        List of orfs common red noises to be added. Default is None,
        which will a spatially uncorrelated common red noise.
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
    """

    if isinstance(upper_limit, str):
        amp_prior = upper_limit
    else:
        amp_prior = 'uniform' if upper_limit else 'log-uniform'
    if upper_limit_red is None:
        amp_prior_red = amp_prior
    elif isinstance(upper_limit_red, str):
        amp_prior_red = upper_limit_red
    else:
        amp_prior_red = 'uniform' if upper_limit_red else 'log-uniform'
    if upper_limit_dm is None:
        amp_prior_dm = amp_prior
    elif isinstance(upper_limit_dm, str):
        amp_prior_dm = upper_limit_dm
    else:
        amp_prior_dm = 'uniform' if upper_limit_dm else 'log-uniform'
    if upper_limit_common is None:
        amp_prior_common = amp_prior
    elif isinstance(upper_limit_common, str):
        amp_prior_common = upper_limit_common
    else:
        amp_prior_common = 'uniform' if upper_limit_common else 'log-uniform'

    # timing model
    if not tm_var and not use_dmdata:
        s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                   coefficients=coefficients)
    elif not tm_var and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
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

    if tnfreq:
        common_components = model_utils.get_tncoeff(Tspan, common_components)
        red_components = model_utils.get_tncoeff(Tspan, red_components)
        dm_components = model_utils.get_tncoeff(Tspan, dm_components)

    # red noise
    red_select = np.atleast_1d(red_select)
    for i in range(red_var):
        s += red_noise_block(psd=red_psd, prior=amp_prior_red, Tspan=Tspan,
                             components=red_components, modes=modes, wgts=wgts,
                             coefficients=coefficients, select=red_select[i],
                             break_flat=red_breakflat, break_flat_fq=red_breakflat_fq)

    # common red noise block
    if orfs:
        orfs = np.atleast_1d(orfs)
        for orf in orfs:
            if orf == 'crn':
                s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                            Tspan=Tspan, components=common_components,
                                            coefficients=coefficients,
                                            gamma_val=gamma_common,
                                            delta_val=delta_common, name='crn',
                                            pshift=pshift, pseed=pseed)
            elif orf in ['monopole','dipole']:
                s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                            Tspan=Tspan, components=common_components,
                                            coefficients=coefficients,
                                            gamma_val=gamma_common,
                                            delta_val=delta_common, orf=orf, name=orf,
                                            pshift=pshift, pseed=pseed)
            elif orf == 'hd':
                s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                            Tspan=Tspan, components=common_components,
                                            coefficients=coefficients,
                                            gamma_val=gamma_common,
                                            delta_val=delta_common, orf='hd', name='gw',
                                            pshift=pshift, pseed=pseed)
            else:
                pass
    else:
        s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                    Tspan=Tspan, components=common_components,
                                    coefficients=coefficients,
                                    gamma_val=gamma_common, delta_val=delta_common,
                                    name='gw', pshift=pshift, pseed=pseed)

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd,
                                prior=amp_prior_dm, components=dm_components,
                                coefficients=coefficients, tndm=tndm, select=dm_select)
        if dm_annual:
            s += chrom.dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic', components=dm_components,
                                       coefficients=coefficients, select=chrom_select)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type,
                                                           sat_orb_elements=sat_orb_elements)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                    select=select)
            if gequad:
                s2 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-9, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / 86400
                tmax = p.toas.max() / 86400
                s3 = s2
                for dd in range(num_dmdips):
                    s3 += chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2, sign='negative',
                                                   name='dmexp_{0}'.format(dd+1))
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                    select=select)
            if gequad:
                s4 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-9, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / 86400
                tmax = p.toas.max() / 86400
                s5 = s4
                for dd in range(num_dmdips):
                    s5 += chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2, sign='negative',
                                                   name='dmexp_{0}'.format(dd+1))
                models.append(s5(p))
            else:
                models.append(s4(p))

    # set up PTA
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
             use_dmdata=False, select='backend', pshift=False):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             use_dmdata=False, select='backend'):
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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             use_dmdata=False, select='backend', pshift=False):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             components=30, n_rnfreqs = None, n_gwbfreqs=None,
             gamma_common=None, delta_common=None, upper_limit=False,
             bayesephem=False, be_type='setIII', sat_orb_elements=False,
             is_wideband=False, use_dmdata=False, select='backend',
             correlationsonly=False, pshift=False, pseed=None, psr_models=False):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components
    # red noise
    s = red_noise_block(psd='infinitepower' if correlationsonly else 'powerlaw',
                        prior=amp_prior,
                        Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common,
                                orf='hd', name='gw', pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type,
                                                           sat_orb_elements=sat_orb_elements)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             use_dmdata=False, select='backend'):
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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             use_dmdata=False, select='backend'):
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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if is_wideband and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
             use_dmdata=False, select='backend'):
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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                model=be_type)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
                     pshift=False):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # red noise
    s = red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True,
                                            k_threshold=k_threshold)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a_drop_crn(psrs, psd='powerlaw', noisedict=None, whict_vary=False,
                      components=30, gamma_common=None, upper_limit=False,
                      bayesephem=False, is_wideband=False, use_dmdata=False,
                      k_threshold=0.5, pshift=False):
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

    k_drop = parameter.Uniform(0.0, 1.0)  # per-pulsar

    drop_pl = do.dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                  k_drop=k_drop, k_threshold=k_threshold)
    crn = gp_signals.FourierBasisGP(drop_pl, components=components,
                                    Tspan=Tspan, name='gw', pshift=pshift)
    s += crn

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

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
                    c_psrs=['J1713+0747']):
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
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # white noise
    s = white_noise_block(vary=white_vary, inc_ecorr=not is_wideband)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
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
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_bwm(psrs, noisedict=None, white_vary=False, tm_svd=False,
              Tmin_bwm=None, Tmax_bwm=None, skyloc=None,
              red_psd='powerlaw', components=30,
              dm_var=False, dm_psd='powerlaw', dm_annual=False,
              upper_limit=False, bayesephem=False, is_wideband=False,
              use_dmdata=False):
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
    :param white_vary:
        boolean for varying white noise or keeping fixed.
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
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
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

    # red noise
    s = red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += chrom.dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = chrom.dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += deterministic.bwm_block(Tmin_bwm, Tmax_bwm,
                                         amp_prior=amp_prior,
                                         skyloc=skyloc, name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel(use_svd=tm_svd)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
            if dm_var and 'J1713+0747' == p.name:
                s2 += dmexp
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            if dm_var and 'J1713+0747' == p.name:
                s3 += dmexp
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_cw(psrs, upper_limit=False, rn_psd='powerlaw', noisedict=None,
             white_vary=False, components=30, bayesephem=False, skyloc=None,
             log10_F=None, ecc=False, psrTerm=False, is_wideband=False,
             use_dmdata=False, gp_ecorr='basis_ecorr'):
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

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            #dmjump = parameter.Constant()
        s += gp_signals.WidebandTimingModel(dmefac=dmefac,
                log10_dmequad=log10_dmequad, dmjump=dmjump,
                dmefac_selection=selections.Selection(selections.by_backend),
                log10_dmequad_selection=selections.Selection(
                    selections.by_backend),
                dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            if gp_ecorr:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                        gp_ecorr=True, name=gp_ecorr)
            else:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
            models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta
