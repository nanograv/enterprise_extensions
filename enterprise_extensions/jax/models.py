# -*- coding: utf-8 -*-

import numpy as np
from enterprise.signals import (parameter,
                                selections)
from enterprise_extensions.jax import gp_signals, signal_base

from enterprise_extensions import dropout as do
from enterprise_extensions import model_utils
from enterprise_extensions.jax.blocks import (common_red_noise_block,
                                          red_noise_block,
                                          white_noise_block)
from enterprise_extensions.jax import deterministic_signals


def model_1(psrs, psd='powerlaw', noisedict=None, white_vary=False,
            components=30, upper_limit=False, bayesephem=False, tnequad=False,
            be_type='orbel', is_wideband=False, use_dmdata=False,
            select='backend', tm_marg=False, dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

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
             be_type='setIII', white_vary=False, is_wideband=False,
             use_dmdata=False, select='backend', tnequad=False,
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.JAXPTA(models)

        # set white noise parameters
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

        return pta


def model_2b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             bayesephem=False, be_type='orbel', is_wideband=False, components=30,
             use_dmdata=False, select='backend', pshift=False, tnequad=False,
             tm_marg=False, dense_like=False, tm_svd=False, upper_limit=False,
             gamma_common=None):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)
    # set white noise parameters

    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', pshift=False,
             tm_marg=False, dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

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
             bayesephem=False, be_type='setIII', is_wideband=False,
             use_dmdata=False, select='backend',
             correlationsonly=False, tnequad=False,
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.JAXPTA(models)

        # set white noise parameters
        if not white_vary or (is_wideband and use_dmdata):
            if noisedict is None:
                print('No noise dictionary provided!...')
            else:
                noisedict = noisedict
                pta.set_default_params(noisedict)

        return pta


def model_3b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='setIII', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

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
                     pshift=False, tm_marg=False, dense_like=False, tm_svd=False,
                     tnequad=False,):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

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
                      dense_like=False, tm_svd=False, tnequad=False,):
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
    :param tm_svd: boolean for svd-stabilised timing model design matrix
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
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

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
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.JAXPTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.JAXPTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta
