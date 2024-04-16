# -*- coding: utf-8 -*-

import types

import numpy as np
from enterprise import constants as const
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise.signals import gp_signals, parameter, selections, utils, white_signals

from enterprise_extensions import deterministic as ee_deterministic

from . import chromatic as chrom
from . import dropout as drop
from . import gp_kernels as gpk
from . import model_orfs
from . import model_utils


__all__ = [
    "white_noise_block",
    "red_noise_block",
    "bwm_block",
    "bwm_sglpsr_block",
    "dm_noise_block",
    "chromatic_noise_block",
    "common_red_noise_block",
]


def channelized_backends(backend_flags):
    """Selection function to split by channelized backend flags only. For ECORR"""
    flagvals = np.unique(backend_flags)
    ch_b = ["ASP", "GASP", "GUPPI", "PUPPI", "YUPPI", "CHIME"]
    flagvals = filter(lambda x: any(map(lambda y: y in x, ch_b)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def white_noise_block(
    vary=False,
    inc_ecorr=False,
    gp_ecorr=False,
    efac1=False,
    tnequad=False,
    select="backend",
    ecorr_select="nanograv",
    common=False,
    ng_twg_setup=False,
    wb_efac_sigma=0.25,
    name=None,
):
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
    :param tnequad:
        Whether to use the TempoNest definition of EQUAD. Use None for EFAC only,
        'equad' for EQUAD only, True to follow Temponest definition.
        Defaults to False to follow Tempo, Tempo2 and Pint definition.
    :param select:
        Used to define the selection function to group ToAs for EFAC and EQUAD.
        Defaults to "backend" to use by_backend function, but could also be a list or dictionnary for custom selection, or None.
    :param ecorr_select:
        Used to define the selection function to group ToAs for ECORR
        Defaults to "nanograv" to use nanograv_backends function, but could also be "channelized" to use channelized_backends function,
        a list or dictionnary for custom selection, or None.
    :param common:
        Whether to set the white-noise signal as a common signal (i.e. multi-psr white noise)
    :param name:
        Define the signal name.
    :param ng_twg_setup:
        If True, set EFAC prior as a Normal(1, wb_efac_sigma). Default is False.
    :param wb_efac_sigma:
        Used for ng_twg_setup. Default is 0.25.
    """

    if select == "backend":
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif isinstance(select, list):
        # define selection by list of custom backend
        selection = selections.Selection(selections.custom_backends(select))
    elif isinstance(select, dict):
        # define selection by dict of custom backend
        selection = selections.Selection(selections.custom_backends_dict(select))
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    # define selection by backends for ECORR
    if ecorr_select == "nanograv":
        selection_ecorr = selections.Selection(selections.nanograv_backends)
    elif ecorr_select == "channelized":
        selection_ecorr = selections.Selection(channelized_backends)
    elif isinstance(ecorr_select, list):
        selection_ecorr = selections.Selection(selections.custom_backends(ecorr_select))
    elif isinstance(ecorr_select, dict):
        selection_ecorr = selections.Selection(
            selections.custom_backends_dict(ecorr_select)
        )
    else:
        selection_ecorr = selections.Selection(selections.no_selection)

    # white noise parameters
    if vary:
        if efac1:
            efac = parameter.Normal(1.0, 0.1)
        elif ng_twg_setup:
            efac = parameter.Normal(1.0, wb_efac_sigma)
        else:
            efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-9, -5)
        if inc_ecorr:
            ecorr = parameter.Uniform(-9, -5)
    else:
        efac = parameter.Constant()
        equad = parameter.Constant()
        if inc_ecorr:
            ecorr = parameter.Constant()

    if common:
        efac = efac(name + "_efac")
        if tnequad:
            equad = equad(name + "_log10_tnequad")
        else:
            equad = equad(name + "_log10_t2equad")

    # white noise signals
    if tnequad is None:
        efeq = white_signals.MeasurementNoise(efac=efac, selection=selection, name=name)
    elif tnequad == "equad":
        efeq = white_signals.TNEquadNoise(
            log10_tnequad=equad, selection=selection, name=name
        )
    elif tnequad:
        efeq = white_signals.MeasurementNoise(efac=efac, selection=selection, name=name)
        efeq += white_signals.TNEquadNoise(
            log10_tnequad=equad, selection=selection, name=name
        )
    else:
        efeq = white_signals.MeasurementNoise(
            efac=efac, log10_t2equad=equad, selection=selection, name=name
        )

    if inc_ecorr:
        if gp_ecorr:
            if name is None:
                ec = gp_signals.EcorrBasisModel(
                    log10_ecorr=ecorr, selection=selection_ecorr
                )
            else:
                ec = gp_signals.EcorrBasisModel(
                    log10_ecorr=ecorr, selection=selection_ecorr, name=name
                )

        else:
            ec = white_signals.EcorrKernelNoise(
                log10_ecorr=ecorr, selection=selection_ecorr, name=name
            )

    # combine signals
    if inc_ecorr:
        s = efeq + ec
    elif not inc_ecorr:
        s = efeq

    return s


def red_noise_block(
    psd="powerlaw",
    prior="log-uniform",
    logmin=None,
    logmax=None,
    modes=None,
    Tspan=None,
    components=30,
    logf=False,
    fmin=None,
    fmax=None,
    tnfreq=False,
    gamma_prior="uniform",
    gamma_val=None,
    gammamin=0,
    gammamax=7,
    delta_val=None,
    coefficients=False,
    select=None,
    wgts=None,
    combine=True,
    break_flat=False,
    break_flat_fq=None,
    dropout=False,
    dropbin=False,
    dropbin_min=10,
    k_threshold=0.5,
    name="red_noise",
):
    """
    Returns achromatic red noise signal object.

    :param psd:
        PSD function [e.g. powerlaw (default), turnover, spectrum, tprocess, tprocess_adapt]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param logmin:
        Prior lower edge for log10_A, if prior in ["uniform", "log-uniform"].
        Center of normal prior distribution if prior="gaussian".
    :param logmax:
        Prior upper edge for log10_A, if prior in ["uniform", "log-uniform"].
        Standard deviation of normal prior distribution if prior="gaussian".
    :param modes:
        List of Fourier modes. Favored against the use of Tspan, components and logf
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of red noise
    :param logf:
        Use log frequency spacing for the Fourier modes
    :param fmin:
        Lower sampling frequency
    :param fmax:
        Upper sampling frequency
    :param tnfreq::
        Number of temponest sampling components.
        If True, components might be given in days^-1
    :param gamma_prior:
        Define the prior distribution of gamma. Default is "uniform".
        "Gaussian" is also available, with gammamin and gammamax being resp. the mu and sigma
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess red noise
    :param gammamin:
        Prior lower edge for gamma, if gamma_prior="uniform" (default).
        Center of normal prior distribution if gamma_prior="gaussian".
    :param gammamax:
        Prior upper edge for gamma, if gamma_prior="uniform" (default).
        Standard deviation of normal prior distribution if gamma_prior="gaussian".
    :param delta_val:
        If not None, constant value for delta parameter of psd="broken_powerlaw".
        Corresponding to slope for frequencies < f_break (fb).
    :param coefficients: include latent coefficients in GP model
    :param select:
        Set the method to apply selection function.
    :param wgts:
        Weights used for the powerlaw_genmodes psd.
    :param combine:
        Combine argument given to BasisGP models.
    :param break_flat:
        Set psd as powerlaw + flat component, with break frequency defined by break_flat_fq
    :param break_flat_fq:
        Frequency used to separate powerlaw and flat component for break_flat=True.
    :param dropout: Use a dropout analysis for intrinsic red noise models.
        Currently only supports power law option.
    :param dropbin: Use a dropout analysis for the number of frequency bins.
        Currently only supports power law option.
    :param dropbin_min: Set the minimal number of freq. bins for the dropbin.
    :param k_threshold: Threshold for dropout analysis.
    :param name: Define the signal name.
    """
    if tnfreq and Tspan is not None:
        components = model_utils.get_tncoeff(Tspan, components)

    # red noise parameters that are common
    if psd in [
        "powerlaw",
        "powerlaw_genmodes",
        "turnover",
        "broken_powerlaw",
        "flat_powerlaw",
        "tprocess",
        "tprocess_adapt",
    ]:
        # parameters shared by PSD functions
        if logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_A = parameter.LinearExp(logmin, logmax)
            elif prior == "log-uniform":
                log10_A = parameter.Uniform(logmin, logmax)
            elif prior == "gaussian":
                log10_A = parameter.Normal(logmin, logmax)
        else:
            if prior == "uniform":
                log10_A = parameter.LinearExp(-20, -11)
            elif prior == "log-uniform" and gamma_val is not None:
                log10_A = parameter.Uniform(-20, -11)

        if gamma_val is not None:
            gamma = parameter.Constant(gamma_val)
        else:
            if gamma_prior == "uniform":
                gamma = parameter.Uniform(gammamin, gammamax)
            elif gamma_prior == "gaussian":
                gamma = parameter.Normal(gammamin, gammamax)

        # different PSD function parameters
        if psd == "powerlaw":
            if any([dropout, dropbin]):
                if dropout:
                    k_drop = parameter.Uniform(0, 1)
                else:
                    k_drop = 1
                if dropbin:
                    k_dropbin = parameter.Uniform(dropbin_min, components + 1)
                else:
                    k_dropbin = None
                pl = drop.dropout_powerlaw(
                    log10_A=log10_A,
                    gamma=gamma,
                    k_drop=k_drop,
                    k_dropbin=k_dropbin,
                    k_threshold=k_threshold,
                )
            else:
                pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        elif psd == "broken_powerlaw":
            kappa = parameter.Uniform(0.01, 0.5)
            log10_fb = parameter.Uniform(-10, -6)

            if delta_val is not None:
                delta = parameter.Constant(delta_val)
            else:
                delta = parameter.Uniform(0, 7)
            pl = gpp.broken_powerlaw(
                log10_A=log10_A,
                gamma=gamma,
                delta=delta,
                log10_fb=log10_fb,
                kappa=kappa,
            )
        elif psd == "powerlaw_genmodes":
            pl = gpp.powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=wgts)
        elif psd == "turnover":
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            pl = utils.turnover(log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa)
        elif psd == "flat_powerlaw":
            log10_B = parameter.Uniform(-10, -4)
            pl = gpp.flat_powerlaw(log10_A=log10_A, gamma=gamma, log10_B=log10_B)
        elif psd == "tprocess":
            df = 2
            alphas = gpp.InvGamma(df / 2, df / 2, size=components)
            pl = gpp.t_process(log10_A=log10_A, gamma=gamma, alphas=alphas)
        elif psd == "tprocess_adapt":
            df = 2
            alpha_adapt = gpp.InvGamma(df / 2, df / 2, size=1)
            nfreq = parameter.Uniform(-0.5, 10 - 0.5)
            pl = gpp.t_process_adapt(
                log10_A=log10_A, gamma=gamma, alphas_adapt=alpha_adapt, nfreq=nfreq
            )
    if psd == "spectrum":
        if logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_rho = parameter.LinearExp(logmin, logmax, size=components)
            elif prior == "log-uniform":
                log10_rho = parameter.Uniform(logmin, logmax, size=components)
        else:
            if prior == "uniform":
                log10_rho = parameter.LinearExp(-10, -4, size=components)
            elif prior == "log-uniform":
                log10_rho = parameter.Uniform(-10, -4, size=components)
            else:
                log10_rho = parameter.Uniform(-9, -4, size=components)

        pl = gpp.free_spectrum(log10_rho=log10_rho)

    if select == "backend":
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == "band" or select == "band+":
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    elif isinstance(select, list):
        # define selection by list of custom backend
        selection = selections.Selection(selections.custom_backends(select))
    elif isinstance(select, dict):
        # define selection by dict of custom backend
        selection = selections.Selection(selections.custom_backends_dict(select))
    elif isinstance(select, type):
        # define selection
        selection = select
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    if break_flat:
        log10_A_flat = parameter.Uniform(-20, -11)
        gamma_flat = parameter.Constant(0)
        pl_flat = utils.powerlaw(log10_A=log10_A_flat, gamma=gamma_flat)

        freqs = 1.0 * np.arange(1, components + 1) / Tspan
        components_low = sum(f < break_flat_fq for f in freqs)
        if components_low < 1.5:
            components_low = 2

        rn = gp_signals.FourierBasisGP(
            pl,
            components=components_low,
            Tspan=Tspan,
            coefficients=coefficients,
            combine=combine,
            selection=selection,
            name=name,
        )

        rn_flat = gp_signals.FourierBasisGP(
            pl_flat,
            modes=freqs[components_low:],
            coefficients=coefficients,
            selection=selection,
            combine=combine,
            name=name + "_hf",
        )
        rn = rn + rn_flat
    else:
        rn = gp_signals.FourierBasisGP(
            pl,
            components=components,
            modes=modes,
            Tspan=Tspan,
            logf=logf,
            fmin=fmin,
            fmax=fmax,
            combine=combine,
            coefficients=coefficients,
            selection=selection,
            name=name,
        )

    if select == "band+":  # Add the common component as well
        rn = rn + gp_signals.FourierBasisGP(
            pl,
            components=components,
            Tspan=Tspan,
            logf=logf,
            fmin=fmin,
            fmax=fmax,
            combine=combine,
            coefficients=coefficients,
            name=name + "_band",
        )

    return rn


def bwm_block(
    Tmin, Tmax, amp_prior="log-uniform", skyloc=None, logmin=-18, logmax=-11, name="bwm"
):
    """
    Returns deterministic GW burst with memory model:
        1. Burst event parameterized by time, sky location,
        polarization angle, and amplitude

    :param Tmin:
        Min time to search, probably first TOA (MJD).
    :param Tmax:
        Max time to search, probably last TOA (MJD).
    :param amp_prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param logmin:
        log of minimum BWM amplitude for prior (log10)
    :param logmax:
        log of maximum BWM amplitude for prior (log10)
    :param name:
        Name of BWM signal.
    """

    # BWM parameters
    amp_name = "{}_log10_A".format(name)
    if amp_prior == "uniform":
        log10_A_bwm = parameter.LinearExp(logmin, logmax)(amp_name)
    elif amp_prior == "log-uniform":
        log10_A_bwm = parameter.Uniform(logmin, logmax)(amp_name)

    pol_name = "{}_pol".format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    t0_name = "{}_t0".format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    costh_name = "{}_costheta".format(name)
    phi_name = "{}_phi".format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2 * np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    # BWM signal
    bwm_wf = ee_deterministic.bwm_delay(
        log10_h=log10_A_bwm, t0=t0, cos_gwtheta=costh, gwphi=phi, gwpol=pol
    )
    bwm = deterministic_signals.Deterministic(bwm_wf, name=name)

    return bwm


def bwm_sglpsr_block(
    Tmin,
    Tmax,
    amp_prior="log-uniform",
    logmin=-17,
    logmax=-12,
    name="ramp",
    fixed_sign=None,
):

    if fixed_sign is None:
        sign = parameter.Uniform(-1, 1)("sign")
    else:
        sign = np.sign(fixed_sign)

    amp_name = "{}_log10_A".format(name)
    if amp_prior == "uniform":
        log10_A_ramp = parameter.LinearExp(logmin, logmax)(amp_name)
    elif amp_prior == "log-uniform":
        log10_A_ramp = parameter.Uniform(logmin, logmax)(amp_name)

    t0_name = "{}_t0".format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    ramp_wf = ee_deterministic.bwm_sglpsr_delay(log10_A=log10_A_ramp, t0=t0, sign=sign)
    ramp = deterministic_signals.Deterministic(ramp_wf, name=name)

    return ramp


def dm_noise_block(
    gp_kernel="diag",
    psd="powerlaw",
    nondiag_kernel="periodic",
    prior="log-uniform",
    logmin=None,
    logmax=None,
    tndm=False,
    dt=15,
    df=200,
    modes=None,
    Tspan=None,
    components=30,
    logf=False,
    fmin=None,
    fmax=None,
    tnfreq=False,
    gamma_prior="uniform",
    gamma_val=None,
    gammamin=0,
    gammamax=7,
    delta_val=None,
    coefficients=False,
    select=None,
    dropout=False,
    dropbin=False,
    dropbin_min=10,
    k_threshold=0.5,
    name="dm_gp",
):
    """
    Returns DM noise signal object.

    :param gp_kernel:
        Type of Gaussian Process kernel [diag (default) or nondiag]
    :param psd:
        PSD function for the GP kernel if gp_kernel="diag" [e.g. powerlaw (default), turnover, spectrum, tprocess, tprocess_adapt]
    :param nondiag_kernel:
        Type of kernel to use if gp_kernel="nondiag" [periodic (default), periodic_rfband, sq_exp, sq_exp_rfband, dmx_like]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param logmin:
        Prior lower edge for log10_A, if prior in ["uniform", "log-uniform"].
        Center of normal prior distribution if prior="gaussian".
    :param logmax:
        Prior upper edge for log10_A, if prior in ["uniform", "log-uniform"].
        Standard deviation of normal prior distribution if prior="gaussian".
    :param tndm:
        Use "dm_tn" Fourier design matrix gp_base, and thus set amplitude at Temponest units.
    :param dt:
        Time-scale for linear interpolation basis (days)
    :param df:
        Frequency-scale for linear interpolation basis (MHz)
    :param modes:
        List of Fourier modes. Favored against the use of Tspan, components and logf
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of DM-variations.
    :param logf:
        Use log frequency spacing for the Fourier modes
    :param fmin:
        Lower sampling frequency
    :param fmax:
        Upper sampling frequency
    :param tnfreq::
        Number of temponest sampling components.
        If True, components might be given in days^-1
    :param gamma_prior:
        Define the prior distribution of gamma. Default is "uniform".
        "Gaussian" is also available, with gammamin and gammamax being resp. the mu and sigma
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess DM-variations
    :param gammamin:
        Prior lower edge for gamma, if gamma_prior="uniform" (default).
        Center of normal prior distribution if gamma_prior="gaussian".
    :param gammamax:
        Prior upper edge for gamma, if gamma_prior="uniform" (default).
        Standard deviation of normal prior distribution if gamma_prior="gaussian".
    :param delta_val:
        If not None, constant value for delta parameter of psd="broken_powerlaw".
        Corresponding to slope for frequencies < f_break (fb).
    :param coefficients: include latent coefficients in GP model
    :param select:
        Set the method to apply selection function.
    :param dropout: Use a dropout analysis for intrinsic red noise models.
        Currently only supports power law option.
    :param dropbin: Use a dropout analysis for the number of frequency bins.
        Currently only supports power law option.
    :param dropbin_min: Set the minimal number of freq. bins for the dropbin.
    :param k_threshold: Threshold for dropout analysis.
    :param name: Define the signal name.
    """
    if tnfreq and Tspan is not None:
        components = get_tncoeff(Tspan, components)

    # dm noise parameters that are common
    if gp_kernel == "diag":
        if psd in [
            "powerlaw", 
            "turnover", 
            "broken_powerlaw", 
            "flat_powerlaw", 
            "tprocess", 
            "tprocess_adapt"
        ]:
            # parameters shared by PSD functions
            if logmin is not None and logmax is not None:
                if prior == "uniform":
                    log10_A_dm = parameter.LinearExp(logmin, logmax)
                elif prior == "log-uniform":
                    log10_A_dm = parameter.Uniform(logmin, logmax)
                elif prior == "gaussian":
                    log10_A_dm = parameter.Normal(logmin, logmax)
            else:
                if prior == "uniform":
                    log10_A_dm = parameter.LinearExp(-20, -10)
                elif prior == "log-uniform":
                    log10_A_dm = parameter.Uniform(-20, -10)

            if gamma_val is not None:
                gamma_dm = parameter.Constant(gamma_val)
            else:
                if gamma_prior == "uniform":
                    gamma_dm = parameter.Uniform(gammamin, gammamax)
                elif gamma_prior == "gaussian":
                    gamma_dm = parameter.Normal(gammamin, gammamax)

            if gamma_val is not None:
                gamma_dm = parameter.Constant(gamma_val)
            else:
                if gamma_prior == "uniform":
                    gamma_dm = parameter.Uniform(gammamin, gammamax)
                elif gamma_prior == "gaussian":
                    gamma_dm = parameter.Normal(gammamin, gammamax)

            # different PSD function parameters
            if psd == "powerlaw":
                if any([dropout, dropbin]):
                    if dropout:
                        k_drop = parameter.Uniform(0, 1)
                    else:
                        k_drop = 1
                    if dropbin:
                        k_dropbin = parameter.Uniform(dropbin_min, components + 1)
                    else:
                        k_dropbin = None
                    dm_prior = drop.dropout_powerlaw(
                        log10_A=log10_A_dm,
                        gamma=gamma_dm,
                        k_drop=k_drop,
                        k_dropbin=k_dropbin,
                        k_threshold=k_threshold,
                    )
                else:
                    dm_prior = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
            elif psd == "broken_powerlaw":
                kappa_dm = parameter.Uniform(0.01, 0.5)
                log10_fb_dm = parameter.Uniform(-10, -6)

                if delta_val is not None:
                    delta_dm = parameter.Constant(delta_val)
                else:
                    delta_dm = parameter.Uniform(0, 7)
                dm_prior = gpp.broken_powerlaw(
                    log10_A=log10_A_dm,
                    gamma=gamma_dm,
                    delta=delta_dm,
                    log10_fb=log10_fb_dm,
                    kappa=kappa_dm,
                )
            elif psd == "turnover":
                kappa_dm = parameter.Uniform(0, 7)
                lf0_dm = parameter.Uniform(-9, -7)
                dm_prior = utils.turnover(
                    log10_A=log10_A_dm, gamma=gamma_dm, lf0=lf0_dm, kappa=kappa_dm
                )
            elif psd == "flat_powerlaw":
                log10_B_dm = parameter.Uniform(-10, -4)
                dm_prior = gpp.flat_powerlaw(
                    log10_A=log10_A_dm, gamma=gamma_dm, log10_B=log10_B_dm
                )
            elif psd == "tprocess":
                df = 2
                alphas_dm = gpp.InvGamma(df / 2, df / 2, size=components)
                dm_prior = gpp.t_process(
                    log10_A=log10_A_dm, gamma=gamma_dm, alphas=alphas_dm
                )
            elif psd == "tprocess_adapt":
                df = 2
                alpha_adapt_dm = gpp.InvGamma(df / 2, df / 2, size=1)
                nfreq_dm = parameter.Uniform(-0.5, 10 - 0.5)
                dm_prior = gpp.t_process_adapt(
                    log10_A=log10_A_dm,
                    gamma=gamma_dm,
                    alphas_adapt=alpha_adapt_dm,
                    nfreq=nfreq_dm,
                )
        if psd == "spectrum":
            if logmin is not None and logmax is not None:
                if prior == "uniform":
                    log10_rho_dm = parameter.LinearExp(logmin, logmax, size=components)
                elif prior == "log-uniform":
                    log10_rho_dm = parameter.Uniform(logmin, logmax, size=components)
            else:
                if prior == "uniform":
                    log10_rho_dm = parameter.LinearExp(-10, -4, size=components)
                elif prior == "log-uniform":
                    log10_rho_dm = parameter.Uniform(-10, -4, size=components)
                else:
                    log10_rho_dm = parameter.Uniform(-9, -4, size=components)

            dm_prior = gpp.free_spectrum(log10_rho=log10_rho_dm)

        if tndm:
            dm_basis = utils.createfourierdesignmatrix_dm_tn(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                modes=modes,
            )
        else:
            dm_basis = utils.createfourierdesignmatrix_dm(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                modes=modes,
            )

    elif gp_kernel == "nondiag":
        if nondiag_kernel == "periodic":
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            dm_basis = gpk.linear_interp_basis_dm(dt=dt * const.day)
            dm_prior = gpk.periodic_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_gam_p=log10_gam_p,
                log10_p=log10_p,
            )
        elif nondiag_kernel == "periodic_rfband":
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            dm_basis = gpk.get_tf_quantization_matrix(df=df, dt=dt * const.day, dm=True)
            dm_prior = gpk.tf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_gam_p=log10_gam_p,
                log10_p=log10_p,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )
        elif nondiag_kernel == "sq_exp":
            # squared-exponential GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            dm_basis = gpk.linear_interp_basis_dm(dt=dt * const.day)
            dm_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)
        elif nondiag_kernel == "sq_exp_rfband":
            # Sq-Exp GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            dm_basis = gpk.get_tf_quantization_matrix(df=df, dt=dt * const.day, dm=True)
            dm_prior = gpk.sf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )
        elif nondiag_kernel == "dmx_like":
            # DMX-like signal
            log10_sigma = parameter.Uniform(-10, -4)

            dm_basis = gpk.linear_interp_basis_dm(dt=dt * const.day)
            dm_prior = gpk.dmx_ridge_prior(log10_sigma=log10_sigma)

    if select is None:
        dmgp = gp_signals.BasisGP(
            dm_prior, 
            dm_basis,
            name=name, 
            coefficients=coefficients
        )
    else:
        dmgp = gp_signals.BasisGP(
            dm_prior,
            dm_basis,
            name=name,
            coefficients=coefficients,
            selection=select,
        )

    return dmgp


def chromatic_noise_block(
    gp_kernel="nondiag",
    psd="powerlaw",
    nondiag_kernel="periodic",
    prior="log-uniform",
    logmin=None,
    logmax=None,
    tndm=False,
    dt=15,
    df=200,
    idx=4,
    idxmin=0,
    idxmax=7,
    include_quadratic=False,
    modes=None,
    Tspan=None,
    components=30,
    logf=False,
    fmin=None,
    fmax=None,
    tnfreq=False,
    gamma_prior="uniform",
    gamma_val=None,
    gammamin=0,
    gammamax=7,
    delta_val=None,
    coefficients=False,
    select=None,
    dropout=False,
    dropbin=False,
    dropbin_min=10,
    k_threshold=0.5,
    name="chrom_gp",
):
    """
    Returns GP chromatic noise signal object

    :param gp_kernel:
        Type of Gaussian Process kernel [diag (default) or nondiag]
    :param psd:
        PSD function for the GP kernel if gp_kernel="diag" [e.g. powerlaw (default), turnover, spectrum, tprocess, tprocess_adapt]
    :param nondiag_kernel:
        Type of kernel to use if gp_kernel="nondiag" [periodic (default), periodic_rfband, sq_exp, sq_exp_rfband, dmx_like]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param logmin:
        Prior lower edge for log10_A, if prior in ["uniform", "log-uniform"].
        Center of normal prior distribution if prior="gaussian".
    :param logmax:
        Prior upper edge for log10_A, if prior in ["uniform", "log-uniform"].
        Standard deviation of normal prior distribution if prior="gaussian".
    :param tndm:
        Use "dm_tn" Fourier design matrix gp_base, and thus set amplitude at Temponest units.
    :param dt:
        Time-scale for linear interpolation basis (days)
    :param df:
        Frequency-scale for linear interpolation basis (MHz)
    :param idx:
        If not Nonw, fix the chromatic index to this value. Default is 4 for scattering delay variations.
        Achromatic if 0, DM noise if 2.
    :param idxmin:
        If idx is None, it is used to set the prior lower edge for idx that will be included as a parameter to fit for. Default is 0
    :param idxmax:
        If idx is None, it is used to set the prior upper edge for idx that will be included as a parameter to fit for. Default is 7
    :param include_quadratic:
        Whether to include a quadratic fit.
    :param modes:
        List of Fourier modes. Favored against the use of Tspan, components and logf
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of DM-variations.
    :param logf:
        Use log frequency spacing for the Fourier modes
    :param fmin:
        Lower sampling frequency
    :param fmax:
        Upper sampling frequency
    :param tnfreq::
        Number of temponest sampling components.
        If True, components might be given in days^-1
    :param gamma_prior:
        Define the prior distribution of gamma. Default is "uniform".
        "Gaussian" is also available, with gammamin and gammamax being resp. the mu and sigma
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess DM-variations
    :param gammamin:
        Prior lower edge for gamma, if gamma_prior="uniform" (default).
        Center of normal prior distribution if gamma_prior="gaussian".
    :param gammamax:
        Prior upper edge for gamma, if gamma_prior="uniform" (default).
        Standard deviation of normal prior distribution if gamma_prior="gaussian".
    :param delta_val:
        If not None, constant value for delta parameter of psd="broken_powerlaw".
        Corresponding to slope for frequencies < f_break (fb).
    :param coefficients: include latent coefficients in GP model
    :param select:
        Set the method to apply selection function.
    :param dropout: Use a dropout analysis for intrinsic red noise models.
        Currently only supports power law option.
    :param dropbin: Use a dropout analysis for the number of frequency bins.
        Currently only supports power law option.
    :param dropbin_min: Set the minimal number of freq. bins for the dropbin.
    :param k_threshold: Threshold for dropout analysis.
    :param name: Define the signal name.
    """
    if tnfreq and Tspan is not None:
        components = get_tncoeff(Tspan, components)

    if idx is None:
        idx = parameter.Uniform(idxmin, idxmax)

    if gp_kernel == "diag":
        if psd in ["powerlaw", "turnover", "broken_powerlaw", "flat_powerlaw"]:
            if logmin is not None and logmax is not None:
                if prior == "uniform":
                    log10_A = parameter.LinearExp(logmin, logmax)
                elif prior == "log-uniform":
                    log10_A = parameter.Uniform(logmin, logmax)
                elif prior == "gaussian":
                    log10_A = parameter.Normal(logmin, logmax)
            else:
                if prior == "uniform":
                    log10_A = parameter.LinearExp(-20, -10)
                elif prior == "log-uniform":
                    log10_A = parameter.Uniform(-20, -10)

            if gamma_val is not None:
                gamma = parameter.Constant(gamma_val)
            else:
                if gamma_prior == "uniform":
                    gamma = parameter.Uniform(gammamin, gammamax)
                elif gamma_prior == "gaussian":
                    gamma = parameter.Normal(gammamin, gammamax)

            # PSD
            if psd == "powerlaw":
                if any([dropout, dropbin]):
                    if dropout:
                        k_drop = parameter.Uniform(0, 1)
                    else:
                        k_drop = 1
                    if dropbin:
                        k_dropbin = parameter.Uniform(dropbin_min, components + 1)
                    else:
                        k_dropbin = None
                    chm_prior = drop.dropout_powerlaw(
                        log10_A=log10_A,
                        gamma=gamma,
                        k_drop=k_drop,
                        k_dropbin=k_dropbin,
                        k_threshold=k_threshold,
                    )
                else:
                    chm_prior = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            elif psd == "broken_powerlaw":
                kappa = parameter.Uniform(0.01, 0.5)
                log10_fb = parameter.Uniform(-10, -6)

                if delta_val is not None:
                    delta = parameter.Constant(delta_val)
                else:
                    delta = parameter.Uniform(0, 7)
                chm_prior = gpp.broken_powerlaw(
                    log10_A=log10_A,
                    gamma=gamma,
                    delta=delta,
                    log10_fb=log10_fb,
                    kappa=kappa,
                )
            elif psd == "turnover":
                kappa = parameter.Uniform(0, 7)
                lf0 = parameter.Uniform(-9, -7)
                chm_prior = utils.turnover(
                    log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa
                )
            elif psd == "flat_powerlaw":
                log10_B = parameter.Uniform(-10, -4)
                chm_prior = gpp.flat_powerlaw(
                    log10_A=log10_A, gamma=gamma, log10_B=log10_B
                )

        if psd == "spectrum":
            if logmin is not None and logmax is not None:
                if prior == "uniform":
                    log10_rho = parameter.LinearExp(logmin, logmax, size=components)
                elif prior == "log-uniform":
                    log10_rho = parameter.Uniform(logmin, logmax, size=components)
            else:
                if prior == "uniform":
                    log10_rho = parameter.LinearExp(-10, -4, size=components)
                elif prior == "log-uniform":
                    log10_rho = parameter.Uniform(-10, -4, size=components)
                else:
                    log10_rho = parameter.Uniform(-9, -4, size=components)

            chm_prior = gpp.free_spectrum(log10_rho=log10_rho)

        if tndm:
            chm_basis = gpb.createfourierdesignmatrix_dm_tn(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                idx=idx,
                modes=modes,
            )
        else:
            chm_basis = gpb.createfourierdesignmatrix_chromatic(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                idx=idx,
                modes=modes,
            )

    elif gp_kernel == "nondiag":
        if nondiag_kernel == "periodic":
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            chm_basis = gpk.linear_interp_basis_chromatic(dt=dt * const.day)
            chm_prior = gpk.periodic_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_gam_p=log10_gam_p,
                log10_p=log10_p,
            )

        elif nondiag_kernel == "periodic_rfband":
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            chm_basis = gpk.get_tf_quantization_matrix(
                df=df, dt=dt * const.day, dm=True, dm_idx=idx
            )
            chm_prior = gpk.tf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_gam_p=log10_gam_p,
                log10_p=log10_p,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )

        elif nondiag_kernel == "sq_exp":
            # squared-exponential kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            chm_basis = gpk.linear_interp_basis_chromatic(dt=dt * const.day, idx=idx)
            chm_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)
        elif nondiag_kernel == "sq_exp_rfband":
            # Sq-Exp GP kernel for Chrom with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            chm_basis = gpk.get_tf_quantization_matrix(
                df=df, dt=dt * const.day, dm=True, dm_idx=idx
            )
            chm_prior = gpk.sf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )

    if select is None:
        cgp = gp_signals.BasisGP(
            chm_prior, 
            chm_basis, 
            name=name, 
            coefficients=coefficients
        )
    else:
        cgp = gp_signals.BasisGP(
            chm_prior,
            chm_basis,
            name=name,
            coefficients=coefficients,
            selection=select,
        )

    if include_quadratic:
        # quadratic piece
        basis_quad = chrom.chromatic_quad_basis(idx=idx)
        prior_quad = chrom.chromatic_quad_prior()
        cquad = gp_signals.BasisGP(prior_quad, basis_quad, name=name + "_quad")
        cgp += cquad

    return cgp


def common_red_noise_block(
    psd="powerlaw",
    prior="log-uniform",
    Tspan=None,
    psrTspan=True,
    components=30,
    tnfreq=False,
    combine=True,
    log10_A_val=None,
    gamma_val=None,
    gamma_prior="uniform",
    gammamin=0,
    gammamax=7,
    delta_val=None,
    logmin=None,
    logmax=None,
    orf=None,
    orf_bins=None,
    orf_ifreq=0,
    leg_lmax=5,
    name="gw",
    coefficients=False,
    select=None,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
    pshift=False,
    pseed=None,
    dropout=False,
    dropout_psr="all",
    dropout_common=False,
    dropbin=False,
    dropbin_psr="all",
    dropbin_common=False,
    dropbin_min=10,
    k_threshold=0.5,
    idx=None,
    tndm=False,
    flagname="group",
    flagval=None,
):
    """
    Returns common red noise signal object.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum', 'broken_powerlaw']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar.
    :param log10_A_val:
        Value of log10_A parameter for fixed amplitude analyses.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param delta_val:
        Value of spectral index for high frequencies in broken power-law
        and turnover models. By default spectral index is varied in range [0,7].\
    :param logmin:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param logmax:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function [default=5]
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process

    """
    if orf_bins is None:
        orf_bin_size = 7
    else:
        orf_bin_size = len(orf_bins) - 1

    orfs = {
        "crn": None,
        "hd": model_orfs.hd_orf(),
        "gw_monopole": model_orfs.gw_monopole_orf(),
        "gw_dipole": model_orfs.gw_dipole_orf(),
        "st": model_orfs.st_orf(),
        "gt": model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)("tau")),
        "dipole": model_orfs.dipole_orf(),
        "monopole": model_orfs.monopole_orf(),
        "param_multiple_corr": model_orfs.param_multiple_corr_orf(
            mp=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_monopole"),
            dp=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_dipole"),
            hd=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_hd"),
        ),
        "zero_diag_param_multiple_corr": model_orfs.param_multiple_corr_orf(
            mp=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_monopole_zero_diag"),
            dp=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_dipole_zero_diag"),
            hd=parameter.Uniform(0.0, 1.0)("gw_orf_param_mc_hd_zero_diag"),
            diag=1e-20,
        ),
        "param_monopole": model_orfs.param_monopole_orf(
            c=parameter.Uniform(0.0, 1.0)("gw_orf_param_monopole")
        ),
        "zero_diag_param_monopole": model_orfs.param_monopole_orf(
            c=parameter.Uniform(0.0, 1.0)("gw_orf_param_monopole_zero_diag"), diag=1e-20
        ),
        "param_hd": model_orfs.param_hd_orf(
            a=parameter.Uniform(-1.5, 3.0)("gw_orf_param_hd_0"),
            b=parameter.Uniform(-1.0, 0.5)("gw_orf_param_hd_1"),
            c=parameter.Uniform(-1.0, 1.0)("gw_orf_param_hd_2"),
        ),
        "zero_diag_param_hd": model_orfs.param_hd_orf(
            a=parameter.Uniform(-1.5, 3.0)("gw_orf_param_hd_zero_diag_0"),
            b=parameter.Uniform(-1.0, 0.5)("gw_orf_param_hd_zero_diag_1"),
            c=parameter.Uniform(-1.0, 1.0)("gw_orf_param_hd_zero_diag_2"),
            diag=1e-20,
        ),
        "spline_orf": model_orfs.spline_orf(
            params=parameter.Uniform(-0.9, 0.9, size=7)("gw_orf_spline")
        ),
        "interp_orf": model_orfs.interp_orf(
            params=parameter.Uniform(-1.0, 1.0, size=7)("gw_orf_interp")
        ),
        "bin_orf": model_orfs.bin_orf(
            params=parameter.Uniform(-1.0, 1.0, size=orf_bin_size)("gw_orf_bin"),
            bins=orf_bins,
        ),
        "bin_cos_orf": model_orfs.bin_cos_orf(
            params=parameter.Uniform(-1.0, 1.0, size=7)("gw_orf_bin_cos")
        ),
        "freq_hd": model_orfs.freq_hd(params=[components, orf_ifreq]),
        "zero_diag_hd": model_orfs.hd_orf(diag=1e-20),
        "zero_diag_dipole": model_orfs.dipole_orf(diag=1e-20),
        "zero_diag_monopole": model_orfs.monopole_orf(diag=1e-20),
        "zero_diag_spline_orf": model_orfs.spline_orf(
            params=parameter.Uniform(-0.9, 0.9, size=7)("gw_orf_spline_zero_diag"),
            diag=1e-20,
        ),
        "zero_diag_interp_orf": model_orfs.interp_orf(
            params=parameter.Uniform(-1.0, 1.0, size=7)("gw_orf_interp_zero_diag"),
            diag=1e-20,
        ),
        "zero_diag_bin_orf": model_orfs.bin_orf(
            params=parameter.Uniform(-1.0, 1.0, size=7)("gw_orf_bin_zero_diag"),
            diag=1e-20,
        ),
        "zero_diag_bin_cos_orf": model_orfs.bin_cos_orf(
            params=parameter.Uniform(-1.0, 1.0, size=7)("gw_orf_bin_cos_zero_diag"),
            diag=1e-20,
        ),
        "legendre_orf": model_orfs.legendre_orf(
            params=parameter.Uniform(-1.0, 1.0, size=leg_lmax + 1)("gw_orf_legendre")
        ),
        "zero_diag_legendre_orf": model_orfs.legendre_orf(
            params=parameter.Uniform(-1.0, 1.0, size=leg_lmax + 1)(
                "gw_orf_legendre_zero_diag"
            ),
            diag=1e-20,
        ),
        "chebyshev_orf": model_orfs.chebyshev_orf(
            params=parameter.Uniform(-1.0, 1.0, size=4)("gw_orf_chebyshev")
        ),
        "zero_diag_chebyshev_orf": model_orfs.chebyshev_orf(
            params=parameter.Uniform(-1.0, 1.0, size=4)("gw_orf_chebyshev_zero_diag"),
            diag=1e-20,
        ),        
    }

    if tnfreq and Tspan is not None:
        components = get_tncoeff(Tspan, components)

    # common red noise parameters
    if psd in [
        "powerlaw", 
        "turnover", 
        "turnover_knee", 
        "broken_powerlaw",
        "flat_powerlaw",
    ]:
        amp_name = "{}_log10_A".format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)
        elif logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == "log-uniform":
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            elif prior == "gaussian":
                log10_Agw = parameter.Normal(logmin, logmax)(amp_name)
        else:
            if prior == "uniform":
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == "log-uniform":
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-20, -11)(amp_name)

        gam_name = "{}_gamma".format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            if gamma_prior == "uniform":
                gamma_gw = parameter.Uniform(gammamin, gammamax)(gam_name)
            elif gamma_prior == "gaussian":
                gamma_gw = parameter.Normal(gammamin, gammamax)(gam_name)

        # common red noise PSD
        if psd == "powerlaw":
            if any([dropout, dropbin]):
                if dropout:
                    if dropout_common:
                        k_drop = parameter.Uniform(0, 1)(name + "_k_drop")
                    else:
                        k_drop = parameter.Uniform(0, 1)
                else:
                    k_drop = None
                if dropbin:
                    if dropbin_common:
                        k_dropbin = parameter.Uniform(dropbin_min, components + 1)(
                            name + "_k_dropbin"
                        )
                    else:
                        k_dropbin = parameter.Uniform(dropbin_min, components + 1)
                else:
                    k_dropbin = None

                cpl = drop.dropout_powerlaw(
                    log10_A=log10_Agw,
                    gamma=gamma_gw,
                    dropout_psr=dropout_psr,
                    k_drop=k_drop,
                    dropbin_psr=dropbin_psr,
                    k_dropbin=k_dropbin,
                    k_threshold=k_threshold,
                )
            else:
                cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == "broken_powerlaw":
            delta_name = "{}_delta".format(name)
            kappa_name = "{}_kappa".format(name)
            log10_fb_name = "{}_log10_fb".format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(
                log10_A=log10_Agw,
                gamma=gamma_gw,
                delta=delta_gw,
                log10_fb=log10_fb_gw,
                kappa=kappa_gw,
            )
        elif psd == "turnover":
            kappa_name = "{}_kappa".format(name)
            lf0_name = "{}_log10_fbend".format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(
                log10_A=log10_Agw, gamma=gamma_gw, lf0=lf0_gw, kappa=kappa_gw
            )
        elif psd == "turnover_knee":
            kappa_name = "{}_kappa".format(name)
            lfb_name = "{}_log10_fbend".format(name)
            delta_name = "{}_delta".format(name)
            lfk_name = "{}_log10_fknee".format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(
                log10_A=log10_Agw,
                gamma=gamma_gw,
                lfb=lfb_gw,
                lfk=lfk_gw,
                kappa=kappa_gw,
                delta=delta_gw,
            )
        elif psd == "flat_powerlaw":
            bmp_name = "{}_log10_B".format(name)
            log10_Bgw = parameter.Uniform(-10, -4)(bmp_name)
            cpl = gpp.flat_powerlaw(
                log10_A=log10_Agw, gamma=gamma_gw, log10_B=log10_Bgw
            )
    if psd == "spectrum":
        rho_name = "{}_log10_rho".format(name)
        if logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(logmin, logmax, size=components)(
                    rho_name
                )
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(
                    rho_name
                )
        else:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(-10, -4, size=components)(rho_name)
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(-10, -4, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if select == "backend":
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == "band" or select == "band+":
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    elif isinstance(select, list):
        # define selection by list of custom backend
        selection = selections.Selection(selections.custom_backends(select))
    elif isinstance(select, dict):
        # define selection by dict of custom backend
        selection = selections.Selection(selections.custom_backends_dict(select))
    elif isinstance(select, type):
        # define selection
        selection = select
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    ## SET BASIS FUNCTIONS
    # TODO: Find a way to use selection functions instead of flags.
    # Note: Selecting here allows to select ToAs, even for correlated signals
    # (not available in enterprise currently)
    if flagname and flagval:
        cbasis = gpb.createfourierdesignmatrix_general(
            flagname=flagname,
            flagval=flagval,
            idx=idx,
            tndm=tndm,
            nmodes=components,
            Tspan=Tspan,
            psrTspan=psrTspan,
            logf=logf,
            fmin=fmin,
            fmax=fmax,
            modes=modes,
            pshift=pshift,
            pseed=pseed,
        )
    elif idx is not None:
        if tndm:
            cbasis = gpb.createfourierdesignmatrix_dm_tn(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                idx=idx,
                modes=modes,
            )
        else:
            cbasis = gpb.createfourierdesignmatrix_chromatic(
                nmodes=components,
                Tspan=Tspan,
                logf=logf,
                fmin=fmin,
                fmax=fmax,
                idx=idx,
                modes=modes,
            )
    else:
        cbasis = gpb.createfourierdesignmatrix_red(
            nmodes=components,
            Tspan=Tspan,
            logf=logf,
            fmin=fmin,
            fmax=fmax,
            modes=modes,
            pshift=pshift,
            pseed=pseed,
        )

    ## SET SIGNAL OBJECT
    if orf is None or "crn" in orf:
        crn = gp_signals.BasisGP(
            cpl,
            cbasis,
            coefficients=coefficients,
            combine=combine,
            selection=selection,
            name=name,
        )
    elif orf in orfs.keys():
        crn = gp_signals.BasisCommonGP(
            cpl,
            cbasis,
            orfs[orf],
            coefficients=coefficients,
            combine=combine,
            name=name,
        )
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.BasisCommonGP(
            cpl, 
            cbasis, 
            orf, 
            coefficients=coefficients, 
            combine=combine, 
            name=name
        )
    else:
        raise ValueError("ORF {} not recognized".format(orf))

    return crn
