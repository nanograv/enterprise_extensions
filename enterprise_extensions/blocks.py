# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import types

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from . import gp_kernels as gpk
from . import chromatic as chrom

__all__ = [
    "white_noise_block",
    "red_noise_block",
    "dm_noise_block",
    "scattering_noise_block",
    "chromatic_noise_block",
    "common_red_noise_block",
    "channelized_backends",
]

def channelized_backends(backend_flags):
    """Selection function to split by channelized backend flags only. For ECORR"""
    flagvals = np.unique(backend_flags)
    ch_b = ['ASP', 'GASP', 'GUPPI', 'PUPPI', 'CHIME']
    flagvals = filter(lambda x: any(map(lambda y: y in x, ch_b)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}

def white_noise_block(
    vary=False,
    inc_ecorr=False,
    gp_ecorr=False,
    efac1=False,
    select="backend",
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
    """

    if select == "backend":
        # define selection by observing backend
        backend = selections.Selection(selections.by_backend)
        # define selection by nanograv backends
        backend_ng = selections.Selection(selections.nanograv_backends)
        backend_ch = selections.Selection(channelized_backends)
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
    ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=name)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=backend, name=name)
    if inc_ecorr:
        if gp_ecorr:
            if name is None:
                name = ''

            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                            selection=backend_ch)

        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,
                                                selection=backend_ch)

    # combine signals
    if inc_ecorr:
        s = ef + eq + ec
    elif not inc_ecorr:
        s = ef + eq

    return s


def red_noise_block(
    psd="powerlaw",
    prior="log-uniform",
    Tspan=None,
    components=30,
    gamma_val=None,
    coefficients=False,
    select=None,
    modes=None,
    wgts=None,
    break_flat=False,
    break_flat_fq=None,
):
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
    if psd in [
        "powerlaw",
        "powerlaw_genmodes",
        "turnover",
        "tprocess",
        "tprocess_adapt",
        "infinitepower",
    ]:
        # parameters shared by PSD functions
        if prior == "uniform":
            log10_A = parameter.LinearExp(-20, -11)
        elif prior == "log-uniform" and gamma_val is not None:
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
        if psd == "powerlaw":
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        elif psd == "powerlaw_genmodes":
            pl = gpp.powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=wgts)
        elif psd == "turnover":
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            pl = utils.turnover(log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa)
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
        elif psd == "infinitepower":
            pl = gpp.infinitepower()

    if psd == "spectrum":
        if prior == "uniform":
            log10_rho = parameter.LinearExp(-10, -4, size=components)
        elif prior == "log-uniform":
            log10_rho = parameter.Uniform(-10, -4, size=components)

        pl = gpp.free_spectrum(log10_rho=log10_rho)

    if select == "backend":
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == "band" or select == "band+":
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
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
            selection=selection,
        )

        rn_flat = gp_signals.FourierBasisGP(
            pl_flat,
            modes=freqs[components_low:],
            coefficients=coefficients,
            selection=selection,
            name="red_noise_hf",
        )
        rn = rn + rn_flat
    else:
        rn = gp_signals.FourierBasisGP(
            pl,
            components=components,
            Tspan=Tspan,
            coefficients=coefficients,
            selection=selection,
            modes=modes,
        )

    if select == "band+":  # Add the common component as well
        rn = rn + gp_signals.FourierBasisGP(
            pl, components=components, Tspan=Tspan, coefficients=coefficients
        )

    return rn


def dm_noise_block(
    gp_kernel="diag",
    psd="powerlaw",
    nondiag_kernel="periodic",
    prior="log-uniform",
    Tspan=None,
    components=30,
    gamma_val=None,
    coefficients=False,
):
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
    if gp_kernel == "diag":
        if psd in ["powerlaw", "turnover", "tprocess", "tprocess_adapt"]:
            # parameters shared by PSD functions
            if prior == "uniform":
                log10_A_dm = parameter.LinearExp(-20, -11)
            elif prior == "log-uniform" and gamma_val is not None:
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
            if psd == "powerlaw":
                dm_prior = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
            elif psd == "turnover":
                kappa_dm = parameter.Uniform(0, 7)
                lf0_dm = parameter.Uniform(-9, -7)
                dm_prior = utils.turnover(
                    log10_A=log10_A_dm, gamma=gamma_dm, lf0=lf0_dm, kappa=kappa_dm
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
            if prior == "uniform":
                log10_rho_dm = parameter.LinearExp(-10, -4, size=components)
            elif prior == "log-uniform":
                log10_rho_dm = parameter.Uniform(-10, -4, size=components)

            dm_prior = gpp.free_spectrum(log10_rho=log10_rho_dm)

        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=components, Tspan=Tspan)

    elif gp_kernel == "nondiag":
        if nondiag_kernel == "periodic":
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            dm_basis = gpk.linear_interp_basis_dm(dt=15 * 86400)
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

            dm_basis = gpk.get_tf_quantization_matrix(df=200, dt=15 * 86400, dm=True)
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

            dm_basis = gpk.linear_interp_basis_dm(dt=15 * 86400)
            dm_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)
        elif nondiag_kernel == "sq_exp_rfband":
            # Sq-Exp GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            dm_basis = gpk.get_tf_quantization_matrix(df=200, dt=15 * 86400, dm=True)
            dm_prior = gpk.sf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )
        elif nondiag_kernel == "dmx_like":
            # DMX-like signal
            log10_sigma = parameter.Uniform(-10, -4)

            dm_basis = gpk.linear_interp_basis_dm(dt=30 * 86400)
            dm_prior = gpk.dmx_ridge_prior(log10_sigma=log10_sigma)

    dmgp = gp_signals.BasisGP(
        dm_prior, dm_basis, name="dm_gp", coefficients=coefficients
    )

    return dmgp


def chromatic_noise_block(
    gp_kernel="nondiag",
    psd="powerlaw",
    nondiag_kernel="periodic",
    prior="log-uniform",
    idx=4,
    include_quadratic=False,
    Tspan=None,
    name="chrom",
    components=30,
    coefficients=False,
):
    """
    Returns GP chromatic noise model :

        1. Chromatic modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param gp_kernel:
        Whether to use a diagonal kernel for the GP. ['diag','nondiag']
    :param nondiag_kernel:
        Which nondiagonal kernel to use for the GP.
        ['periodic','sq_exp','periodic_rfband','sq_exp_rfband']
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
    :param prior:
        What type of prior to use for amplitudes. ['log-uniform','uniform']
    :param idx:
        Index of radio frequency dependence (i.e. DM is 2). Any float will work.
    :param include_quadratic:
        Whether to include a quadratic fit.
    :param name: Name of signal
    :param Tspan:
        Tspan from which to calculate frequencies for PSD-based GPs.
    :param components:
        Number of frequencies to use in 'diag' GPs.
    :param coefficients:
        Whether to keep coefficients of the GP.

    """
    if gp_kernel == "diag":
        chm_basis = gpb.createfourierdesignmatrix_chromatic(
            nmodes=components, Tspan=Tspan
        )
        if psd in ["powerlaw", "turnover"]:
            if prior == "uniform":
                log10_A = parameter.LinearExp(-18, -11)
            elif prior == "log-uniform":
                log10_A = parameter.Uniform(-18, -11)
            gamma = parameter.Uniform(0, 7)

            # PSD
            if psd == "powerlaw":
                chm_prior = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            elif psd == "turnover":
                kappa = parameter.Uniform(0, 7)
                lf0 = parameter.Uniform(-9, -7)
                chm_prior = utils.turnover(
                    log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa
                )

        if psd == "spectrum":
            if prior == "uniform":
                log10_rho = parameter.LinearExp(-10, -4, size=components)
            elif prior == "log-uniform":
                log10_rho = parameter.Uniform(-10, -4, size=components)
            chm_prior = gpp.free_spectrum(log10_rho=log10_rho)

    elif gp_kernel == "nondiag":
        if nondiag_kernel == "periodic":
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            chm_basis = gpk.linear_interp_basis_chromatic(dt=15 * 86400)
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
                df=200, dt=15 * 86400, dm=True, idx=idx
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

            chm_basis = gpk.linear_interp_basis_chromatic(dt=15 * 86400, idx=idx)
            chm_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell)
        elif nondiag_kernel == "sq_exp_rfband":
            # Sq-Exp GP kernel for Chrom with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            log10_alpha_wgt = parameter.Uniform(-4, 1)

            dm_basis = gpk.get_tf_quantization_matrix(
                df=200, dt=15 * 86400, dm=True, idx=idx
            )
            dm_prior = gpk.sf_kernel(
                log10_sigma=log10_sigma,
                log10_ell=log10_ell,
                log10_alpha_wgt=log10_alpha_wgt,
                log10_ell2=log10_ell2,
            )

    cgp = gp_signals.BasisGP(
        chm_prior, chm_basis, name=name + "_gp", coefficients=coefficients
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
    components=30,
    gamma_val=None,
    orf=None,
    name="gw",
    coefficients=False,
    pshift=False,
    pseed=None,
):
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
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process

    """

    orfs = {
        "hd": utils.hd_orf(),
        "dipole": utils.dipole_orf(),
        "monopole": utils.monopole_orf(),
    }

    # common red noise parameters
    if psd in ["powerlaw", "turnover", "turnover_knee"]:
        amp_name = "{}_log10_A".format(name)
        if prior == "uniform":
            log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
        elif prior == "log-uniform" and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_Agw = parameter.Uniform(-18, -14)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = "{}_gamma".format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == "powerlaw":
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
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

    if psd == "spectrum":
        rho_name = "{}_log10_rho".format(name)
        if prior == "uniform":
            log10_rho_gw = parameter.LinearExp(-9, -4, size=components)(rho_name)
        elif prior == "log-uniform":
            log10_rho_gw = parameter.Uniform(-9, -4, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(
            cpl,
            coefficients=coefficients,
            components=components,
            Tspan=Tspan,
            name=name,
            pshift=pshift,
            pseed=pseed,
        )
    elif orf in orfs.keys():
        crn = gp_signals.FourierBasisCommonGP(
            cpl,
            orfs[orf],
            components=components,
            Tspan=Tspan,
            name=name,
            pshift=pshift,
            pseed=pseed,
        )
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.FourierBasisCommonGP(
            cpl,
            orf,
            components=components,
            Tspan=Tspan,
            name=name,
            pshift=pshift,
            pseed=pseed,
        )
    else:
        raise ValueError("ORF {} not recognized".format(orf))

    return crn
