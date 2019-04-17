from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
from enterprise import constants as const
from enterprise.signals import signal_base

@signal_base.function
def deterministic_solar_dm(toas, freqs, planetssb, pos_t,
                           n_earth=5, n_earth_bins=None,
                           t_init=None, t_final=None):

    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param freqs: radio frequencies of observations [MHz]
    :param log10_n_earth: log10 of the electron density from the solar wind
                at 1 AU.
    :param n_earth_bins: Number of binned values of n_earth for which to fit or
                an array or list of bin edges to use for binned n_Earth values.
                In the latter case the first and last edges must encompass all
                TOAs and in all cases it must match the size (number of
                elements) of log10_n_earth.
    :param t_init: Initial time of earliest TOA in entire dataset, including all
                pulsar.
    :param t_final: Final time of latest TOA in entire dataset, including all
                pulsar.

    :return dt_DM: DM due to solar wind
    """

    if n_earth_bins is None:
        earth = planetssb[:, 2, :3]
        R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
        Re_cos_theta_impact = np.einsum('ij,ij->i',earth, pos_t)

        theta_impact = np.arccos(-Re_cos_theta_impact/R_earth)

        dm_sol_wind = model_utils.dm_solar(n_earth,theta_impact,R_earth)
        dt_DM = (dm_sol_wind) * 4.148808e3 / freqs**2

    else:
        if isinstance(n_earth_bins,int) and (t_init is None or t_final is None):
            raise ValueError('Need to enter t_init and t_final to make binned n_earth values.')

        elif isinstance(n_earth_bins, int):
            edges, step = np.linspace(t_init, t_final, n_earth_bins,
                                      endpoint=True, retstep=True)

        elif isinstance(n_earth_bins, list) or isinstance(n_earth_bins, np.ndarray):
            edges = n_earth_bins

        #print('Fitting {0} binned values of n_Earth of mean width {1}.'.format(n_earth_bins,step))

        dt_DM = []
        for ii, bin in enumerate(edges[:-1]):

            bin_mask = np.logical_and(toas>=bin, toas<=edges[ii+1])
            earth = planetssb[bin_mask, 2, :3]
            R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
            Re_cos_theta_impact = np.einsum('ij,ij->i',earth, pos_t[bin_mask])

            theta_impact = np.arccos(-Re_cos_theta_impact/R_earth)
            dm_sol_wind = model_utils.dm_solar(n_earth[ii],theta_impact,R_earth)

            if dm_sol_wind.size != 0:
                dt_DM.extend((dm_sol_wind)
                             * 4.148808e3 / freqs[bin_mask]**2)
            else:
                pass

        dt_DM = np.array(dt_DM)

    return dt_DM


@signal_base.function
def createfourierdesignmatrix_solar_dm(toas, freqs, planetssb, pos_t,
                                       log10_n_earth=8.7, nmodes=30, Tspan=None,
                                       logf=False, fmin=None, fmax=None):

    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = utils.createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf,
        fmin=fmin, fmax=fmax)

    earth = planetssb[:, 2, :3]
    R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
    Re_cos_theta_impact = np.einsum('ij,ij->i',earth, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact/R_earth)

    dm_sol_wind = model_utils.dm_solar(1.0,theta_impact,R_earth)


    dt_DM = (dm_sol_wind - dm_sol_wind.mean() ) * 4.148808e3 / freqs**2

    N = len(toas)

    add_on = np.zeros((N,2*nmodes))

    add_on[:,0] += log10_n_earth

    return (add_on + F)* dt_DM[:, None], Ffreqs


def solar_dm_block(psd='powerlaw', prior='log-uniform', Tspan=None,
                   components=30, gamma_val=None):
    """
    Returns Solar Wind DM noise model:

        1. Solar Wind DM noise modeled as a power-law with 30 sampling frequencies

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
        If given, this is the fixed slope of a power-law
        DM-variation spectrum for the solar wind.
    """
    # dm noise parameters that are common
    if psd in ['powerlaw', 'turnover', 'tprocess', 'tprocess_adapt']:
        # parameters shared by PSD functions
        if prior == 'uniform':
            log10_A_dm_sw = parameter.LinearExp(-20,4)('log10_A_sol')
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_A_dm_sw = parameter.Uniform(-20,4)('log10_A_sol')
            else:
                log10_A_dm_sw = parameter.Uniform(-20,4)('log10_A_sol')
        else:
            log10_A_dm_sw = parameter.Uniform(-20,4)('log10_A_sol')

        if gamma_val is not None:
            gamma_dm_sw = parameter.Constant(gamma_val)('gamma_sol')
        else:
            gamma_dm_sw = parameter.Uniform(-7,7)('gamma_sol')


        # different PSD function parameters
        if psd == 'powerlaw':
            dm_sw_prior = utils.powerlaw(log10_A=log10_A_dm_sw, gamma=gamma_dm_sw)
        elif psd == 'turnover':
            kappa_dm = parameter.Uniform(0, 7)
            lf0_dm = parameter.Uniform(-9, -7)
            dm_sw_prior = utils.turnover(log10_A=log10_A_dm_sw, gamma=gamma_dm_sw,
                                 lf0=lf0_dm, kappa=kappa_dm)
        elif psd == 'tprocess':
            df = 2
            alphas_dm = InvGamma(df/2, df/2, size=components)
            dm_sw_prior = t_process(log10_A=log10_A_dm_sw, gamma=gamma_dm_sw, alphas=alphas_dm)
        elif psd == 'tprocess_adapt':
            df = 2
            alpha_adapt_dm = InvGamma(df/2, df/2, size=1)
            nfreq_dm = parameter.Uniform(-0.5, 10-0.5)
            dm_sw_prior = t_process_adapt(log10_A=log10_A_dm_sw, gamma=gamma_dm_sw,
                                 alphas_adapt=alpha_adapt_dm, nfreq=nfreq_dm)

    if psd == 'spectrum':
        if prior == 'uniform':
            log10_rho_dm_sw = parameter.LinearExp(-6, 8, size=components)('log10_rho_sol')

        elif prior == 'log-uniform':
            log10_rho_dm_sw = parameter.Uniform(-6, 8, size=components)('log10_rho_sol')


        dm_sw_prior = free_spectrum(log10_rho=log10_rho_dm_sw)


    log10_n_earth = parameter.Uniform(np.log10(0.01),np.log10(50))('n_earth')

    dm_sw_basis = createfourierdesignmatrix_solar_dm(log10_n_earth=log10_n_earth,nmodes=components)

    dm_sw = gp_signals.BasisGP(dm_sw_prior, dm_sw_basis, name='dm_sw')

    return dm_sw

##### Utility Functions #########

AU_light_sec = const.AU / const.c #1 AU in light seconds
AU_pc = const.AU / const.pc #1 AU in parsecs (for DM normalization)

def _dm_solar_close(n_earth,r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)

def _dm_solar(n_earth,theta_impact,r_earth):
    return ( (np.pi - theta_impact) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta_impact))) )


def dm_solar(n_earth,theta_impact,r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta_impact: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 20007 for more details.
    """
    return np.where(np.pi - theta_impact >= 1e-5,
                    _dm_solar(n_earth, theta_impact, r_earth),
                    _dm_solar_close(n_earth, r_earth))

def solar_wind(psr, n_earth=8.7):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    dispersion measure due to the solar wind and solar impact angle.

    param:: psr enterprise Pulsar objects
    param:: n_earth, proton density at 1 Au

    returns: DM due to solar wind (pc/cm^3) and solar impact angle (rad)
    """
    earth = psr.planetssb[:, 2, :3]
    R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
    Re_cos_theta_impact = np.einsum('ij,ij->i',earth, psr.pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)

    dm_sol_wind = dm_solar(n_earth, theta_impact, R_earth)

    return dm_sol_wind, theta_impact


def solar_wind_mask(psrs,angle_cutoff=None,std_cutoff=None):
    """
    Convenience function for masking TOAs lower than a certain solar impact.
    Alternatively one can set a standard deviation limit, so that all TOAs above
        a certain st dev away from the solar wind DM average for a given pulsar
        can be excised.
    param:: psrs list of enterprise Pulsar objects
    param:: angle_cutoff (degrees) Mask TOAs within this angle
    param:: std_cutoff (float number) Number of St. Devs above average to excise

    returns:: dictionary of maskes for each pulsar
    """
    solar_wind_mask = {}
    if std_cutoff and angle_cutoff:
        raise ValueError('Can not make mask using St Dev and Angular Cutoff!!')
    if std_cutoff:
        for ii,p in enumerate(psrs):
            dm_sw, _ = solar_wind(p)
            std = np.std(dm_sw)
            mean = np.mean(dm_sw)
            solar_wind_mask[p.name] = np.where(dm_sw < (mean + std_cutoff * std),
                                               True, False)
    elif angle_cutoff:
        angle_cutoff = np.deg2rad(angle_cutoff)
        for ii,p in enumerate(psrs):
            _, impact_ang = solar_wind(p)
            solar_wind_mask[p.name] = np.where(impact_ang > angle_cutoff,
                                               True, False)

    return solar_wind_mask
