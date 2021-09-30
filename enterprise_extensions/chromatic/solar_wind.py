# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.stats as sps
from enterprise import constants as const
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                signal_base, utils)

from .. import gp_kernels as gpk

defpath = os.path.dirname(__file__)

yr_in_sec = 365.25*24*3600


@signal_base.function
def solar_wind(toas, freqs, planetssb, pos_t, n_earth=5, n_earth_bins=None,
               t_init=None, t_final=None):
    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param freqs: radio frequencies of observations [MHz]
    :param n_earth: The electron density from the solar wind at 1 AU.
    :param n_earth_bins: Number of binned values of n_earth for which to fit or
                an array or list of bin edges to use for binned n_Earth values.
                In the latter case the first and last edges must encompass all
                TOAs and in all cases it must match the size (number of
                elements) of n_earth.
    :param t_init: Initial time of earliest TOA in entire dataset, including
                all pulsars.
    :param t_final: Final time of last TOA in entire dataset, including all
                pulsars.

    :return dt_DM: Chromatic time delay due to solar wind
    """

    if n_earth_bins is None:
        theta, R_earth = theta_impact(planetssb, pos_t)
        dm_sol_wind = dm_solar(n_earth, theta, R_earth)
        dt_DM = (dm_sol_wind) * 4.148808e3 / freqs**2

    else:
        if isinstance(n_earth_bins, int) and (t_init is None
                                              or t_final is None):
            err_msg = 'Need to enter t_init and t_final '
            err_msg += 'to make binned n_earth values.'
            raise ValueError(err_msg)

        elif isinstance(n_earth_bins, int):
            edges, step = np.linspace(t_init, t_final, n_earth_bins,
                                      endpoint=True, retstep=True)

        elif isinstance(n_earth_bins, list) or isinstance(n_earth_bins,
                                                          np.ndarray):
            edges = n_earth_bins

        dt_DM = []
        if hasattr(n_earth, 'sample'):
            # Sample if enterprise parameter object, else pass array
            n_earth = n_earth().sample()
        else:
            pass

        for ii, bin in enumerate(edges[:-1]):

            bin_mask = np.logical_and(toas >= bin, toas <= edges[ii + 1])
            earth = planetssb[bin_mask, 2, :3]
            R_earth = np.sqrt(np.einsum('ij,ij->i', earth, earth))
            Re_cos_theta_impact = np.einsum('ij,ij->i', earth, pos_t[bin_mask])
            theta = np.arccos(-Re_cos_theta_impact / R_earth)
            dm_sol_wind = dm_solar(n_earth[ii], theta, R_earth)

            if dm_sol_wind.size != 0:
                dt_DM.extend((dm_sol_wind)
                             * 4.148808e3 / freqs[bin_mask]**2)
            else:
                pass

        dt_DM = np.array(dt_DM)

    return dt_DM

# linear interpolation basis in time with nu^-2 scaling


@signal_base.function
def linear_interp_basis_sw_dm(toas, freqs, planetssb, pos_t, dt=7*86400):

    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    theta, R_earth = theta_impact(planetssb, pos_t)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 / (freqs**2)

    return U * dt_DM[:, None], avetoas


@signal_base.function
def createfourierdesignmatrix_solar_dm(toas, freqs, planetssb, pos_t,
                                       modes=None, nmodes=30,
                                       Tspan=None, logf=True, fmin=None,
                                       fmax=None):
    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: SW DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = utils.createfourierdesignmatrix_red(toas, nmodes=nmodes,
                                                    modes=modes,
                                                    Tspan=Tspan, logf=logf,
                                                    fmin=fmin, fmax=fmax)
    theta, R_earth = theta_impact(planetssb, pos_t)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 /(freqs**2)

    return F * dt_DM[:, None], Ffreqs


def solar_wind_block(n_earth=None, ACE_prior=False, include_swgp=True,
                     swgp_prior=None, swgp_basis=None, Tspan=None):
    """
    Returns Solar Wind DM noise model. Best model from Hazboun, et al (in prep)
        Contains a single mean electron density with an auxiliary perturbation
        modeled using a gaussian process. The GP has common prior parameters
        between all pulsars, but the realizations are different for all pulsars.

    Solar Wind DM noise modeled as a power-law with 30 sampling frequencies

    :param n_earth:
        Solar electron density at 1 AU.
    :param ACE_prior:
        Whether to use the ACE SWEPAM data as an astrophysical prior.
    :param swgp_prior:
        Prior function for solar wind Gaussian process. Default is a power law.
    :param swgp_basis:
        Basis to be used for solar wind Gaussian process.
        Options includes ['powerlaw'.'periodic','sq_exp']
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar. Default is to use 15
        frequencies (1/Tspan,15/Tspan).

    """

    if n_earth is None and not ACE_prior:
        n_earth = parameter.Uniform(0, 30)('n_earth')
    elif n_earth is None and ACE_prior:
        n_earth = ACE_SWEPAM_Parameter()('n_earth')
    else:
        pass

    deter_sw = solar_wind(n_earth=n_earth)
    mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
    sw_model = mean_sw

    if include_swgp:
        if swgp_basis == 'powerlaw':
            # dm noise parameters that are common
            log10_A_sw = parameter.Uniform(-10, 1)
            gamma_sw = parameter.Uniform(-2, 1)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)

            if Tspan is not None:
                freqs = np.linspace(1/Tspan, 30/Tspan, 30)
                freqs = freqs[1/freqs > 1.5*yr_in_sec]
                sw_basis = createfourierdesignmatrix_solar_dm(modes=freqs)
            else:
                sw_basis = createfourierdesignmatrix_solar_dm(nmodes=15,
                                                              Tspan=Tspan)

        elif swgp_basis == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_p = parameter.Uniform(-4, 1)
            log10_gam_p = parameter.Uniform(-3, 2)

            sw_basis = gpk.linear_interp_basis_dm(dt=6*86400)
            sw_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                           log10_ell=log10_ell,
                                           log10_gam_p=log10_gam_p,
                                           log10_p=log10_p)
        elif swgp_basis == 'sq_exp':
            # squared-exponential GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)

            sw_basis = gpk.linear_interp_basis_dm(dt=6*86400)
            sw_prior = gpk.se_dm_kernel(log10_sigma=log10_sigma,
                                        log10_ell=log10_ell)

        gp_sw = gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')
        sw_model += gp_sw

    return sw_model

##### Utility Functions #########


AU_light_sec = const.AU / const.c  # 1 AU in light seconds
AU_pc = const.AU / const.pc  # 1 AU in parsecs (for DM normalization)


def _dm_solar_close(n_earth, r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)


def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta))))


def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 20007 for more details.
    """
    return np.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))


def theta_impact(planetssb, pos_t):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    solar impact angle.

    ::param :planetssb Solar system barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.

    returns: Solar impact angle (rad), Distance to Earth
    """
    earth = planetssb[:, 2, :3]
    R_earth = np.sqrt(np.einsum('ij,ij->i', earth, earth))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earth, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)

    return theta_impact, R_earth


def sw_mask(psrs, angle_cutoff=None):
    """
    Convenience function for masking TOAs lower than a certain solar impact
        angle.
    param:: :psrs list of enterprise.Pulsar objects
    param:: :angle_cutoff (degrees) Mask TOAs within this angle

    returns:: dictionary of masks for each pulsar
    """
    solar_wind_mask = {}
    angle_cutoff = np.deg2rad(angle_cutoff)
    for ii, p in enumerate(psrs):
        impact_ang = theta_impact(p)
        solar_wind_mask[p.name] = np.where(impact_ang > angle_cutoff,
                                           True, False)

    return solar_wind_mask

# ACE Solar Wind Monitoring data prior for SW electron data.
# Using proton density as a stand in.


def ACE_SWEPAM_Prior(value):
    """Prior function for ACE SWEPAM parameters."""
    return ACE_RV.pdf(value)


def ACE_SWEPAM_Sampler(size=None):
    """Sampling function for Uniform parameters."""
    return ACE_RV.rvs(size=size)


def ACE_SWEPAM_Parameter(size=None):
    """Class factory for ACE SWEPAM parameters."""
    class ACE_SWEPAM_Parameter(parameter.Parameter):
        _size = size
        _typename = parameter._argrepr('ACE_SWEPAM')
        _prior = parameter.Function(ACE_SWEPAM_Prior)
        _sampler = staticmethod(ACE_SWEPAM_Sampler)

    return ACE_SWEPAM_Parameter

######## Scipy defined RV for ACE SWEPAM proton density data. ########


data_file = defpath + '/ACE_SWEPAM_daily_proton_density_1998_2018_MJD_cm-3.txt'
proton_density = np.loadtxt(data_file)
ne_hist = np.histogram(proton_density[:, 1], bins=100, density=True)
ACE_RV = sps.rv_histogram(ne_hist)
