# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
<<<<<<< HEAD
import scipy.stats
=======
from collections import OrderedDict
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise import constants as const

from enterprise_extensions import model_utils
<<<<<<< HEAD

# Extra model components not part of base enterprise ####

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

    if which == 'all':
        keys = tmparams_orig.keys()
    else:
        keys = which

    # grab original timing model parameters and errors in dictionary
    orig_params = np.array([tmparams_orig[key] for key in keys])

    # put varying parameters into dictionary
    tmparams_rescaled = np.atleast_1d(np.double(orig_params[:, 0] +
                                                tmparams * orig_params[:, 1]))
    tmparams_vary = OrderedDict(zip(keys, tmparams_rescaled))

    # set to new values
    t2pulsar.vals(tmparams_vary)
    new_res = np.double(t2pulsar.residuals().copy())

    # remember to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params[:, 0])))))

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
def periodic_kernel(avetoas, log10_sigma=-7, log10_ell=2, gam_p=1, p=1):

    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    p *= 3.16e7
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2) + d
    return K

# squared-exponential kernel for FD
@signal_base.function
def se_kernel(avefreqs, log10_sigma=-7, log10_lam=np.log10(1000)):
    tm = np.abs(avefreqs[None, :] - avefreqs[:, None])
    lam = 10**log10_lam
    sigma = 10**log10_sigma
    d = np.eye(tm.shape[0]) * (sigma/500)**2
    return sigma**2 * np.exp(-tm**2/2/lam) + d

# quantization matrix in time and radio frequency to cut down on the kernel size.
@signal_base.function
def get_tf_quantization_matrix(toas, freqs, dt=30*86400, df=None, dm=False):
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
         weights = (1400/freqs)**2
    else:
        weights = np.ones_like(freqs)

    return U[:, idx] * weights[:, None], {'avetoas': avetoas[idx],
                                          'avefreqs': avefreqs[idx]}

# kernel is the product of a quasi-periodic time kernel and
# a rational-quadratic frequency kernel.
@signal_base.function
def tf_kernel(labels, log10_sigma=-7, log10_ell=2, gam_p=1,
              p=1, log10_ell2=4, alpha_wgt=2):

    avetoas = labels['avetoas']
    avefreqs = labels['avefreqs']

    r = np.abs(avetoas[None, :] - avetoas[:, None])
    r2 = np.abs(avefreqs[None, :] - avefreqs[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * 86400
    l2 = 10**log10_ell2
    p *= 3.16e7
    d = np.eye(r.shape[0]) * (sigma/500)**2
    Kt = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2)
    Kv = (1+r2**2/2/alpha_wgt/l2**2)**(-alpha_wgt)

    return Kt * Kv + d

@signal_base.function
def jump_constant_offset(Amp=1e-6):
    ''' JUMP in the whole backend, with Amp in seconds,
    must be used with enterprise Selection to choose a backend '''
    return Amp

@signal_base.function
def log10_jump_constant_offset(Amp=-6):
    ''' JUMP in the whole backend, with Amp in seconds,
    must be used with enterprise Selection to choose a backend '''
    return 10**Amp

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
    wf = 10**log10_Amp * np.heaviside(toas - t0, 1) * \
        np.exp(- (toas - t0) / tau)

    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

@signal_base.function
def chrom_gaussian_bump(toas, freqs, log10_Amp=-2.5, sign_param=1.0,
                    t0=53890, sigma=81, idx=2):
    """
    Chromatic exponential-cusp delay term in TOAs.
    Example: J1603-7202 in Lentati et al, MNRAS 458, 2016.
    :param t0: time of Gaussian maximum [MJD]
    :param sigma: time of Gaussian maximum [MJD]
    :param log10_Amp: amplitude of the bump
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence
    :return wf: delay time-series [s]
    """
    t0 *= const.day
    sigma *= const.day
    wf = 10**log10_Amp * np.exp(-(toas - t0)**2/2/sigma**2)
    return np.sign(sign_param) * wf * (1400 / freqs) ** idx
  
@signal_base.function
def chrom_exp_cusp(toas, freqs, log10_Amp=-7, sign_param=-1.0,
                    t0=54000, log10_tau=1.7, idx=2):
    """
    Chromatic exponential-cusp delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau: 1/e time of exponential [s]
    :param log10_Amp: amplitude of cusp
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0 *= const.day
    tau = 10**log10_tau * const.day

    wf = (10**log10_Amp * np.heaviside(toas - t0, 1) *
          np.exp(- (toas - t0) / tau)) + (10**log10_Amp *
                                          (1 - np.heaviside(toas - t0, 1)) * np.exp(- (t0 - toas) / tau))

    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

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

@signal_base.function
def createfourierdesignmatrix_chromatic(toas, freqs, nmodes=30, Tspan=None,
                                        logf=False, fmin=None, fmax=None,
                                        idx=4):

    """
    Construct Scattering-variation fourier design matrix.

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param freqs: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param idx: Index of chromatic effects

    :return: F: Chromatic-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = utils.createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf,
        fmin=fmin, fmax=fmax)

    # compute the DM-variation vectors
    Dm = (1400/freqs) ** idx

    return F * Dm[:, None], Ffreqs

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

@signal_base.function
def free_spectrum(f, log10_rho=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10**(2*np.array(log10_rho)), 2)

@signal_base.function
def t_process(f, log10_A=-15, gamma=4.33, alphas=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    alphas = np.ones_like(f) if alphas is None else np.repeat(alphas, 2)
    return utils.powerlaw(f, log10_A=log10_A, gamma=gamma) * alphas

@signal_base.function
def t_process_adapt(f, log10_A=-15, gamma=4.33, alphas_adapt=None, nfreq=None):
    """
    t-process model. PSD  amplitude at each frequency
    is a fuzzy power-law.
    """
    if alphas_adapt is None:
        alpha_model = np.ones_like(f)
    else:
        if nfreq is None:
            alpha_model = np.repeat(alphas_adapt, 2)
        else:
            alpha_model = np.ones_like(f)
            alpha_model[2*int(np.rint(nfreq))] = alphas_adapt
            alpha_model[2*int(np.rint(nfreq))+1] = alphas_adapt

    return utils.powerlaw(f, log10_A=log10_A, gamma=gamma) * alpha_model

def InvGammaPrior(value, alpha=1, gamma=1):
    """Prior function for InvGamma parameters."""
    return scipy.stats.invgamma.pdf(value, alpha, scale=gamma)

def InvGammaSampler(alpha=1, gamma=1, size=None):
    """Sampling function for Uniform parameters."""
    return scipy.stats.invgamma.rvs(alpha, scale=gamma, size=size)

def InvGamma(alpha=1, gamma=1, size=None):
    """Class factory for Inverse Gamma parameters."""
    class InvGamma(parameter.Parameter):
        _size = size
        _prior = parameter.Function(InvGammaPrior, alpha=alpha, gamma=gamma)
        _sampler = staticmethod(InvGammaSampler)
        _alpha = alpha
        _gamma = gamma

        def __repr__(self):
            return '"{}": InvGamma({},{})'.format(self.name, alpha, gamma) \
                + ('' if self._size is None else '[{}]'.format(self._size))

    return InvGamma

@signal_base.function
def turnover_knee(f, log10_A, gamma, lfb, lfk, kappa, delta):
    """
    Generic turnover spectrum with a high-frequency knee.
    :param f: sampling frequencies of GWB
    :param log10_A: characteristic strain amplitude at f=1/yr
    :param gamma: negative slope of PSD around f=1/yr (usually 13/3)
    :param lfb: log10 transition frequency at which environment dominates GWs
    :param lfk: log10 knee frequency due to population finiteness
    :param kappa: smoothness of turnover (10/3 for 3-body stellar scattering)
    :param delta: slope at higher frequencies
    """
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    hcf = (10**log10_A * (f / const.fyr) ** ((3-gamma) / 2) *
           (1.0 + (f / 10**lfk)) ** delta / np.sqrt(1 + (10**lfb / f) ** kappa))
    return hcf**2 / 12 / np.pi**2 / f**3 * np.repeat(df, 2)

@signal_base.function
def generalized_gwpol_psd(f, log10_A_tt=-15, log10_A_st=-15,
                          log10_A_vl=-15, log10_A_sl=-15,
                          kappa=10/3, p_dist=1.0):
    """
    PSD for a generalized mixture of scalar+vector dipole radiation
    and tensorial quadrupole radiation from SMBHBs.
    """

    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    euler_e = 0.5772156649
    pdist = p_dist * const.kpc / const.c

    orf_aa_tt = (2/3) * np.ones(len(f))
    orf_aa_st = (2/3) * np.ones(len(f))
    orf_aa_vl = 2*np.log(4*np.pi*f*pdist) - 14/3 + 2*euler_e
    orf_aa_sl = np.pi**2*f*pdist/4 - \
        np.log(4*np.pi*f*pdist) + 37/24 - euler_e

    prefactor = (1 + kappa**2) / (1 + kappa**2 * (f / const.fyr)**(-2/3))
    gwpol_amps = 10**(2*np.array([log10_A_tt, log10_A_st,
                                  log10_A_vl, log10_A_sl]))
    gwpol_factors = np.array([orf_aa_tt*gwpol_amps[0],
                              orf_aa_st*gwpol_amps[1],
                              orf_aa_vl*gwpol_amps[2],
                              orf_aa_sl*gwpol_amps[3]])

    S_psd = prefactor * (gwpol_factors[0,:] * (f / const.fyr)**(-4/3) +
                         np.sum(gwpol_factors[1:,:],axis=0) *
                         (f / const.fyr)**(-2)) / \
    (8*np.pi**2*f**3)

    return S_psd * np.repeat(df, 2)

@signal_base.function
def dropout_powerlaw(f, log10_A=-16, gamma=5, k_drop=0.5, k_threshold=0.5):
    """
    Dropout powerlaw for a stochastic process. Switches a stochastic
    process on or off depending on whether k_drop exceeds k_threshold.
    """

    df = np.diff(np.concatenate((np.array([0]), f[::2])))

    if k_drop >= k_threshold:
        k_switch = 1.0
    elif k_drop < k_threshold:
        k_switch = 0.0

    return k_switch * ((10**log10_A)**2 / 12.0 / np.pi**2 *
                       const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, 2))

@signal_base.function
def dropout_physical_ephem_delay(toas, planetssb, pos_t, frame_drift_rate=0,
                                 d_jupiter_mass=0, d_saturn_mass=0, d_uranus_mass=0,
                                 d_neptune_mass=0, jup_orb_elements=np.zeros(6),
                                 sat_orb_elements=np.zeros(6), inc_jupiter_orb=False,
                                 jup_orbelxyz=None, jup_mjd=None, inc_saturn_orb=False,
                                 sat_orbelxyz=None, sat_mjd=None, equatorial=True,
                                 k_drop=0.5, k_threshold=0.5):
    """
    Dropout BayesEphem model. Switches BayesEphem on or off depending on
    whether k_drop exceeds k_threshold.
    """

    # get dropout switch
    if k_drop >= k_threshold:
        k_switch = 1.0
    elif k_drop < k_threshold:
        k_switch = 0.0

    # convert toas to MJD
    mjd = toas / 86400

    # grab planet-to-SSB vectors
    earth = planetssb[:, 2, :3]
    jupiter = planetssb[:, 4, :3]
    saturn = planetssb[:, 5, :3]
    uranus = planetssb[:, 6, :3]
    neptune = planetssb[:, 7, :3]

    # do frame rotation
    earth = utils.ss_framerotate(mjd, earth, 0.0, 0.0, 0.0, frame_drift_rate,
                           offset=None, equatorial=equatorial)

    # mass perturbations
    mpert = [(jupiter, d_jupiter_mass), (saturn, d_saturn_mass),
             (uranus, d_uranus_mass), (neptune, d_neptune_mass)]
    for planet, dm in mpert:
        earth += utils.dmass(planet, dm)

    # jupter orbital element perturbations
    if inc_jupiter_orb:
        jup_perturb_tmp = 0.0009547918983127075 * np.einsum(
            'i,ijk->jk', jup_orb_elements, jup_orbelxyz)
        earth += np.array([np.interp(mjd, jup_mjd, jup_perturb_tmp[:,aa])
                           for aa in range(3)]).T

    # saturn orbital element perturbations
    if inc_saturn_orb:
        sat_perturb_tmp = 0.00028588567008942334 * np.einsum(
            'i,ijk->jk', sat_orb_elements, sat_orbelxyz)
        earth += np.array([np.interp(mjd, sat_mjd, sat_perturb_tmp[:,aa])
                           for aa in range(3)]).T

    # construct the true geocenter to barycenter roemer
    tmp_roemer = np.einsum('ij,ij->i', planetssb[:, 2, :3], pos_t)

    # create the delay
    delay = tmp_roemer - np.einsum('ij,ij->i', earth, pos_t)

    return k_switch * delay


def Dropout_PhysicalEphemerisSignal(
    frame_drift_rate=parameter.Uniform(-1e-9, 1e-9)('frame_drift_rate'),
    d_jupiter_mass=parameter.Normal(0, 1.54976690e-11)('d_jupiter_mass'),
    d_saturn_mass=parameter.Normal(0, 8.17306184e-12)('d_saturn_mass'),
    d_uranus_mass=parameter.Normal(0, 5.71923361e-11)('d_uranus_mass'),
    d_neptune_mass=parameter.Normal(0, 7.96103855e-11)('d_neptune_mass'),
    jup_orb_elements=parameter.Uniform(-0.05,0.05,size=6)('jup_orb_elements'),
    sat_orb_elements=parameter.Uniform(-0.5,0.5,size=6)('sat_orb_elements'),
    inc_jupiter_orb=True, inc_saturn_orb=False, use_epoch_toas=True,
    k_drop=parameter.Uniform(0.0,1.0), k_threshold=0.5, name=''):

    """ Class factory for dropout physical ephemeris model signal."""

    # turn off saturn orbital element parameters if not including in signal
    if not inc_saturn_orb:
        sat_orb_elements = np.zeros(6)

    # define waveform
    jup_mjd, jup_orbelxyz, sat_mjd, sat_orbelxyz = (
        utils.get_planet_orbital_elements())
    wf = dropout_physical_ephem_delay(frame_drift_rate=frame_drift_rate,
                                        d_jupiter_mass=d_jupiter_mass,
                                        d_saturn_mass=d_saturn_mass,
                                        d_uranus_mass=d_uranus_mass,
                                        d_neptune_mass=d_neptune_mass,
                                        jup_orb_elements=jup_orb_elements,
                                        sat_orb_elements=sat_orb_elements,
                                        inc_jupiter_orb=inc_jupiter_orb,
                                        jup_orbelxyz=jup_orbelxyz,
                                        jup_mjd=jup_mjd,
                                        inc_saturn_orb=inc_saturn_orb,
                                        sat_orbelxyz=sat_orbelxyz,
                                        sat_mjd=sat_mjd,
                                        k_drop=k_drop, k_threshold=k_threshold)

    BaseClass = deterministic_signals.Deterministic(wf, name=name)

    class Dropout_PhysicalEphemerisSignal(BaseClass):
        signal_name = 'phys_ephem'
        signal_id = 'phys_ephem_' + name if name else 'phys_ephem'

        def __init__(self, psr):

            # not available for PINT yet
            if isinstance(psr, enterprise.pulsar.PintPulsar):
                msg = 'Physical Ephemeris model is not compatible with PINT '
                msg += 'at this time.'
                raise NotImplementedError(msg)

            super(Dropout_PhysicalEphemerisSignal, self).__init__(psr)

            if use_epoch_toas:
                # get quantization matrix and calculate daily average TOAs
                U, _ = utils.create_quantization_matrix(psr.toas, nmin=1)
                self.uinds = utils.quant2ind(U)
                avetoas = np.array([psr.toas[sc].mean() for sc in self.uinds])
                self._wf[''].add_kwarg(toas=avetoas)

                # interpolate ssb planet position vectors to avetoas
                planetssb = np.zeros((len(avetoas), 9, 3))
                for jj in range(9):
                    planetssb[:, jj, :] = np.array([
                        np.interp(avetoas, psr.toas, psr.planetssb[:,jj,aa])
                        for aa in range(3)]).T
                self._wf[''].add_kwarg(planetssb=planetssb)

                # Inteprolating the pulsar position vectors onto epoch TOAs
                pos_t = np.array([np.interp(avetoas, psr.toas, psr.pos_t[:,aa])
                                  for aa in range(3)]).T
                self._wf[''].add_kwarg(pos_t=pos_t)

            # initialize delay
            self._delay = np.zeros(len(psr.toas))

        @base.cache_call('delay_params')
        def get_delay(self, params):
            delay = self._wf[''](params=params)
            if use_epoch_toas:
                for slc, val in zip(self.uinds, delay):
                    self._delay[slc] = val
                return self._delay
            else:
                return delay

    return Dropout_PhysicalEphemerisSignal

@signal_base.function
def cw_delay(toas, pos, pdist,
             cos_gwtheta=0, gwphi=0, cos_inc=0,
             log10_mc=9, log10_fgw=-8, log10_dist=None, log10_h=None,
             phase0=0, psi=0,
             psrTerm=False, p_dist=1, p_phase=None,
             evolve=False, phase_approx=False, check=False,
             tref=0):
    """
    Function to create GW incuced residuals from a SMBMB as
    defined in Ellis et. al 2012,2013.
    :param toas:
        Pular toas in seconds
    :param pos:
        Unit vector from the Earth to the pulsar
    :param pdist:
        Pulsar distance (mean and uncertainty) [kpc]
    :param cos_gwtheta:
        Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi:
        Azimuthal angle of GW source in celestial coords [radians]
    :param cos_inc:
        cosine of Inclination of GW source [radians]
    :param log10_mc:
        log10 of Chirp mass of SMBMB [solar masses]
    :param log10_fgw:
        log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param log10_dist:
        log10 of Luminosity distance to SMBMB [Mpc],
        used to compute strain, if not None
    :param log10_h:
        log10 of GW strain,
        used to compute distance, if not None
    :param phase0:
        Initial Phase of GW source [radians]
    :param psi:
        Polarization angle of GW source [radians]
    :param psrTerm:
        Option to include pulsar term [boolean]
    :param p_dist:
        Pulsar distance parameter
    :param p_phase:
        Use pulsar phase to determine distance [radian]
    :param evolve:
        Option to include/exclude full evolution [boolean]
    :param phase_approx:
        Option to include/exclude phase evolution across observation time
        [boolean]
    :param check:
        Check if frequency evolves significantly over obs. time [boolean]
    :param tref:
        Reference time for phase and frequency [s]
    :return: Vector of induced residuals
    """

    # convert units to time
    mc = 10**log10_mc * const.Tsun
    fgw = 10**log10_fgw
    gwtheta = np.arccos(cos_gwtheta)
    inc = np.arccos(cos_inc)
    p_dist = (pdist[0] + pdist[1]*p_dist)*const.kpc/const.c

    if log10_h is None and log10_dist is None:
        raise ValueError("one of log10_dist or log10_h must be non-None")
    elif log10_h is not  None and log10_dist is not None:
        raise ValueError("only one of log10_dist or log10_h can be non-None")
    elif log10_h is None:
        dist = 10**log10_dist * const.Mpc / const.c
    else:
        dist = 2 * mc**(5/3) * (np.pi*fgw)**(2/3) / 10**log10_h

    if check:
        # check that frequency is not evolving significantly over obs. time
        fstart = fgw * (1 - 256/5 * mc**(5/3) * fgw**(8/3) * toas[0])**(-3/8)
        fend = fgw * (1 - 256/5 * mc**(5/3) * fgw**(8/3) * toas[-1])**(-3/8)
        df = fend - fstart

        # observation time
        Tobs = toas.max()-toas.min()
        fbin = 1/Tobs

        if np.abs(df) > fbin:
            print('WARNING: Frequency is evolving over more than one '
                  'frequency bin.')
            print('f0 = {0}, f1 = {1}, df = {2}, fbin = {3}'
                  .format(fstart, fend, df,  fbin))
            return np.ones(len(toas)) * np.nan

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(pos, gwtheta, gwphi)

    # get pulsar time
    toas -= tref
    if p_dist > 0:
        tp = toas-p_dist*(1-cosMu)
    else:
        tp = toas

    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    omegadot = 96/5 * mc**(5/3) * w0**(11/3)

    # evolution
    if evolve:
        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)

        if p_dist > 0:
            omega_p0 = w0 * (1 + 256/5
                             * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
        else:
            omega_p0 = w0

        # calculate time dependent phase
        phase = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

        if p_phase is None:
            phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3))
        else:
            phase_p = (phase0 + p_phase
                       + 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3)))

    elif phase_approx:
        # monochromatic
        omega = w0
        if p_dist > 0:
            omega_p = w0 * (1 + 256/5
                            * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
        else:
            omega_p = w0

        # phases
        phase = phase0 + omega * toas
        if p_phase is not None:
            phase_p = phase0 + p_phase + omega_p*toas
        else:
            phase_p = (phase0 + omega_p*toas
                       + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)))

    # no evolution
    else:
        # monochromatic
        omega = np.pi*fgw
        omega_p = omega

        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp

    # define time dependent coefficients
    At = -0.5*np.sin(2*phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(2*phase)*np.cos(inc)
    At_p = -0.5*np.sin(2*phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(2*phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))

    # define rplus and rcross
    rplus = alpha*(-At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(-At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # residuals
    if psrTerm:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res

@signal_base.function
def compute_eccentric_residuals(toas, theta, phi, cos_gwtheta, gwphi,
                                log10_mc, log10_dist, log10_h, log10_F, cos_inc,
                                psi, gamma0, e0, l0, q, nmax=400, pdist=1.0,
                                pphase=None, pgam=None, psrTerm=False,
                                tref=0, check=False):
    """
    Simulate GW from eccentric SMBHB. Waveform models from
    Taylor et al. (2015) and Barack and Cutler (2004).
    WARNING: This residual waveform is only accurate if the
    GW frequency is not significantly evolving over the
    observation time of the pulsar.
    :param toas: pulsar observation times
    :param theta: polar coordinate of pulsar
    :param phi: azimuthal coordinate of pulsar
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_mc: Base-10 lof of chirp mass of SMBMB [solar masses]
    :param log10_dist: Base-10 uminosity distance to SMBMB [Mpc]
    :param log10_F: base-10 orbital frequency of SMBHB [Hz]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param gamma0: Initial angle of periastron [radians]
    :param e0: Initial eccentricity of SMBHB
    :param l0: Initial mean anomoly [radians]
    :param q: Mass ratio of SMBHB
    :param nmax: Number of harmonics to use in waveform decomposition
    :param pdist: Pulsar distance [kpc]
    :param pphase: Pulsar phase [rad]
    :param pgam: Pulsar angle of periastron [rad]
    :param psrTerm: Option to include pulsar term [boolean]
    :param tref: Fidicuial time at which initial parameters are referenced [s]
    :param check: Check if frequency evolves significantly over obs. time
    :returns: Vector of induced residuals
    """

    # convert from sampling
    F = 10.0**log10_F
    mc = 10.0**log10_mc
    dist = 10.0**log10_dist
    if log10_h is not None:
        h0 = 10.0**log10_h
    else:
        h0 = None
    inc = np.arccos(cos_inc)
    gwtheta = np.arccos(cos_gwtheta)

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    # pulsar position vector
    phat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    # get values from pulsar object
    toas = toas.copy() - tref

    if check:
        # check that frequency is not evolving significantly over obs. time
        y = utils.solve_coupled_ecc_solution(F, e0, gamma0, l0, mc, q,
                                          np.array([0.0,toas.max()]))

        # initial and final values over observation time
        Fc0, ec0, gc0, phic0 = y[0,:]
        Fc1, ec1, gc1, phic1 = y[-1,:]

        # observation time
        Tobs = 1/(toas.max()-toas.min())

        if np.abs(Fc0-Fc1) > 1/Tobs:
            print('WARNING: Frequency is evolving over more than one frequency bin.')
            print('F0 = {0}, F1 = {1}, delta f = {2}'.format(Fc0, Fc1, 1/Tobs))
            return np.ones(len(toas)) * np.nan

    # get gammadot for earth term
    gammadot = utils.get_gammadot(F, mc, q, e0)

    # get number of harmonics to use
    if not isinstance(nmax, int):
        if e0 < 0.999 and e0 > 0.001:
            nharm = int(nmax(e0))
        elif e0 < 0.001:
            nharm = 2
        else:
            nharm = int(nmax(0.999))
    else:
        nharm = nmax

    # no more than 100 harmonics
    nharm = min(nharm, 100)

    # earth term #
    splus, scross = utils.calculate_splus_scross(nmax=nharm, mc=mc, dl=dist,
                                                 h0=h0, F=F, e=e0, t=toas.copy(),
                                                 l0=l0, gamma=gamma0,
                                                 gammadot=gammadot, inc=inc)

    # pulsar term #
    if psrTerm:
        # pulsar distance
        pd = pdist

        # convert units
        pd *= const.kpc / const.c

        # get pulsar time
        tp = toas.copy() - pd * (1-cosMu)

        # solve coupled system of equations to get pulsar term values
        y = utils.solve_coupled_ecc_solution(F, e0, gamma0, l0, mc,
                                       q, np.array([0.0, tp.min()]))

        # get pulsar term values
        if np.any(y):
            Fp, ep, gp, phip = y[-1,:]

            # get gammadot at pulsar term
            gammadotp = utils.get_gammadot(Fp, mc, q, ep)

            # get phase at pulsar
            if pphase is None:
                lp = phip
            else:
                lp = pphase

            # get angle of periastron at pulsar
            if pgam is None:
                gp = gp
            else:
                gp = pgam

            # get number of harmonics to use
            if not isinstance(nmax, int):
                if e0 < 0.999 and e0 > 0.001:
                    nharm = int(nmax(e0))
                elif e0 < 0.001:
                    nharm = 2
                else:
                    nharm = int(nmax(0.999))
            else:
                nharm = nmax

            # no more than 1000 harmonics
            nharm = min(nharm, 100)
            splusp, scrossp = utils.calculate_splus_scross(nmax=nharm, mc=mc,
                                                           dl=dist, h0=h0, F=Fp, e=ep,
                                                           t=toas.copy(), l0=lp, gamma=gp,
                                                           gammadot=gammadotp, inc=inc)

            rr = (fplus*cos2psi - fcross*sin2psi) * (splusp - splus) + \
                (fplus*sin2psi + fcross*cos2psi) * (scrossp - scross)

        else:
            rr = np.ones(len(toas)) * np.nan

    else:
        rr = - (fplus*cos2psi - fcross*sin2psi) * splus - \
            (fplus*sin2psi + fcross*cos2psi) * scross

    return rr

def CWSignal(cw_wf, ecc=False, psrTerm=False):

    BaseClass = deterministic_signals.Deterministic(cw_wf, name='cw')

    class CWSignal(BaseClass):

        def __init__(self, psr):
            super(CWSignal, self).__init__(psr)
            self._wf[''].add_kwarg(psrTerm=psrTerm)
            if ecc:
                pgam = parameter.Uniform(0, 2*np.pi)('_'.join([psr.name, 'pgam', 'cw']))
                self._params['pgam'] = pgam
                self._wf['']._params['pgam'] = pgam

    return CWSignal

@signal_base.function
def deterministic_solar_dm(toas, freqs, planetssb, pos_t,
                           log10_n_earth=None, n_earth_bins=None,
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
    if log10_n_earth is None: log10_n_earth = np.log10(8.7)

    n_earth = 10**log10_n_earth

    if n_earth_bins is None:
        earth = planetssb[:, 2, :3]
        R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
        Re_cos_theta_impact = np.einsum('ij,ij->i',earth, pos_t)

        theta_impact = np.arccos(-Re_cos_theta_impact/R_earth)

        dm_sol_wind = model_utils.dm_solar(n_earth,theta_impact,R_earth)
        dt_DM = (dm_sol_wind - dm_sol_wind.mean()) * 4.148808e3 / freqs**2

    else:
        if isinstance(n_earth_bins,int) and (t_init is None or t_final is None):
            raise ValueError('Need to enter t_init and t_final to make binned n_earth values.')

        elif isinstance(n_earth_bins, int):
            edges, step = np.linspace(t_init, t_final, n_earth_bins,
                                      endpoint=True, retstep=True)

        elif isinstance(n_earth_bins, list) or isinstance(n_earth_bins, array):
            edges = n_earth_bins
            step = np.mean(np.diff(edges))

        #print('Fitting {0} binned values of n_Earth of mean width {1}.'.format(n_earth_bins,step))

        dt_DM = []
        for ii, bin in enumerate(edges[:-1]):

            bin_mask = np.logical_and(toas>bin, toas<=edges[ii+1])
            earth = planetssb[bin_mask, 2, :3]
            R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
            Re_cos_theta_impact = np.einsum('ij,ij->i',earth, pos_t[bin_mask])

            theta_impact = np.arccos(-Re_cos_theta_impact/R_earth)
            dm_sol_wind = model_utils.dm_solar(n_earth[ii],theta_impact,R_earth)
            dt_DM.extend((dm_sol_wind - dm_sol_wind.mean()) * 4.148808e3 / freqs[bin_mask]**2)

        dt_DM = np.array(dt_DM)

    return dt_DM

#### Model component building blocks ####

def white_noise_block(vary=False, inc_ecorr=False, gp_ecorr=False,
                      efac1=False, select='backend'):
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
    ef = white_signals.MeasurementNoise(efac=efac, selection=backend)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=backend)
    if inc_ecorr:
        if gp_ecorr:
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ng)
        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ng)

    # combine signals
    if inc_ecorr:
        s = ef + eq + ec
    elif not inc_ecorr:
        s = ef + eq

    return s

def red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=None,
                    components=50, gamma_val=None, coefficients=False,
                    select=None, common=True, select_band_names=None):
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
    :param coefficients:
        include latent coefficients in GP model?
    :param select:
        Choose which TOAs to apply red noise model to
    :param common:
        Include a common component in addition to the red selection
    """
    # red noise parameters that are common
    if psd in ['powerlaw', 'turnover', 'tprocess', 'tprocess_adapt']:
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
        elif psd == 'turnover':
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            pl = utils.turnover(log10_A=log10_A, gamma=gamma,
                                 lf0=lf0, kappa=kappa)
        elif psd == 'tprocess':
            df = 2
            alphas = InvGamma(df/2, df/2, size=components)
            pl = t_process(log10_A=log10_A, gamma=gamma, alphas=alphas)
        elif psd == 'tprocess_adapt':
            df = 2
            alpha_adapt = InvGamma(df/2, df/2, size=1)
            nfreq = parameter.Uniform(-0.5, 10-0.5)
            pl = t_process_adapt(log10_A=log10_A, gamma=gamma,
                                 alphas_adapt=alpha_adapt, nfreq=nfreq)

    if psd == 'spectrum':
        if prior == 'uniform':
            log10_rho = parameter.LinearExp(-10, -4, size=components)
        elif prior == 'log-uniform':
            log10_rho = parameter.Uniform(-10, -4, size=components)

        pl = free_spectrum(log10_rho=log10_rho)

    if select == 'backend':
        # selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == 'band':
        # selection by observing band
        selection = selections.Selection(selections.by_band)
    elif select == 'freqs' or select == 'frequencies':
        # selection by observing frequency (f<=1Ghz, 1GHz<f<=2Ghz, and f>2Ghz)
        selection = selections.Selection(selections.by_frequency)
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    if common or (select is not None):
        # then we select on our selection or add a common process
        rn = gp_signals.FourierBasisGP(pl, components=components, Tspan=Tspan,
                                       coefficients=coefficients, selection=selection)
    if select_band_names is not None:
         # We add red noise for each band in select_band_names
             band = select_band_names
             band = band.lower()
             selection = return_band_selection_class_from_name(band)

             rn = gp_signals.FourierBasisGP(pl, components=components, Tspan=Tspan,

                                             coefficients=coefficients,
                                             selection=selection, name='red_noise_'+band)
    else:
        rn2 = None

        if common or (select is not None):
            if rn2 is not None:
                rn += rn2
        else:
            rn = rn2
            
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
                alphas_dm = InvGamma(df/2, df/2, size=components)
                dm_prior = t_process(log10_A=log10_A_dm, gamma=gamma_dm, alphas=alphas_dm)
            elif psd == 'tprocess_adapt':
                df = 2
                alpha_adapt_dm = InvGamma(df/2, df/2, size=1)
                nfreq_dm = parameter.Uniform(-0.5, 10-0.5)
                dm_prior = t_process_adapt(log10_A=log10_A_dm, gamma=gamma_dm,
                                           alphas_adapt=alpha_adapt_dm, nfreq=nfreq_dm)

        if psd == 'spectrum':
            if prior == 'uniform':
                log10_rho_dm = parameter.LinearExp(-10, -4, size=components)
            elif prior == 'log-uniform':
                log10_rho_dm = parameter.Uniform(-10, -4, size=components)

            dm_prior = free_spectrum(log10_rho=log10_rho_dm)

        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=components,
                                                      Tspan=Tspan)

    elif gp_kernel == 'nondiag':
        if nondiag_kernel == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            period = parameter.Uniform(0.2, 5.0)
            gam_p = parameter.Uniform(0.1, 30.0)

            dm_basis = linear_interp_basis_dm(dt=15*86400)
            dm_prior = periodic_kernel(log10_sigma=log10_sigma,
                                    log10_ell=log10_ell, gam_p=gam_p, p=period)
        elif nondiag_kernel == 'periodic_rfband':
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            alpha_wgt = parameter.Uniform(0.2, 6)
            period = parameter.Uniform(0.2, 5.0)
            gam_p = parameter.Uniform(0.1, 30.0)

            dm_basis = get_tf_quantization_matrix(df=200, dt=15*86400, dm=True)
            dm_prior = tf_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                 gam_p=gam_p, p=period, alpha_wgt=alpha_wgt,
                                 log10_ell2=log10_ell2)
        elif nondiag_kernel == 'dmx_like':
            # DMX-like signal
            log10_sigma = parameter.Uniform(-10, -4)

            dm_basis = linear_interp_basis_dm(dt=30*86400)
            dm_prior = dmx_ridge_prior(log10_sigma=log10_sigma)

    dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp',
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

def dm_gaussian_bump(tmin, tmax, idx=2, sigma_min=20, sigma_max=140,
    log10_A_low=-6, log10_A_high=-1, sign=False, name='dm_bump'):
    """
    Returns chromatic Gaussian bump (i.e. TOA advance):
    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sigma_min, sigma_max:
        standard deviation of a Gaussian in MJD
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal
    :return dm_bump:
        chromatic Gaussian bump waveform.
    """
    t0_dm_bump = parameter.Uniform(tmin,tmax)
    sigma_dm_bump = parameter.Uniform(sigma_min,sigma_max)
    log10_Amp_dm_bump = parameter.Uniform(log10_A_low, log10_A_high)
    if sign:
        sign_param = parameter.Uniform(-1.0, 1.0)
    else:
        sign_param = 1.0
    wf = chrom_gaussian_bump(log10_Amp=log10_Amp_dm_bump,
                         t0=t0_dm_bump, sigma=sigma_dm_bump,
                         sign_param=sign_param, idx=idx)
    dm_bump = deterministic_signals.Deterministic(wf, name=name)

    return dm_bump

def dm_exponential_dip(tmin, tmax, idx=2, sign=False, name='dmexp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(-10, -2)
    log10_tau_dmexp = parameter.Uniform(np.log10(5), np.log10(100))
    if sign:
        sign_param = parameter.Uniform(-1.0, 1.0)
    else:
        sign_param = -1.0
    wf = chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp
  
def jump_in_backend(prior,prior_bounds,band_selection_name=None):
    ''' Time-independent JUMP in backend: a constant offset in one backend '''
    if band_selection_name==None:
        print('Jump with no selection makes no sense. Please add a selection.')
        raise ValueError
    else:
        selection=selections.Selection(selections.custom_selection)
        #selection=return_band_selection_class_from_name(band_selection_name)
    if prior=='uniform':
      Amp = parameter.Uniform(prior_bounds[0], prior_bounds[1])
    elif prior=='normal':
      Amp = parameter.Normal(mu=prior_bounds[0], sigma=prior_bounds[1])
    name = 'jump_'+band_selection_name

    wf = jump_constant_offset(Amp=Amp)

    backend_jump = deterministic_signals.Deterministic(wf, name=name, selection=selection) 

    return backend_jump

def magnetosphere_exponential_dip(tmin, tmax, idx='vary', sign=False,
                                  name='magneto_exp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from -7 to 7
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(-10, -2)
    log10_tau_dmexp = parameter.Uniform(np.log10(2), np.log10(1000))
    if idx == 'vary':
        idx = parameter.Uniform(-7, 7)
    if sign:
        sign_param = parameter.Uniform(-1.0, 1.0)
    else:
        sign_param = -1.0
    wf = chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    magnetoexp = deterministic_signals.Deterministic(wf, name=name)

    return magnetoexp
  
def dm_exponential_cusp(tmin, tmax, idx=2, sign=False, name='dm_cusp'):
    """
    Returns chromatic exponential cusp (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dm_cusp = parameter.Uniform(tmin,tmax)
    log10_Amp_dm_cusp = parameter.Uniform(-10, -2)
    log10_tau_dm_cusp = parameter.Uniform(np.log10(5), np.log10(100))
    if sign:
        sign_param = parameter.Uniform(-1.0, 1.0)
    else:
        sign_param = 1.0
    wf = chrom_exp_cusp(log10_Amp=log10_Amp_dm_cusp,
                         t0=t0_dm_cusp, log10_tau=log10_tau_dm_cusp,
                         sign_param=sign_param, idx=idx)
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

def chromatic_noise_block(psd='powerlaw', idx=4, Tspan=None,
                          name='chromatic', components=30,
                          coefficients=False, quadratic=False):
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
        log10_rho = parameter.LinearExp(-9, -4, size=components)
        cpl = free_spectrum(log10_rho=log10_rho)

    # set up signal
    if idx == 'vary':
        idx = parameter.Uniform(0, 6)

    # quadratic piece
    basis_quad = chromatic_quad_basis(idx=idx)
    prior_quad = chromatic_quad_prior()
    cquad = gp_signals.BasisGP(prior_quad, basis_quad, name=name+'_quad')

    # Fourier piece
    basis_gp = createfourierdesignmatrix_chromatic(nmodes=components,
                                                   Tspan=Tspan, idx=idx)
    cgp = gp_signals.BasisGP(cpl, basis_gp, name=name+'_gp',
                             coefficients=coefficients)

    if quadratic:
      return cquad + cgp
    else:
      return cgp

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
            cpl = turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                lfb=lfb_gw, lfk=lfk_gw,
                                kappa=kappa_gw, delta=delta_gw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)
        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(-9, -4, size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(-9, -4, size=components)(rho_name)

        cpl = free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients,
                                        components=components, Tspan=Tspan,
                                        name=name)
    elif orf in orfs.keys():
        crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf],
                                              components=components, Tspan=Tspan,
                                              name=name)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn

def bwm_block(Tmin, Tmax, amp_prior='log-uniform',
              skyloc=None, logmin=-18, logmax=-11,
              name='bwm'):
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
    amp_name = '{}_log10_A'.format(name)
    if amp_prior == 'uniform':
        log10_A_bwm = parameter.LinearExp(logmin, logmax)(amp_name)
    elif amp_prior == 'log-uniform':
        log10_A_bwm = parameter.Uniform(logmin, logmax)(amp_name)

    pol_name = '{}_pol'.format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    t0_name = '{}_t0'.format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)


    # BWM signal
    bwm_wf = utils.bwm_delay(log10_h=log10_A_bwm, t0=t0,
                            cos_gwtheta=costh, gwphi=phi, gwpol=pol)
    bwm = deterministic_signals.Deterministic(bwm_wf, name=name)

    return bwm

def cw_block_circ(amp_prior='log-uniform', dist_prior=None,
                  skyloc=None, log10_fgw=None,
                  psrTerm=False, tref=0, name='cw'):
    """
    Returns deterministic, cirular orbit continuous GW model:
    :param amp_prior:
        Prior on log10_h. Default is "log-uniform."
        Use "uniform" for upper limits, or "None" to search over
        log10_dist instead.
    :param dist_prior:
        Prior on log10_dist. Default is "None," meaning that the
        search is over log10_h instead of log10_dist. Use "log-uniform"
        to search over log10_h with a log-uniform prior.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_fgw:
        Fixed log10 GW frequency of CW signal search.
        Search over GW frequency if ``None`` given.
    :param ecc:
        Fixed log10 distance to SMBHB search.
        Search over distance or strain if ``None`` given.
    :param psrTerm:
        Boolean for whether to include the pulsar term. Default is False.
    :param name:
        Name of CW signal.
    """

    if dist_prior == None:
        log10_dist = None

        if amp_prior == 'uniform':
            log10_h = parameter.LinearExp(-18.0, -11.0)('{}_log10_h'.format(name))
        elif amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(-18.0, -11.0)('{}_log10_h'.format(name))

    elif dist_prior == 'log-uniform':
        log10_dist = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))
        log10_h = None

    # chirp mass [Msol]
    log10_Mc = parameter.Uniform(6.0, 10.0)('{}_log10_Mc'.format(name))

    # GW frequency [Hz]
    if log10_fgw is None:
        log10_fgw = parameter.Uniform(-9.0, -7.0)('{}_log10_fgw'.format(name))
    else:
        log10_fgw = parameter.Constant(log10_fgw)('{}_log10_fgw'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # initial GW phase [radians]
    phase0 = parameter.Uniform(0.0, np.pi)('{}_phase0'.format(name))

    # polarization
    psi_name = '{}_psi'.format(name)
    psi = parameter.Uniform(0, np.pi)(psi_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    if psrTerm:
        p_phase = parameter.Uniform(0, 2*np.pi)
        p_dist = parameter.Normal(0, 1)
    else:
        p_phase = None
        p_dist = 0

    # continuous wave signal
    wf = cw_delay(cos_gwtheta=costh, gwphi=phi, cos_inc=cosinc,
                  log10_mc=log10_Mc, log10_fgw=log10_fgw,
                  log10_h=log10_h, log10_dist=log10_dist,
                  phase0=phase0, psi=psi,
                  psrTerm=True, p_dist=p_dist, p_phase=p_phase,
                  phase_approx=True, check=False,
                  tref=tref)
    cw = CWSignal(wf, ecc=False, psrTerm=psrTerm)

    return cw

def cw_block_ecc(amp_prior='log-uniform', skyloc=None, log10_F=None,
                 ecc=None, psrTerm=False, tref=0, name='cw'):
    """
    Returns deterministic, eccentric orbit continuous GW model:
    :param amp_prior:
        Prior on log10_h and log10_Mc/log10_dL. Default is "log-uniform" with
        log10_Mc and log10_dL searched over. Use "uniform" for upper limits,
        log10_h searched over.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_F:
        Fixed log-10 orbital frequency of CW signal search.
        Search over orbital frequency if ``None`` given.
    :param ecc:
        Fixed eccentricity of SMBHB search.
        Search over eccentricity if ``None`` given.
    :param psrTerm:
        Boolean for whether to include the pulsar term. Default is False.
    :param name:
        Name of CW signal.
    """

    if amp_prior == 'uniform':
        log10_h = parameter.LinearExp(-18.0, -11.0)('{}_log10_h'.format(name))
    elif amp_prior == 'log-uniform':
        log10_h = None
    # chirp mass [Msol]
    log10_Mc = parameter.Uniform(6.0, 10.0)('{}_log10_Mc'.format(name))
    # luminosity distance [Mpc]
    log10_dL = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))

    # orbital frequency [Hz]
    if log10_F is None:
        log10_Forb = parameter.Uniform(-9.0, -7.0)('{}_log10_Forb'.format(name))
    else:
        log10_Forb = parameter.Constant(log10_F)('{}_log10_Forb'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # periapsis position angle [radians]
    gamma_0 = parameter.Uniform(0.0, np.pi)('{}_gamma0'.format(name))

    # Earth-term eccentricity
    if ecc is None:
        e_0 = parameter.Uniform(0.0, 0.99)('{}_e0'.format(name))
    else:
        e_0 = parameter.Constant(ecc)('{}_e0'.format(name))

    # initial mean anomaly [radians]
    l_0 = parameter.Uniform(0.0, 2.0*np.pi)('{}_l0'.format(name))
    # mass ratio = M_2/M_1
    q = parameter.Constant(1.0)('{}_q'.format(name))

    # polarization
    pol_name = '{}_pol'.format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    # continuous wave signal
    wf = compute_eccentric_residuals(cos_gwtheta=costh, gwphi=phi,
                                     log10_mc=log10_Mc, log10_dist=log10_dL,
                                     log10_h=log10_h, log10_F=log10_Forb,
                                     cos_inc=cosinc, psi=pol, gamma0=gamma_0,
                                     e0=e_0, l0=l_0, q=q, nmax=400,
                                     pdist=None, pphase=None, pgam=None,
                                     tref=tref, check=False)
    cw = CWSignal(wf, ecc=True, psrTerm=psrTerm)

    return cw

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

#### PTA models from paper ####

###############################
###  PTA models from paper  ###
###############################

def model_singlepsr_noise(psr, red_var=False, psd='powerlaw',
=======
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
                          tmparam_list=None,
                          red_var=True, psd='powerlaw', red_select=None,
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
                          noisedict=None, tm_svd=False, tm_norm=True,
                          white_vary=True, components=50, upper_limit=False,
                          wideband=False, gamma_val=None, dm_var=False,
                          dm_type='gp', dmgp_kernel='diag', dm_psd='powerlaw',
                          dm_nondiag_kernel='periodic', dmx_data=None,
<<<<<<< HEAD
                          dm_annual=False, dm_solar=False, gamma_dm_val=None, dm_chrom=False,
                          dmchrom_psd='powerlaw', dmchrom_idx=4,
                          dm_expdip=False, dmexp_sign=False, dm_expdip_idx=2,
                          dm_expdip_tmin=None, dm_expdip_tmax=None,
                          dm_gauss = False, dm_gauss_tmin=None, dm_gauss_tmax=None,  
                          num_dmdips=1, dmdip_seqname=None, dm_cusp=False, 
                          dm_cusp_sign=False, dm_cusp_idx=2, dm_cusp_tmin=None, 
                          dm_cusp_tmax=None, magn_expdip=False, magnexp_sign=False, 
                          magn_expdip_tmin=None, magn_expdip_tmax=None, num_magndips=1,
                          magndip_seqname=None, coefficients=False, red_select=None,
                          select_band_names=None, red_common=True,
                          constant_jump_name=None,constant_jump_prior=None,
                          constant_jump_prior_bounds=None):

=======
                          dm_annual=False, gamma_dm_val=None, chrom_gp=False,
                          chrom_gp_kernel='nondiag',
                          chrom_psd='powerlaw', chrom_idx=4,
                          chrom_kernel='periodic',
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
                          coefficients=False, extra_sigs=None):
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
    """
    Single pulsar noise model
    :param psr: enterprise pulsar object
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
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    tmin = psr.toas.min() / 86400
    tmax = psr.toas.max() / 86400
    components = int((tmax-tmin)/60)  # components down to 60 days

    # timing model
    s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                               coefficients=coefficients)

    if constant_jump_name is not None:
        s += jump_in_backend(constant_jump_prior,constant_jump_prior_bounds,\
                             band_selection_name=constant_jump_name)

    # red nois/
    if red_var and red_common is not None:
        s += red_noise_block(psd=psd, prior=amp_prior,
<<<<<<< HEAD
                            components=components, gamma_val=gamma_val,
                            coefficients=coefficients, select=red_select,
                            select_band_names=None, common=red_common)

    if red_var and select_band_names is not None:
      for band in select_band_names:
        s += red_noise_block(psd=psd, prior=amp_prior,
                            components=components, gamma_val=gamma_val,
                            coefficients=coefficients, select=red_select,
                            select_band_names=band, common=red_common)
=======
                             components=components, gamma_val=gamma_val,
                             coefficients=coefficients, select=red_select)
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

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
            s += chrom.dmx_signal(dmx_data=dmx_data[psr.name])
        if dm_annual:
<<<<<<< HEAD
            s += dm_annual_signal()
        if dm_solar:
            s += solar_dm_block()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic', components=components,
=======
            s += chrom.dm_annual_signal()
        if chrom_gp:
            s += chromatic_noise_block(gp_kernel=chrom_gp_kernel,
                                       psd=chrom_psd, idx=chrom_idx,
                                       components=components,
                                       nondiag_kernel=chrom_kernel,
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
                                       coefficients=coefficients)

        if dm_expdip:
<<<<<<< HEAD
=======
            if dm_expdip_tmin is None and dm_expdip_tmax is None:
                tmin = [psr.toas.min() / 86400 for ii in range(num_dmdips)]
                tmax = [psr.toas.max() / 86400 for ii in range(num_dmdips)]
            else:
                tmin = (dm_expdip_tmin if isinstance(dm_expdip_tmin,list)
                                     else [dm_expdip_tmin])
                tmax = (dm_expdip_tmax if isinstance(dm_expdip_tmax,list)
                                     else [dm_expdip_tmax])
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
            if dmdip_seqname is not None:
                dmdipname_base = (['dmexp_' + nm for nm in dmdip_seqname]
                                   if isinstance(dmdip_seqname,list)
                                   else ['dmexp_' + dmdip_seqname])
            else:
<<<<<<< HEAD
                dmdipname_base = 'dmexp_'

            if len(dm_expdip_tmin)==len(dm_expdip_tmax):
                num_dmdips = len(dm_expdip_tmin)
            else:
                print('DM exp dip: number of start times and end times should be the same!')
                raise ValueError
            if not hasattr(dm_expdip_idx,'__iter__'):
                dm_expdip_idx = [dm_expdip_idx for ii in range(num_dmdips)]
            if not hasattr(dmexp_sign,'__iter__'):
                dmexp_sign = [dmexp_sign for ii in range(num_dmdips)]
            if not hasattr(dm_expdip_tmin,'__iter__'):
                dm_expdip_tmin = [dm_expdip_tmin for ii in range(num_dmdips)]
            if not hasattr(dm_expdip_tmax,'__iter__'):
                dm_expdip_tmax = [dm_expdip_tmax for ii in range(num_dmdips)]

            for dd in range(num_dmdips):
                if dm_expdip_tmin[dd] is None and dm_expdip_tmax[dd] is None:
                    tmin = psr.toas.min() / 86400
                    tmax = psr.toas.max() / 86400
                else:
                    tmin = dm_expdip_tmin[dd]
                    tmax = dm_expdip_tmax[dd]

                s += dm_exponential_dip(tmin=tmin, tmax=tmax,
                                        idx=dm_expdip_idx[dd],
                                        sign=dmexp_sign[dd],
                                        name=dmdipname_base+str(dd))

        #if dm_expdip_2:
        #    if dm_expdip_tmin_2 is None and dm_expdip_tmax_2 is None:
        #        tmin_2 = psr.toas.min() / 86400
        #        tmax_2 = psr.toas.max() / 86400
        #    else:
        #        tmin_2 = dm_expdip_tmin_2
        #        tmax_2 = dm_expdip_tmax_2
        #    dmdipname_base_2 = 'dmexp2_'
        #    for dd in range(1,num_dmdips+1):
        #        s += dm_exponential_dip(tmin=tmin_2, tmax=tmax_2,
        #                                idx=dm_expdip_idx_2,
        #                                sign=dmexp_sign_2,
        #                                name=dmdipname_base_2+str(dd))

        if dm_gauss:
            if dm_gauss_tmin is None and dm_gauss_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = dm_gauss_tmin
                tmax = dm_gauss_tmax
            dmgaussname_base = 'dm_gauss_'
            #for dd in range(1,num_dmgauss+1):
            s += dm_gaussian_bump(tmin=tmin, tmax=tmax,
                                        sigma_min=20, sigma_max=140, 
                                        log10_A_low=-6, log10_A_high=-1,
                                        name=dmgaussname_base)#+str(dd))

        if magn_expdip:
            if magn_expdip_tmin is None and magn_expdip_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = magn_expdip_tmin
                tmax = magn_expdip_tmax
            if magndip_seqname is not None:
                magndipname_base = 'magnexp_'+magndip_seqname+'_'
            else:
                magndipname_base = 'magnexp_'
            for dd in range(1,num_magndips+1):
                s += magnetosphere_exponential_dip(tmin=tmin, tmax=tmax, 
                                                   idx='vary', sign=magnexp_sign,
                                                   name=magndipname_base+str(dd))

=======
                dmdipname_base = ['dmexp_{0}'.format(ii+1)
                                  for ii in range(num_dmdips)]

            dm_expdip_idx = (dm_expdip_idx if isinstance(dm_expdip_idx,list)
                                           else [dm_expdip_idx])
            for dd in range(num_dmdips):
                s += chrom.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                              idx=dm_expdip_idx[dd],
                                              sign=dmexp_sign,
                                              name=dmdipname_base[dd])
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
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
             n_rnfreqs = None, n_gwbfreqs=None,
             gamma_common=None, upper_limit=False, bayesephem=False,
             be_type='orbel', wideband=False, select='backend',
             pshift=False, pseed=None, psr_models=False):
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
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
                                name='gw', pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type)

    # timing model
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True,
                                       select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False,
                                       select=select)
            models.append(s3(p))

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
                  common_psd='powerlaw', red_psd='powerlaw', orf=None,
                  common_components=30, red_components=30, dm_components=30,
                  modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  noisedict=None,
                  tm_svd=False, tm_norm=True, gamma_common=None,
                  upper_limit=False, upper_limit_red=None, upper_limit_dm=None,
                  upper_limit_common=None,
                  bayesephem=False, be_type='orbel', wideband=False,
                  dm_var=False, dm_type='gp', dm_psd='powerlaw',
                  dm_annual=False,
                  white_vary=False, gequad=False, dm_chrom=False,
                  dmchrom_psd='powerlaw', dmchrom_idx=4,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  coefficients=False, pshift=False):
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
    :param be_type:
        orbel, orbel-v2, setIII
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # timing model
    s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                coefficients=coefficients)

    if constant_jump_name is not None:
        s += jump_in_backend(constant_jump_amin,constant_jump_amax,\
                             band_selection_name=constant_jump_name)

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

<<<<<<< HEAD
    # red noise
    s += red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                        components=components, coefficients=coefficients,
                        select=red_select)

    # common red noise block
    if orf is None:
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, coefficients=coefficients,
                                    gamma_val=gamma_common, name='red')
    elif orf == 'hd':
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, gamma_val=gamma_common,
                                    orf='hd', name='gw', coefficients=coefficients)
    elif orf == 'dipole':
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, gamma_val=gamma_common,
                                    orf='dipole', name='dipole')
    elif orf == 'monopole':
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, gamma_val=gamma_common,
                                    orf='monopole', name='monopole')
    elif orf == 'monopole+':
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, coefficients=coefficients,
                                    gamma_val=gamma_common, name='red')
        s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                    components=components, gamma_val=gamma_common,
                                    orf='monopole', name='monopole')
=======
    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(Tspan, nmodes_log,
                                             1.0 / fmin / Tspan,
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
        s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                    Tspan=Tspan,
                                    components=common_components,
                                    coefficients=coefficients, pshift=pshift,
                                    gamma_val=gamma_common, name='gw')
    elif orf == 'hd':
        s += common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                    Tspan=Tspan,
                                    components=common_components,
                                    coefficients=coefficients,
                                    gamma_val=gamma_common, orf='hd',
                                    name='gw')
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

    # DM variations
    if dm_var:
        if dm_type == 'gp':
<<<<<<< HEAD
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd, prior=amp_prior,
                                components=components, gamma_val=None,
=======
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd,
                                prior=amp_prior_dm,
                                components=dm_components, gamma_val=None,
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
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
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type)

    # adding white-noise, and extra events and red terms acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
<<<<<<< HEAD
        else:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
        if '1713' in p.name and chrom_events:
            # Add DM exponential dip
            tmin = p.toas.min() / 86400
            tmax = p.toas.max() / 86400
            s2 += dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                     sign=False, name='dmexp')
        elif ('1643' in p.name or '0437' in p.name) and chrom_events:
            # Add chromatic exponential dip for profile shape change events
            tmin = p.toas.min() / 86400
            tmax = p.toas.max() / 86400
            s2 += magnetosphere_exponential_dip(tmin=tmin, tmax=tmax,
                                                sign=False, idx='vary',
                                                name=p.name+'_magexp')
        # Add extra red noise terms
        if select_psrs is not None:
            for psrname, select_bands in select_psrs.items():
                if psrname in p.name:
                    s2 += red_noise_block(psd=psd, prior=amp_prior,
                                          components=components, coefficients=coefficients,
                                          select_band_names=select_bands,
                                          common=False)
        models.append(s2(p))
=======
            if gequad:
                s2 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / 86400
                tmax = p.toas.max() / 86400
                s3 = s2 + chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
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
                s5 = s4 + chrom.dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                                   sign=False, name='dmexp')
                models.append(s5(p))
            else:
                models.append(s4(p))
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

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
             wideband=False, pshift=False):
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
                                orf='dipole', name='dipole', pshift=pshift)

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
             wideband=False, pshift=False):
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
                                orf='monopole', name='monopole', pshift=pshift)

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
             n_rnfreqs = None, n_gwbfreqs=None,
             gamma_common=None, upper_limit=False, bayesephem=False,
             be_type='orbel', wideband=False, correlationsonly=False,
             pshift=False, pseed=None, psr_models=False):
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
    :param be_type:
        orbel, orbel-v2, setIII
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
                                orf='hd', name='gw', pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type)

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
                                name='gw', pshift=pshift)

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True,
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
                      wideband=False, k_threshold=0.5, pshift=False):
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

    k_drop = parameter.Uniform(0.0, 1.0)  # per-pulsar

    drop_pl = do.dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                  k_drop=k_drop, k_threshold=k_threshold)
    crn = gp_signals.FourierBasisGP(drop_pl, components=components,
                                    Tspan=Tspan, name='gw', pshift=pshift)
    s += crn

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True)

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


# Does not yet work with IPTA datasets due to white-noise modeling issues.
def model_chromatic(psrs, psd='powerlaw', noisedict=None, components=30,
                    gamma_common=None, upper_limit=False, bayesephem=False,
                    wideband=False, pshift=False,
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
                                name='gw', pshift=pshift)

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
    s += gp_signals.TimingModel()

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True,
                                       gp_ecorr=True)
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

def return_band_selection_class_from_name(band):
    ''' Return selection-by-band class from band name '''
    if band == '50cm' or band == 'band_50cm' or band == 'by_50cm':
        # selection by only 50cm data
        selection = selections.Selection(selections.band_50cm)
    if band == '40cm' or band == 'band_40cm' or band == 'by_40cm':
        # selection by only 40cm data
        selection = selections.Selection(selections.band_40cm)
    if band == '4050cm' or band == 'band_4050cm' or band == 'by_4050cm':
        # selection by only 40cm and 50cm data
        selection = selections.Selection(selections.band_4050cm)
    if band == '20cm' or band == 'band_20cm' or band == 'by_20cm':
        # selection by only 20cm data
        selection = selections.Selection(selections.band_20cm)
    if band == '10cm' or band == 'band_10cm' or band == 'by_10cm':
        # selection by only 10cm data
        selection = selections.Selection(selections.band_10cm)
    if band == 'caspsr':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_caspsr)
    if band == 'cpsr2':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_cpsr2)
    if band == 'cpsr2_20cm':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_cpsr2_20cm)
    if band == 'cpsr2_50cm':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_cpsr2_50cm)
    if band == 'wbcorr':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_wbcorr)
    if band == 'jflag_pdfb4_20cm':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_jflag_pdfb4_20cm)
    if band == 'pdfb1_early':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_pdfb1_early)
    if band == 'pdfb1':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_pdfb1)
    if band == 'pdfb':
        # selection by only 10cm data
        selection = selections.Selection(selections.system_pdfb)
    return selection
