# -*- coding: utf-8 -*-

import enterprise
import numpy as np
from enterprise import constants as const
from enterprise.signals import (deterministic_signals, parameter, signal_base,
                                utils)


@signal_base.function
def dropout_powerlaw(f, name, log10_A=-16, gamma=5,
                     dropout_psr='B1855+09', k_drop=0.5, k_threshold=0.5):
    """
    Dropout powerlaw for a stochastic process. Switches a stochastic
    process on or off in a single pulsar depending on whether k_drop exceeds
    k_threshold.
    """

    df = np.diff(np.concatenate((np.array([0]), f[::2])))

    if name == dropout_psr:

        if k_drop >= k_threshold:
            k_switch = 1.0
        elif k_drop < k_threshold:
            k_switch = 0.0

        return k_switch * ((10**log10_A)**2 / 12.0 / np.pi**2 *
                           const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, 2))

    else:

        return ((10**log10_A)**2 / 12.0 / np.pi**2 *
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
        earth += np.array([np.interp(mjd, jup_mjd, jup_perturb_tmp[:, aa])
                           for aa in range(3)]).T

    # saturn orbital element perturbations
    if inc_saturn_orb:
        sat_perturb_tmp = 0.00028588567008942334 * np.einsum(
            'i,ijk->jk', sat_orb_elements, sat_orbelxyz)
        earth += np.array([np.interp(mjd, sat_mjd, sat_perturb_tmp[:, aa])
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
        jup_orb_elements=parameter.Uniform(-0.05, 0.05, size=6)('jup_orb_elements'),
        sat_orb_elements=parameter.Uniform(-0.5, 0.5, size=6)('sat_orb_elements'),
        inc_jupiter_orb=True, inc_saturn_orb=False, use_epoch_toas=True,
        k_drop=parameter.Uniform(0.0, 1.0), k_threshold=0.5, name=''):
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
                        np.interp(avetoas, psr.toas, psr.planetssb[:, jj, aa])
                        for aa in range(3)]).T
                self._wf[''].add_kwarg(planetssb=planetssb)

                # Inteprolating the pulsar position vectors onto epoch TOAs
                pos_t = np.array([np.interp(avetoas, psr.toas, psr.pos_t[:, aa])
                                  for aa in range(3)]).T
                self._wf[''].add_kwarg(pos_t=pos_t)

            # initialize delay
            self._delay = np.zeros(len(psr.toas))

        @signal_base.cache_call('delay_params')
        def get_delay(self, params):
            delay = self._wf[''](params=params)
            if use_epoch_toas:
                for slc, val in zip(self.uinds, delay):
                    self._delay[slc] = val
                return self._delay
            else:
                return delay

    return Dropout_PhysicalEphemerisSignal
