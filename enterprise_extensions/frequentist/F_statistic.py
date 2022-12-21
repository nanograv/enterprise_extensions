# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
from enterprise.signals import deterministic_signals, gp_signals, signal_base

from enterprise_extensions import blocks, deterministic


def get_xCy(Nvec, T, sigmainv, x, y):
    """Get x^T C^{-1} y"""
    TNx = Nvec.solve(x, left_array=T)
    TNy = Nvec.solve(y, left_array=T)
    xNy = Nvec.solve(y, left_array=x)
    return xNy - TNx @ sigmainv @ TNy


def get_TCy(Nvec, T, y, sigmainv, TNT):
    """Get T^T C^{-1} y"""
    TNy = Nvec.solve(y, left_array=T)
    return TNy - TNT @ sigmainv @ TNy


def innerprod(Nvec, T, sigmainv, TNT, x, y):
    """Get the inner product between x and y"""
    xCy = get_xCy(Nvec, T, sigmainv, x, y)
    TCy = get_TCy(Nvec, T, y, sigmainv, TNT)
    TCx = get_TCy(Nvec, T, x, sigmainv, TNT)
    return xCy - TCx.T @ sigmainv @ TCy


class FpStat(object):
    """
    Class for the Fp-statistic.

    :param psrs: List of `enterprise` Pulsar instances.
    :param noisedict: Dictionary of white noise parameter values. Default=None
    :param psrTerm: Include the pulsar term in the CW signal model. Default=True
    :param bayesephem: Include BayesEphem model. Default=True
    """

    def __init__(self, psrs, noisedict=None,
                 psrTerm=True, bayesephem=True, pta=None, tnequad=False):

        if pta is None:

            # initialize standard model with fixed white noise
            # and powerlaw red noise
            # uses the implementation of ECORR in gp_signals
            print('Initializing the model...')

            tmin = np.min([p.toas.min() for p in psrs])
            tmax = np.max([p.toas.max() for p in psrs])
            Tspan = tmax - tmin
            s = gp_signals.TimingModel(use_svd=True)
            s += deterministic.cw_block_circ(amp_prior='log-uniform',
                                             psrTerm=psrTerm, tref=tmin, name='cw')
            s += blocks.red_noise_block(prior='log-uniform', psd='powerlaw',
                                        Tspan=Tspan, components=30)

            if bayesephem:
                s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

            # adding white-noise, and acting on psr objects
            models = []
            for p in psrs:
                if 'NANOGrav' in p.flags['pta']:
                    s2 = s + blocks.white_noise_block(vary=False, inc_ecorr=True,
                                                      gp_ecorr=True, tnequad=tnequad)
                    models.append(s2(p))
                else:
                    s3 = s + blocks.white_noise_block(vary=False, inc_ecorr=False, tnequad=tnequad)
                    models.append(s3(p))

            pta = signal_base.PTA(models)

            # set white noise parameters
            if noisedict is None:
                print('No noise dictionary provided!')
            else:
                pta.set_default_params(noisedict)

            self.pta = pta

        else:
            # user can specify their own pta object
            # if ECORR is included, use the implementation in gp_signals
            self.pta = pta

        self.psrs = psrs
        self.noisedict = noisedict

        # precompute important bits:
        self.phiinvs = self.pta.get_phiinv(noisedict)
        self.TNTs = self.pta.get_TNT(noisedict)
        self.Nvecs = self.pta.get_ndiag(noisedict)
        self.Ts = self.pta.get_basis(noisedict)
        # self.cf_TNT = [sl.cho_factor(TNT + np.diag(phiinv)) for TNT, phiinv in zip(self.TNTs, self.phiinvs)]
        self.sigmainvs = [np.linalg.pinv(TNT + np.diag(phiinv)) for TNT, phiinv in zip(self.TNTs, self.phiinvs)]

    def compute_Fp(self, fgw):
        """
        Computes the Fp-statistic.

        :param fgw: GW frequency

        :returns:
            fstat: value of the Fp-statistic at the given frequency

        """
        N = np.zeros(2)
        M = np.zeros((2, 2))
        fstat = 0
        for psr, Nvec, TNT, T, sigmainv in zip(self.psrs, self.Nvecs, self.TNTs, self.Ts, self.sigmainvs):

            ntoa = len(psr.toas)

            A = np.zeros((2, ntoa))
            A[0, :] = 1 / fgw ** (1 / 3) * np.sin(2 * np.pi * fgw * psr.toas)
            A[1, :] = 1 / fgw ** (1 / 3) * np.cos(2 * np.pi * fgw * psr.toas)

            ip1 = innerprod(Nvec, T, sigmainv, TNT, A[0, :], psr.residuals)
            # logger.info(ip1)
            ip2 = innerprod(Nvec, T, sigmainv, TNT, A[1, :], psr.residuals)
            # logger.info(ip2)
            N = np.array([ip1, ip2])

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    M[jj, kk] = innerprod(Nvec, T, sigmainv, TNT, A[jj, :], A[kk, :])

            # take inverse of M
            Minv = np.linalg.pinv(M)
            fstat += 0.5 * np.dot(N, np.dot(Minv, N))

        return fstat

    def compute_fap(self, fgw):
        """
        Compute false alarm rate for Fp-Statistic. We calculate
        the log of the FAP and then exponentiate it in order
        to avoid numerical precision problems

        :param fgw: GW frequency

        :returns: False alarm probability as defined in Eq (64)
                  of Ellis, Seiemens, Creighton (2012)

        """

        fp0 = self.compute_Fp(fgw)

        N = len(self.psrs)
        n = np.arange(0, N)

        return np.sum(np.exp(n*np.log(fp0)-fp0-np.log(scipy.special.gamma(n+1))))
