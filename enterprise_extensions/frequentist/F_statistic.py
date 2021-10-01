# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl
import scipy.special
from enterprise.signals import deterministic_signals, gp_signals, signal_base

from enterprise_extensions import blocks, deterministic


class FpStat(object):
    """
    Class for the Fp-statistic.

    :param psrs: List of `enterprise` Pulsar instances.
    :param noisedict: Dictionary of white noise parameter values. Default=None
    :param psrTerm: Include the pulsar term in the CW signal model. Default=True
    :param bayesephem: Include BayesEphem model. Default=True
    """

    def __init__(self, psrs, params=None,
                 psrTerm=True, bayesephem=True, pta=None):

        if pta is None:

            # initialize standard model with fixed white noise
            # and powerlaw red noise
            # uses the implementation of ECORR in gp_signals
            print('Initializing the model...')

            tmin = np.min([p.toas.min() for p in psrs])
            tmax = np.max([p.toas.max() for p in psrs])
            Tspan = tmax - tmin

            s = deterministic.cw_block_circ(amp_prior='log-uniform',
                                            psrTerm=psrTerm, tref=tmin, name='cw')
            s += gp_signals.TimingModel()
            s += blocks.red_noise_block(prior='log-uniform', psd='powerlaw',
                                        Tspan=Tspan, components=30)

            if bayesephem:
                s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

            # adding white-noise, and acting on psr objects
            models = []
            for p in psrs:
                if 'NANOGrav' in p.flags['pta']:
                    s2 = s + blocks.white_noise_block(vary=False, inc_ecorr=True,
                                                      gp_ecorr=True)
                    models.append(s2(p))
                else:
                    s3 = s + blocks.white_noise_block(vary=False, inc_ecorr=False)
                    models.append(s3(p))

            pta = signal_base.PTA(models)

            # set white noise parameters
            if params is None:
                print('No noise dictionary provided!')
            else:
                pta.set_default_params(params)

            self.pta = pta

        else:

            # user can specify their own pta object
            # if ECORR is included, use the implementation in gp_signals
            self.pta = pta

        self.psrs = psrs
        self.params = params

        self.Nmats = self.get_Nmats()

    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.pta.get_TNT(self.params)
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        # Get noise parameters for pta toaerr**2
        Nvecs = self.pta.get_ndiag(self.params)
        # Get the basis matrix
        Ts = self.pta.get_basis(self.params)

        Nmats = [make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

        return Nmats

    def compute_Fp(self, fgw):
        """
        Computes the Fp-statistic.

        :param fgw: GW frequency

        :returns:
        fstat: value of the Fp-statistic at the given frequency
        """

        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()

        N = np.zeros(2)
        M = np.zeros((2, 2))
        fstat = 0

        for psr, Nmat, TNT, phiinv, T in zip(self.psrs, self.Nmats,
                                             TNTs, phiinvs, Ts):

            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            ntoa = len(psr.toas)

            A = np.zeros((2, ntoa))
            A[0, :] = 1 / fgw ** (1 / 3) * np.sin(2 * np.pi * fgw * psr.toas)
            A[1, :] = 1 / fgw ** (1 / 3) * np.cos(2 * np.pi * fgw * psr.toas)

            ip1 = innerProduct_rr(A[0, :], psr.residuals, Nmat, T, Sigma)
            ip2 = innerProduct_rr(A[1, :], psr.residuals, Nmat, T, Sigma)
            N = np.array([ip1, ip2])

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    M[jj, kk] = innerProduct_rr(A[jj, :], A[kk, :], Nmat, T, Sigma)

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


def innerProduct_rr(x, y, Nmat, Tmat, Sigma, TNx=None, TNy=None):
    r"""
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

        :param x: vector timeseries 1
        :param y: vector timeseries 2
        :param Nmat: white noise matrix
        :param Tmat: Modified design matrix including red noise/jitter
        :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
        :param TNx: T^T N^{-1} x precomputed
        :param TNy: T^T N^{-1} y precomputed
        :return: inner product (x|y)
        """

    # white noise term
    Ni = Nmat
    xNy = np.dot(np.dot(x, Ni), y)
    Nx, Ny = np.dot(Ni, x), np.dot(Ni, y)

    if TNx is None and TNy is None:
        TNx = np.dot(Tmat.T, Nx)
        TNy = np.dot(Tmat.T, Ny)

    cf = sl.cho_factor(Sigma)
    SigmaTNy = sl.cho_solve(cf, TNy)

    ret = xNy - np.dot(TNx, SigmaTNy)

    return ret


def make_Nmat(phiinv, TNT, Nvec, T):

    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    # Nshape = np.shape(T)[0] # Not currently used in code

    TtN = np.multiply((1/Nvec)[:, None], T).T

    # Put pulsar's autoerrors in a diagonal matrix
    Ndiag = np.diag(1/Nvec)

    expval2 = sl.cho_solve(cf, TtN)
    # TtNt = np.transpose(TtN) # Not currently used in code

    # An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T, expval2)
