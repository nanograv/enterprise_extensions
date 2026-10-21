# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl

from enterprise.signals import (gp_signals, parameter, signal_base,
                                white_signals)
from enterprise.signals import selections

from enterprise_extensions import blocks as ee_blocks


class FiStat(object):
    """
    Class for the Fi-statistic.

    This class can be used for both standard ML or noise-marginalized OS.

    :param psrs: List of `enterprise` Pulsar instances.
    :param noisedict: Dictionary of noise parameters.
    :param pta: pre-constructed enterprise PT
    :param include_curn: does the dataset contain a common red noise process (True/False)
    :param gamma_val: default 13./3., CURN slope

    """

    def __init__(self, psrs, noisedict=None, pta=None, include_curn=True, gamma_val=13./3.):

        if pta is None:
            pmaxes = [p.toas.max() for p in psrs]
            pmins = [p.toas.min() for p in psrs]

            full_tspan = max(pmaxes) - min(pmins)

            efac = parameter.Constant()
            equad = parameter.Constant()
            # ecorr = parameter.Constant()

            selection = selections.Selection(selections.by_backend)
            ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
            # ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')
            rn = ee_blocks.red_noise_block(psd='powerlaw', prior='log-uniform', components=30, Tspan=full_tspan, logmin=-20, logmax=-11)
            tm = gp_signals.TimingModel(use_svd=True)

            if include_curn:
                if gamma_val is None:
                    gamma = parameter.Uniform(0, 7)
                else:
                    gamma = gamma_val
                curn = ee_blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=full_tspan, components=15, logmin=-20, logmax=-11, gamma_val=gamma, name='gw')
                s = tm + ef + curn + rn # + ec
            else:
                s = tm + ef + rn # + ec

            print('noise generated')

            model = []
            for p in psrs:
                model.append(s(p))
            self.pta = signal_base.PTA(model)

        else:
            if np.any(['marginalizing_linear_timing' in sig for sig in pta.signals]):
                msg = "Can't run optimal statistic with `enterprise.gp_signals.MarginalizingTimingModel`."
                msg += " Try creating PTA with `enterprise.gp_signals.TimingModel`, or if using `enterprise_extensions`"
                msg += " set `tm_marg=False`."
                raise ValueError(msg)
            self.pta = pta

        # set white noise parameters
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            self.pta.set_default_params(noisedict)

        self.psrs = psrs
        self.params = noisedict
        self.param_names = self.pta.param_names
        self.Nmats = None

    def compute_Fi(self, t0):
        """
        Computes the Fi-statistic for Gravitational Wave Memory at a specific starting epoch (see Jerry p. Sun, Xavier Siemens, Dustin R. Madison, 2024)

        :param t0: epoch of memory event (MJD)

        :returns:
            fstat: value of the Fi-statistic

        """
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        Nvecs = self.pta.get_ndiag(self.params)

        t0_sec = t0*24*3600

        fstat = 0

        for idx, (psr, Nvec, TNT, phiinv, T) in enumerate(zip(self.psrs, Nvecs, TNTs, phiinvs, Ts)):

            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            Nmat = self.make_Nmat(phiinv, TNT, Nvec, T)

            ntoa = len(psr.toas)

            # populate these with the right signal templates
            # Since we multiply by the Fp and Fc antenna patterns later, we just populate these with
            # the the time-dependent shapes
            psr_bwm_template = np.zeros(len(psr.toas))
            for toa_idx, toa in enumerate(psr.toas):
                if toa > t0_sec:
                    psr_bwm_template[toa_idx] = toa - t0_sec
            # A = np.zeros((2, ntoa))
            # A[0, :] = psr_bwm_template
            # A[1, :] = psr_bwm_template

            N = self.innerProduct_rr(psr_bwm_template, psr.residuals, Nmat, T, Sigma, brave=False)

            M = self.innerProduct_rr(psr_bwm_template, psr_bwm_template, Nmat, T, Sigma, brave=False)

            fstat += 0.5 * N*N/M

        return fstat

    def compute_Fi_t0_range(self, t0_range):
        """
        Computes the Fi-statistic for Gravitational Wave Memory over a range of starting epochs(see Jerry p. Sun, Xavier Siemens, Dustin R. Madison, 2024)

        :param t0_range: list of memory event epochs (MJD)

        :returns:
            fstat: list of Fi-statistic values

        """
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        Nvecs = self.pta.get_ndiag(self.params)

        t0_range_sec = t0_range*24*3600

        fstat_list = np.zeros((len(t0_range)))

        for idx, (psr, Nvec, TNT, phiinv, T) in enumerate(zip(self.psrs, Nvecs, TNTs, phiinvs, Ts)):
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            Nmat = self.make_Nmat(phiinv, TNT, Nvec, T)
            ntoa = len(psr.toas)
            for index, t0_sec in enumerate(t0_range_sec):
                # populate these with the right signal templates
                # Since we multiply by the Fp and Fc antenna patterns later, we just populate these with
                # the the time-dependent shapes
                psr_bwm_template = np.zeros(len(psr.toas))
                for toa_idx, toa in enumerate(psr.toas):
                    if toa > t0_sec:
                        psr_bwm_template[toa_idx] = toa - t0_sec
                # A = np.zeros((2, ntoa))
                # A[0, :] = psr_bwm_template
                # A[1, :] = psr_bwm_template

                N = self.innerProduct_rr(psr_bwm_template, psr.residuals, Nmat, T, Sigma, brave=False)

                M = self.innerProduct_rr(psr_bwm_template, psr_bwm_template, Nmat, T, Sigma, brave=False)

                fstat_list[index] += 0.5 * N*N/M

        return fstat_list

    def compute_maximum_likelihood_Fi(self, epoch, noisedict, chain):
        """
        Computes the Fi-statistic for the noise parameters pulled from the maximum likelihood value in a Bayesian chain posterior

        :param epoch: memory event epoch (MJD) to calculate for
        :param noisedict: Dictionary of noise parameters
        :param cahin: parameter list from a Bayesian chain

        :returns:
            fstat: value for the Fi computed with the specified noise parameters

        """
        print('FeStat made')
        index_of_max = np.argmax(chain[:, -3])  # gets the index of the max likelihood value
        setpars = dict(zip(self.param_names, chain[index_of_max, :-4]))
        self.params = setpars

        fstat = self.compute_Fi(epoch)

        return fstat

    def compute_noise_marginalized_Fi(self, epochs, noisedict, chain, Nitr=1000):
        """
        Computes the Fi-statistic for the noise parameters pulled from N random points in a Bayesian chain posterior

        :param epoch: memory event epoch (MJD) to calculate for
        :param noisedict: Dictionary of noise parameters
        :param chain: parameter list from a Bayesian chain
        :param N: number of samples to pull from chain

        :returns:
            fstat: list of values for the Fi computed with the specified noise parameters

        """
        fi_all = []

        print('FeStat made')
        for ii in range(Nitr):
            idx = np.random.randint(0, chain.shape[0])
            setpars = dict(zip(self.param_names, chain[idx, :-4]))
            self.params = setpars

            fi_all.append(self.compute_Fi_t0_range(epochs))
            if ii % 10 == 0:
                print('{} done'.format(ii/Nitr))

        return fi_all

    def innerProduct_rr(self, x, y, Nmat, Tmat, Sigma, TNx=None, TNy=None, brave=False):
        """
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T Sigma^{-1} T^T N^{-1} y

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

        if brave:
            cf = sl.cho_factor(Sigma, check_finite=False)
            SigmaTNy = sl.cho_solve(cf, TNy, check_finite=False)
        else:
            cf = sl.cho_factor(Sigma)
            SigmaTNy = sl.cho_solve(cf, TNy)

        ret = xNy - np.dot(TNx, SigmaTNy)

        return ret

    def innerProduct_rr_NoInv(x, y, Nvec, Tmat, Sigma, TNx=None, TNy=None, brave=False): #chol_Sigma
        """
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

        :param x: vector timeseries 1
        :param y: vector timeseries 2
        :param Nvec: white noise matrix
        :param Tmat: Modified design matrix including red noise/jitter
        :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
        :param TNx: T^T N^{-1} x precomputed
        :param TNy: T^T N^{-1} y precomputed

        :return: inner product (x|y)
        """

        # white noise term
        xNy = Nvec.solve(y, left_array=x)
        Nx, Ny = np.dot(Ni, x), np.dot(Ni, y)

        if TNx is None and TNy is None:
            TNx = Nvec.solve(x, left_array=Tmat)
            TNy = Nvec.solve(y, left_array=Tmat)

        if brave:
            cf = sl.cho_factor(Sigma, check_finite=False)
            SigmaTNy = sl.cho_solve(cf, TNy, check_finite=False)
        else:
            cf = sl.cho_factor(Sigma)
            SigmaTNy = sl.cho_solve(cf, TNy)

        ret = xNy - np.dot(TNx, SigmaTNy)

        return ret

    def make_Nmat(self, phiinv, TNT, Nvec, T):
        """
        Generate the N noise matrix for a specific pulsar using a Cholesky factorization of the matrix

        :param phiinv: inverse of the phi matrix
        :param TNT: white noise inner product of T matrix
        :param Nvec: pulsar auto errors
        :param T: design matrix and fourier basis


        :return: Nmatrix

        """
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        cf = sl.cho_factor(Sigma)

        TtN = np.multiply((1/Nvec)[:, None], T).T

        # Put pulsar's autoerrors in a diagonal matrix
        Ndiag = np.diag(1/Nvec)

        expval2 = sl.cho_solve(cf, TtN)

        # An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
        return Ndiag - np.dot(TtN.T, expval2)
