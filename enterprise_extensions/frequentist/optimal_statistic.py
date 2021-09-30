# -*- coding: utf-8 -*-

import warnings

import numpy as np
import scipy.linalg as sl
from enterprise.signals import gp_priors, signal_base, utils

from enterprise_extensions import model_orfs, models


# Define the output to be on a single line.
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


# Override default format.
warnings.formatwarning = warning_on_one_line


class OptimalStatistic(object):
    """
    Class for the Optimal Statistic as used in the analysis paper.

    This class can be used for both standard ML or noise-marginalized OS.

    :param psrs: List of `enterprise` Pulsar instances.
    :param bayesephem: Include BayesEphem model. Default=True
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].

    """

    def __init__(self, psrs, bayesephem=True, gamma_common=4.33, orf='hd',
                 wideband=False, select=None, noisedict=None, pta=None):

        # initialize standard model with fixed white noise and
        # and powerlaw red and gw signal

        if pta is None:
            self.pta = models.model_2a(psrs, psd='powerlaw',
                                       bayesephem=bayesephem,
                                       gamma_common=gamma_common,
                                       is_wideband=wideband,
                                       select='backend', noisedict=noisedict)
        else:
            self.pta = pta

        self.gamma_common = gamma_common
        # get frequencies here
        self.freqs = self._get_freqs(psrs)

        # get F-matrices and set up cache
        self.Fmats = self.get_Fmats()
        self._set_cache_parameters()

        # pulsar locations
        self.psrlocs = [p.pos for p in psrs]

        # overlap reduction function
        if orf == 'hd':
            self.orf = model_orfs.hd_orf
        elif orf == 'dipole':
            self.orf = model_orfs.dipole_orf
        elif orf == 'monopole':
            self.orf = model_orfs.monopole_orf
        elif orf == 'gw_monopole':
            self.orf = model_orfs.gw_monopole_orf
        elif orf == 'gw_dipole':
            self.orf = model_orfs.gw_dipole_orf
        elif orf == 'st':
            self.orf = model_orfs.st_orf
        else:
            raise ValueError('Unknown ORF!')

    def compute_os(self, params=None, psd='powerlaw', fgw=None):
        """
        Computes the optimal statistic values given an
        `enterprise` parameter dictionary.

        :param params: `enterprise` parameter dictionary.
        :param psd: choice of cross-power psd [powerlaw,spectrum]
        :fgw: frequency of GW spectrum to probe, in Hz [default=None]

        :returns:
            xi: angular separation [rad] for each pulsar pair
            rho: correlation coefficient for each pulsar pair
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair.
            OS: Optimal statistic value (units of A_gw^2)
            OS_sig: 1-sigma uncertainty on OS

        .. note:: SNR is computed as OS / OS_sig. In the case of a 'spectrum' model
        the OS variable will be the PSD(fgw) * Tspan value at the relevant fgw bin.

        """

        if params is None:
            params = {name: par.sample() for name, par
                      in zip(self.pta.param_names, self.pta.params)}
        else:
            # check to see that the params dictionary includes values
            # for all of the parameters in the model
            for p in self.pta.param_names:
                if p not in params.keys():
                    msg = '{0} is not included '.format(p)
                    msg += 'in the parameter dictionary. '
                    msg += 'Drawing a random value.'

                    warnings.warn(msg)

        # get matrix products
        TNrs = self.get_TNr(params=params)
        TNTs = self.get_TNT(params=params)
        FNrs = self.get_FNr(params=params)
        FNFs = self.get_FNF(params=params)
        FNTs = self.get_FNT(params=params)

        phiinvs = self.pta.get_phiinv(params, logdet=False)

        X, Z = [], []
        for TNr, TNT, FNr, FNF, FNT, phiinv in zip(TNrs, TNTs, FNrs, FNFs, FNTs, phiinvs):

            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            try:
                cf = sl.cho_factor(Sigma)
                SigmaTNr = sl.cho_solve(cf, TNr)
                SigmaTNF = sl.cho_solve(cf, FNT.T)
            except np.linalg.LinAlgError:
                SigmaTNr = np.linalg.solve(Sigma, TNr)
                SigmaTNF = np.linalg.solve(Sigma, FNT.T)

            FNTSigmaTNr = np.dot(FNT, SigmaTNr)
            X.append(FNr - FNTSigmaTNr)
            Z.append(FNF - np.dot(FNT, SigmaTNF))

        npsr = len(self.pta._signalcollections)
        rho, sig, ORF, xi = [], [], [], []
        for ii in range(npsr):
            for jj in range(ii+1, npsr):

                if psd == 'powerlaw':
                    if self.gamma_common is None and 'gw_gamma' in params.keys():
                        print('{0:1.2}'.format(params['gw_gamma']))
                        phiIJ = utils.powerlaw(self.freqs, log10_A=0,
                                               gamma=params['gw_gamma'])
                    else:
                        phiIJ = utils.powerlaw(self.freqs, log10_A=0,
                                               gamma=self.gamma_common)
                elif psd == 'spectrum':
                    Sf = -np.inf * np.ones(int(len(self.freqs)/2))
                    idx = (np.abs(np.unique(self.freqs) - fgw)).argmin()
                    Sf[idx] = 0.0
                    phiIJ = gp_priors.free_spectrum(self.freqs,
                                                    log10_rho=Sf)

                top = np.dot(X[ii], phiIJ * X[jj])
                bot = np.trace(np.dot(Z[ii]*phiIJ[None, :], Z[jj]*phiIJ[None, :]))

                # cross correlation and uncertainty
                rho.append(top / bot)
                sig.append(1 / np.sqrt(bot))

                # Overlap reduction function for PSRs ii, jj
                ORF.append(self.orf(self.psrlocs[ii], self.psrlocs[jj]))

                # angular separation
                xi.append(np.arccos(np.dot(self.psrlocs[ii], self.psrlocs[jj])))

        rho = np.array(rho)
        sig = np.array(sig)
        ORF = np.array(ORF)
        xi = np.array(xi)
        OS = (np.sum(rho*ORF / sig ** 2) / np.sum(ORF ** 2 / sig ** 2))
        OS_sig = 1 / np.sqrt(np.sum(ORF ** 2 / sig ** 2))

        return xi, rho, sig, OS, OS_sig

    def compute_noise_marginalized_os(self, chain, param_names=None, N=10000):
        """
        Compute noise marginalized OS.

        :param chain: MCMC chain from Bayesian run.
        :param param_names: list of parameter names for the chain file
        :param N: number of iterations to run.

        :returns: (os, snr) array of OS and SNR values for each iteration.

        """

        # check that the chain file has the same number of parameters as the model
        if chain.shape[1] - 4 != len(self.pta.param_names):
            msg = 'MCMC chain does not have the same number of parameters '
            msg += 'as the model.'

            warnings.warn(msg)

        opt, sig = np.zeros(N), np.zeros(N)
        rho, rho_sig = [], []
        setpars = {}
        for ii in range(N):
            idx = np.random.randint(0, chain.shape[0])

            # if param_names is not specified, the parameter dictionary
            # is made by mapping the values from the chain to the
            # parameters in the pta object
            if param_names is None:
                setpars.update(self.pta.map_params(chain[idx, :-4]))
            else:
                setpars = dict(zip(param_names, chain[idx, :-4]))
            xi, rho_tmp, rho_sig_tmp, opt[ii], sig[ii] = self.compute_os(params=setpars)
            rho.append(rho_tmp)
            rho_sig.append(rho_sig_tmp)

        return (np.array(xi), np.array(rho), np.array(rho_sig), opt, opt/sig)

    def compute_noise_maximized_os(self, chain, param_names=None):
        """
        Compute noise maximized OS.

        :param chain: MCMC chain from Bayesian run.

        :returns:
            xi: angular separation [rad] for each pulsar pair
            rho: correlation coefficient for each pulsar pair
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair.
            OS: Optimal statistic value (units of A_gw^2)
            SNR: OS / OS_sig

        """

        # check that the chain file has the same number of parameters as the model
        if chain.shape[1] - 4 != len(self.pta.param_names):
            msg = 'MCMC chain does not have the same number of parameters '
            msg += 'as the model.'

            warnings.warn(msg)

        idx = np.argmax(chain[:, -4])

        # if param_names is not specified, the parameter dictionary
        # is made by mapping the values from the chain to the
        # parameters in the pta object
        if param_names is None:
            setpars = (self.pta.map_params(chain[idx, :-4]))
        else:
            setpars = dict(zip(param_names, chain[idx, :-4]))

        xi, rho, sig, Opt, Sig = self.compute_os(params=setpars)

        return (xi, rho, sig, Opt, Opt/Sig)

    def compute_multiple_corr_os(self, params=None, psd='powerlaw', fgw=None,
                                 correlations=['monopole', 'dipole', 'hd']):
        """
        Fits the correlations to multiple spatial correlation functions

        :param params: `enterprise` parameter dictionary.
        :param psd: choice of cross-power psd [powerlaw,spectrum]
        :fgw: frequency of GW spectrum to probe, in Hz [default=None]
        :param correlations: list of correlation functions

        :returns:
            xi: angular separation [rad] for each pulsar pair
            rho: correlation coefficient for each pulsar pair
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair.
            A: An array of correlation amplitudes
            OS_sig: An array of 1-sigma uncertainties on the correlation amplitudes
        """

        xi, rho, sig, _, _ = self.compute_os(params=params, psd='powerlaw', fgw=None)

        # construct a list of all the ORFs to be fit simultaneously
        ORFs = []
        for corr in correlations:
            if corr == 'hd':
                orf_func = model_orfs.hd_orf
            elif corr == 'dipole':
                orf_func = model_orfs.dipole_orf
            elif corr == 'monopole':
                orf_func = model_orfs.monopole_orf
            elif corr == 'gw_monopole':
                orf_func = model_orfs.gw_monopole_orf
            elif corr == 'gw_dipole':
                orf_func = model_orfs.gw_dipole_orf
            elif corr == 'st':
                orf_func = model_orfs.st_orf
            else:
                raise ValueError('Unknown ORF!')

            ORF = []

            npsr = len(self.pta._signalcollections)
            for ii in range(npsr):
                for jj in range(ii+1, npsr):
                    ORF.append(orf_func(self.psrlocs[ii], self.psrlocs[jj]))

            ORFs.append(np.array(ORF))

        Bmat = np.array([[np.sum(ORFs[i]*ORFs[j]/sig**2) for i in range(len(ORFs))]
                         for j in range(len(ORFs))])

        Bmatinv = np.linalg.inv(Bmat)

        Cmat = np.array([np.sum(rho*ORFs[i]/sig**2) for i in range(len(ORFs))])

        A = np.dot(Bmatinv, Cmat)
        A_err = np.array([np.sqrt(Bmatinv[i, i]) for i in range(len(ORFs))])

        return xi, rho, sig, A, A_err

    def compute_noise_marginalized_multiple_corr_os(self, chain, param_names=None, N=10000,
                                                    correlations=['monopole', 'dipole', 'hd']):
        """
        Noise-marginalized fitting of the correlations to multiple spatial
        correlation functions

        :param correlations: list of correlation functions
        :param chain: MCMC chain from Bayesian run.
        :param param_names: list of parameter names for the chain file
        :param N: number of iterations to run.

        :returns:
            xi: angular separation [rad] for each pulsar pair
            rho: correlation coefficient for each pulsar pair and for each noise realization
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair
                 and for each noise realization
            A: An array of correlation amplitudes for each noise realization
            OS_sig: An array of 1-sigma uncertainties on the correlation amplitudes
                    for each noise realization
        """

        # check that the chain file has the same number of parameters as the model
        if chain.shape[1] - 4 != len(self.pta.param_names):
            msg = 'MCMC chain does not have the same number of parameters '
            msg += 'as the model.'

            warnings.warn(msg)

        rho, sig, A, A_err = [], [], [], []
        setpars = {}
        for ii in range(N):
            idx = np.random.randint(0, chain.shape[0])

            # if param_names is not specified, the parameter dictionary
            # is made by mapping the values from the chain to the
            # parameters in the pta object
            if param_names is None:
                setpars.update(self.pta.map_params(chain[idx, :-4]))
            else:
                setpars = dict(zip(param_names, chain[idx, :-4]))

            xi, rho_tmp, sig_tmp, A_tmp, A_err_tmp = self.compute_multiple_corr_os(params=setpars,
                                                                                   correlations=correlations)

            rho.append(rho_tmp)
            sig.append(sig_tmp)
            A.append(A_tmp)
            A_err.append(A_err_tmp)

        return np.array(xi), np.array(rho), np.array(sig), np.array(A), np.array(A_err)

    def get_Fmats(self, params={}):
        """Kind of a hack to get F-matrices"""
        Fmats = []
        for sc in self.pta._signalcollections:
            ind = []
            for signal, idx in sc._idx.items():
                if signal.signal_name == 'red noise' and signal.signal_id in ['gw', 'gw_crn']:
                    ind.append(idx)
            ix = np.unique(np.concatenate(ind))
            Fmats.append(sc.get_basis(params=params)[:, ix])

        return Fmats

    def _get_freqs(self, psrs):
        """ Hackish way to get frequency vector."""
        for sig in self.pta._signalcollections[0]._signals:
            if sig.signal_name == 'red noise' and sig.signal_id in ['gw', 'gw_crn']:
                sig._construct_basis()
                freqs = np.array(sig._labels[''])
                break
        return freqs

    def _set_cache_parameters(self):
        """ Set cache parameters for efficiency. """
        self.white_params = []
        self.basis_params = []
        self.delay_params = []

        for sc in self.pta._signalcollections:
            self.white_params.extend(sc.white_params)
            self.basis_params.extend(sc.basis_params)
            self.delay_params.extend(sc.delay_params)

    def get_TNr(self, params={}):
        return self.pta.get_TNr(params=params)

    @signal_base.cache_call(['white_params', 'delay_params'])
    def get_FNr(self, params={}):
        FNrs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.Fmats[ct]
            res = sc.get_detres(params=params)
            FNrs.append(N.solve(res, left_array=F))
        return FNrs

    @signal_base.cache_call(['white_params'])
    def get_FNF(self, params={}):
        FNFs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.Fmats[ct]
            FNFs.append(N.solve(F, left_array=F))
        return FNFs

    def get_TNT(self, params={}):
        return self.pta.get_TNT(params=params)

    @signal_base.cache_call(['white_params', 'basis_params'])
    def get_FNT(self, params={}):
        FNTs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.Fmats[ct]
            T = sc.get_basis(params=params)
            FNTs.append(N.solve(T, left_array=F))
        return FNTs
