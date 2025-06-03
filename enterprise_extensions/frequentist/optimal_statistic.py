# -*- coding: utf-8 -*-

import warnings

import numpy as np
import scipy.linalg as sl
import scipy.integrate as sint
from enterprise.signals import gp_priors, signal_base, utils

from enterprise_extensions import model_orfs, models


# Define the output to be on a single line.
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


# Override default format.
warnings.formatwarning = warning_on_one_line


def imhof(u, x, eigen_values, output='cdf'):
    theta = 0.5 * np.sum(np.arctan(eigen_values[:,np.newaxis] * u), axis=0) - 0.5 * x * u
    rho = np.prod((1.0 + (eigen_values[:,np.newaxis] * u)**2)**0.25, axis=0)

    rv = np.sin(theta) / (u * rho) if output=='cdf' else np.cos(theta) / rho

    return rv


def gx2pdf(eigen_values, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 PDF as a function of sx, based off of eigenvalues 'eigen_values'"""

    eigen_values = eigen_values[:cutoff] if cutoff > 1 else eigen_values[np.abs(eigen_values) > cutoff]

    return np.array([sint.quad(lambda u: float(imhof(u, x, eigen_values, output='pdf')),
                                                0, np.inf, limit=limit, epsabs=epsabs)[0] / (2*np.pi) for x in xs])


def gx2cdf(eigr, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 CDF as a function of sx, based off of eigenvalues 'eigr'"""

    eigen_values = eigr[:cutoff] if cutoff > 1 else eigr[np.abs(eigr) > cutoff]

    return np.array([0.5 - sint.quad(lambda u: float(imhof(u, x, eigen_values)),
                                                0, np.inf, limit=limit, epsabs=epsabs)[0] / np.pi for x in xs])


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

    def __init__(
        self,
        psrs,
        bayesephem=True,
        gamma_common=4.33,
        orf="hd",
        wideband=False,
        select=None,
        noisedict=None,
        pta=None,
    ):

        # initialize standard model with fixed white noise and
        # and powerlaw red and gw signal

        if pta is None:
            self.pta = models.model_2a(
                psrs,
                psd="powerlaw",
                bayesephem=bayesephem,
                gamma_common=gamma_common,
                is_wideband=wideband,
                select="backend",
                noisedict=noisedict,
            )
        else:
            if np.any(["marginalizing_linear_timing" in sig for sig in pta.signals]):
                msg = "Can't run optimal statistic with `enterprise.gp_signals.MarginalizingTimingModel`."
                msg += " Try creating PTA with `enterprise.gp_signals.TimingModel`, or if using `enterprise_extensions`"
                msg += " set `tm_marg=False`."
                raise ValueError(msg)
            self.pta = pta

        self.gamma_common = gamma_common
        # get frequencies here
        self.freqs = self._get_freqs(psrs)

        # set up cache
        self._set_cache_parameters()

        # pulsar locations
        self.psrlocs = [p.pos for p in psrs]

        # overlap reduction function
        if orf == "hd":
            self.orf = model_orfs.hd_orf
        elif orf == "dipole":
            self.orf = model_orfs.dipole_orf
        elif orf == "monopole":
            self.orf = model_orfs.monopole_orf
        elif orf == "gw_monopole":
            self.orf = model_orfs.gw_monopole_orf
        elif orf == "gw_dipole":
            self.orf = model_orfs.gw_dipole_orf
        elif orf == "st":
            self.orf = model_orfs.st_orf
        else:
            raise ValueError("Unknown ORF!")

    def compute_os(self, params=None, psd="powerlaw", fgw=None):
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
            params = {
                name: par.sample()
                for name, par in zip(self.pta.param_names, self.pta.params)
            }
        else:
            # check to see that the params dictionary includes values
            # for all of the parameters in the model
            for p in self.pta.param_names:
                if p not in params.keys():
                    msg = "{0} is not included ".format(p)
                    msg += "in the parameter dictionary. "
                    msg += "Drawing a random value."

                    warnings.warn(msg)

        # get matrix products
        TNrs = self.get_TNr(params=params)
        TNTs = self.get_TNT(params=params)
        FNrs = self.get_FNr(params=params)
        FNFs = self.get_FNF(params=params)
        FNTs = self.get_FNT(params=params)

        phiinvs = self.pta.get_phiinv(params, logdet=False)

        X, Z = [], []
        for TNr, TNT, FNr, FNF, FNT, phiinv in zip(
            TNrs, TNTs, FNrs, FNFs, FNTs, phiinvs
        ):

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
            for jj in range(ii + 1, npsr):

                if psd == "powerlaw":
                    if self.gamma_common is None and "gw_gamma" in params.keys():
                        phiIJ = utils.powerlaw(
                            self.freqs, log10_A=0, gamma=params["gw_gamma"]
                        )
                    else:
                        phiIJ = utils.powerlaw(
                            self.freqs, log10_A=0, gamma=self.gamma_common
                        )
                elif psd == "spectrum":
                    Sf = -np.inf * np.ones(int(len(self.freqs) / 2))
                    idx = (np.abs(np.unique(self.freqs) - fgw)).argmin()
                    Sf[idx] = 0.0
                    phiIJ = gp_priors.free_spectrum(self.freqs, log10_rho=Sf)

                top = np.dot(X[ii], phiIJ * X[jj])
                bot = np.trace(np.dot(Z[ii] * phiIJ[None, :], Z[jj] * phiIJ[None, :]))

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
        OS = np.sum(rho * ORF / sig**2) / np.sum(ORF**2 / sig**2)
        OS_sig = 1 / np.sqrt(np.sum(ORF**2 / sig**2))

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
            msg = "MCMC chain does not have the same number of parameters "
            msg += "as the model."

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

        return (np.array(xi), np.array(rho), np.array(rho_sig), opt, opt / sig)

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
            msg = "MCMC chain does not have the same number of parameters "
            msg += "as the model."

            warnings.warn(msg)

        idx = np.argmax(chain[:, -4])

        # if param_names is not specified, the parameter dictionary
        # is made by mapping the values from the chain to the
        # parameters in the pta object
        if param_names is None:
            setpars = self.pta.map_params(chain[idx, :-4])
        else:
            setpars = dict(zip(param_names, chain[idx, :-4]))

        xi, rho, sig, Opt, Sig = self.compute_os(params=setpars)

        return (xi, rho, sig, Opt, Opt / Sig)

    def compute_multiple_corr_os(
        self,
        params=None,
        psd="powerlaw",
        fgw=None,
        correlations=["monopole", "dipole", "hd"],
    ):
        """
        Fits the correlations to multiple spatial correlation functions

        :param params: `enterprise` parameter dictionary.
        :param psd: choice of cross-power psd [powerlaw,spectrum]
        :param fgw: frequency of GW spectrum to probe, in Hz [default=None]
        :param correlations: list of correlation functions

        :returns:
            xi: angular separation [rad] for each pulsar pair
            rho: correlation coefficient for each pulsar pair
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair.
            A: An array of correlation amplitudes
            OS_sig: An array of 1-sigma uncertainties on the correlation amplitudes

        """

        xi, rho, sig, _, _ = self.compute_os(params=params, psd="powerlaw", fgw=None)

        # construct a list of all the ORFs to be fit simultaneously
        ORFs = []
        for corr in correlations:
            if corr == "hd":
                orf_func = model_orfs.hd_orf
            elif corr == "dipole":
                orf_func = model_orfs.dipole_orf
            elif corr == "monopole":
                orf_func = model_orfs.monopole_orf
            elif corr == "gw_monopole":
                orf_func = model_orfs.gw_monopole_orf
            elif corr == "gw_dipole":
                orf_func = model_orfs.gw_dipole_orf
            elif corr == "st":
                orf_func = model_orfs.st_orf
            else:
                raise ValueError("Unknown ORF!")

            ORF = []

            npsr = len(self.pta._signalcollections)
            for ii in range(npsr):
                for jj in range(ii + 1, npsr):
                    ORF.append(orf_func(self.psrlocs[ii], self.psrlocs[jj]))

            ORFs.append(np.array(ORF))

        Bmat = np.array(
            [
                [np.sum(ORFs[i] * ORFs[j] / sig**2) for i in range(len(ORFs))]
                for j in range(len(ORFs))
            ]
        )

        Bmatinv = np.linalg.inv(Bmat)

        Cmat = np.array([np.sum(rho * ORFs[i] / sig**2) for i in range(len(ORFs))])

        A = np.dot(Bmatinv, Cmat)
        A_err = np.array([np.sqrt(Bmatinv[i, i]) for i in range(len(ORFs))])

        return xi, rho, sig, A, A_err

    def compute_noise_marginalized_multiple_corr_os(
        self,
        chain,
        param_names=None,
        N=10000,
        correlations=["monopole", "dipole", "hd"],
    ):
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
            sig: 1-sigma uncertainty on correlation coefficient for each pulsar pair and for each noise realization
            A: An array of correlation amplitudes for each noise realization
            OS_sig: An array of 1-sigma uncertainties on the correlation amplitudes for each noise realization

        """

        # check that the chain file has the same number of parameters as the model
        if chain.shape[1] - 4 != len(self.pta.param_names):
            msg = "MCMC chain does not have the same number of parameters "
            msg += "as the model."

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

            xi, rho_tmp, sig_tmp, A_tmp, A_err_tmp = self.compute_multiple_corr_os(
                params=setpars, correlations=correlations
            )

            rho.append(rho_tmp)
            sig.append(sig_tmp)
            A.append(A_tmp)
            A_err.append(A_err_tmp)

        return np.array(xi), np.array(rho), np.array(sig), np.array(A), np.array(A_err)

    @signal_base.cache_call(["basis_params"])
    def get_Fmats(self, params={}):
        """Kind of a hack to get F-matrices"""
        Fmats = []
        for sc in self.pta._signalcollections:
            ind = []
            for signal, idx in sc._idx.items():
                if "gw" in signal.signal_id:
                    ind.append(idx)
            ix = np.unique(np.concatenate(ind))
            Fmats.append(sc.get_basis(params=params)[:, ix])

        return Fmats

    def _get_freqs(self, psrs):
        """Hackish way to get frequency vector."""

        for sig in self.pta._signalcollections[0]._signals:
            if "gw" in sig.signal_id:
                # make sure the basis is created
                _ = sig.get_basis()

                if isinstance(sig._labels, np.ndarray):
                    return sig._labels
                else:
                    return sig._labels[""]

        raise ValueError("No frequency basis in pulsar models")

    def _set_cache_parameters(self):
        """Set cache parameters for efficiency."""

        self.white_params = list(
            set(par for sc in self.pta._signalcollections for par in sc.white_params)
        )
        self.basis_params = list(
            set(par for sc in self.pta._signalcollections for par in sc.basis_params)
        )
        self.delay_params = list(
            set(par for sc in self.pta._signalcollections for par in sc.delay_params)
        )

    def get_TNr(self, params={}):
        return self.pta.get_TNr(params=params)

    @signal_base.cache_call(["white_params", "delay_params", "basis_params"])
    def get_FNr(self, params={}):
        FNrs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.get_Fmats(params)[ct]
            res = sc.get_detres(params=params)
            FNrs.append(N.solve(res, left_array=F))
        return FNrs

    @signal_base.cache_call(["white_params", "basis_params"])
    def get_FNF(self, params={}):
        FNFs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.get_Fmats(params)[ct]
            FNFs.append(N.solve(F, left_array=F))
        return FNFs

    def get_TNT(self, params={}):
        return self.pta.get_TNT(params=params)

    @signal_base.cache_call(["white_params", "basis_params"])
    def get_FNT(self, params={}):
        FNTs = []
        for ct, sc in enumerate(self.pta._signalcollections):
            N = sc.get_ndiag(params=params)
            F = self.get_Fmats(params)[ct]
            T = sc.get_basis(params=params)
            FNTs.append(N.solve(T, left_array=F))
        return FNTs


def inv_RPR(phi, r):
    """Invert the RphiRT matrix"""
    I = np.identity(len(r))
    cf = sl.cho_factor(phi)
    phi_inv = sl.cho_solve(cf, I)
    Sigma = r.T @ r + phi_inv
    cf = sl.cho_factor(Sigma)
    return I - r @ sl.cho_solve(cf, r.T)


class DetectionStatistic(object):
    """
    Class for the Detection Statistic as used in the p-value paper.

    This class is specifically made for classical hypothesis testing. It
    requires an enterprise object for H0 and for H1. For now we assume that we
    keep the white noise fixed and that the only difference between the two
    hypotheses is in the way we model the Phi prior matrix.

    :param pta_h0: The enterprise PTA object for the null hypothesis.
    :param pta_h1: The enterprise PTA object for the signal hypothesis.
    """

    def __init__(
        self,
        pta_h0,
        pta_h1,
        npopt=False
    ):
        # set up cache
        self._set_cache_parameters(pta_h0, pta_h1)
        self._npopt = nptop

    def _set_cache_parameters(self, pta_h0, pta_h1):
        """Set the cache parameters according to the Equations in van Haasteren (2025)"""
        # N = L_N L_N^T  ==> L_N^{-T} L_N^{-1} = N^{-1}
        # T^{prime} = L_N^{-1} Tmat         ( Tprime = NT in code )
        # P_T T^{prime} = T^{prime}   ===> P_T = G_T G_T^T    (G_T = NU in code?)
        # R = G_T @ Tprime = G_T @ L_N^{-1} Tmat
        # L_0 = L_B^{-1} in code with   L_B @ L_B^T = C_i = I + R phi R^T
        # A = L_B^{-1} G^T_T T^{prime}
        # P_F = G_F @ G_F^T  (Project only on the basis of the common red noise)
        # A P_F = P_A A P_F = U_A U_A^T A P_F (do with the SVD of (A P_F))
        # Then, the final data and Q are:
        # chi = U_A^T @ L_0 @ G_T @ L_N^{-1} @ Tmat
        # Q = U_A^T @ L_0^T @ G_T^T @ L_N^{-1} @ F @ DeltaPhi @ F^T L_N^{-T} @ G_T @ L_0 @ U_A
        self.pta_h0 = pta_h0
        self.pta_h1 = pta_h1

        # Calculate lists of H0 quantities (11 seconds, only need it once)
        Tmat = pta_h0.get_basis({})           # List of 2D matrices
        Ndiag = pta_h0.get_ndiag({})          # Objects for sqrtsolve
        NT = [nd.sqrtsolve(tm) for (nd, tm) in zip(Ndiag, Tmat)]        # List of 2D matrices
        G_T = [sl.svd(nt, full_matrices=False)[0] for nt in NT]        # List of 2D matrices
        self.R = [gt.T @ nt for (gt, nt) in zip(G_T, NT)]      # List of 2D matrices

        # Transformation 1:
        Nres = [nd.sqrtsolve(r) for (nd, r) in zip(Ndiag, pta_h0.get_residuals())]   # List of 1D arrays (the weighted data)

        # Transformation 2:
        self.GTNr = [gt.T @ nr for (gt, nr) in zip(G_T, Nres)]                            # List of 1D arrays (transformed data)

        # We do this here so we can avoid calculating the mask later
        xs = np.array([par_val for p in pta_h0.params for par_val in np.atleast_1d(p.sample())])          # 1D array of all parameters
        pd = pta_h1.map_params(xs)
        Phi_0 = [np.diag(p) for p in pta_h0.get_phi(pd)]          # Phi matrix of H0 -- 2D arrays
        BigPhiDiff = pta_h1.get_phi(pd) - sl.block_diag(*Phi_0)

        # Get only the non-zero elements of the BigPhiDiff matrix for selections later
        self.par_msk = (np.sum(np.abs(BigPhiDiff), axis=1)>0)                # Mask for BigPhiDiff
        par_inds_offset = np.cumsum([0] + [len(p) for p in Phi_0])
        par_inds_start = par_inds_offset[:-1]
        par_inds_end = par_inds_offset[1:]
        par_inds_slices = [np.arange(p_start, p_end) for (p_start, p_end) in zip(par_inds_start, par_inds_end)]
        self.par_psr_msk = [self.par_msk[slc] for slc in par_inds_slices]         # Mask per psr for Phi_0 and Tmat

    def _get_compressed_coordinates(self, params):
        """Returns OS, chi, and Q for the given parameters"""

        # These quantities have to be re-calculated for new hyperparameters
        Phi_0 = [np.diag(p) for p in self.pta_h0.get_phi(params)]          # Phi matrix of H0 -- 2D arrays

        # This is a BIG matrix, but it's sparse. Not using that right now though
        # It's currently 0.4 secdonds for NG15
        BigPhiDiff = self.pta_h1.get_phi(params) - sl.block_diag(*Phi_0)                   # 2D prior diff array

        # Inverse Noise matrix
        C2i_0 = [inv_RPR(p, r) for (r, p) in zip(self.R, Phi_0)]                  # List of matrix inverses -- 2D arrays

        # Get the Square-Root (we take it from the inv for numerical stability)
        C2i_0_svd = []
        for c in C2i_0:
            try:
                c_svd = sl.svd(c, full_matrices=True)
            except sl.LinAlgError:
                # GESVD is more numerically stable, but slower
                c_svd = sl.svd(c, full_matrices=True, lapack_driver='gesvd')

            C2i_0_svd.append(c_svd)

        # Select only non-singular values
        C2i_sqrt_sing = [np.array([(np.sqrt(sv) if np.abs(sv) > 1e-10 else 0.0) for sv in s[1]]) for s in C2i_0_svd]          # Singular values -- 1D arrays
        L_0 = [svd[0] @ np.diag(s) @ svd[0].T for (svd, s) in zip(C2i_0_svd, C2i_sqrt_sing)]          # L matrix -- 2D arrays

        # Transformation 3:
        # From now also construct the filter transform, because it is of manaeable size
        LGNr = [l @ gnr for (l, gnr) in zip(L_0, self.GTNr)]          # List of 1D arrays (transformed data)
        S3 = [l_bi @ r for (l_bi, r) in zip(L_0, self.R)]             # List of 2D arrays (Q transformer)

        # Slice BigPhiDiff, because we only want non-zero items! 
        PhiDiff = BigPhiDiff[self.par_msk, :][:, self.par_msk]

        # A = L_B^{-1} G^T_T T^{prime} = L_B^{-1} @ R
        # T^{prime} = L_N^{-1} Tmat
        # P_T T^{prime} = T^{prime}   ===> P_T = G_T G_T^T
        # S3m = S3 @ G_F (is same thing as selecting the columns of S3)
        S3m = [s3[:, msk] for (msk, s3) in zip(self.par_psr_msk, S3)]          # S3 matrix with only the relevant columns

        # Need to swap the projector S3m = S3 @ G_F = P_A @ S3 @ G_F = U_A U_A^T
        U_A = [sl.svd(s3m, full_matrices=False)[0][:, :s3m.shape[1]] for s3m in S3m]

        # Transformation 4:
        # So now the data is:
        ULGNr = [ua.T @ lgnr for (ua, lgnr) in zip(U_A, LGNr)]
        S4 = [ua.T @ s3m for (ua, s3m) in zip(U_A, S3m)]

        # For testing, we could also use different coordinates:
        chi = ULGNr                         # Whitened data
        chi_tot = np.concatenate(chi)       # ''
        S = S4                              # Q transform
        Phi = PhiDiff                       # H1-H0 difference

        # build the list of block‐sizes and cumulative indices
        block_sizes = [s.shape[0] for s in S]
        idx = np.cumsum([0] + block_sizes)

        npsrs = len(block_sizes)

        # slice PhiDiff into npsrs×npsrs little blocks
        Phi_blocks = [
            [
                Phi[idx[i] : idx[i + 1], idx[j] : idx[j + 1]]
                for j in range(npsrs)
            ]
            for i in range(npsrs)
        ]

        # numerator (inner product) of os
        # denominator (trace) of os
        num = 0.0
        den2 = 0.0
        ddmat = np.zeros((npsrs, npsrs))
        Q = np.zeros_like(Phi)
        for i, (Si) in enumerate(S):   # The error is likely here
            for j, (Sj) in enumerate(S):
                SPS = Si @ Phi_blocks[i][j] @ Sj.T
                Q[idx[i] : idx[i + 1], idx[j] : idx[j + 1]] = SPS

                if not: self._npopt:
                    num += chi[i].dot(SPS @ chi[j])
                    ddmat[i, j] = np.trace(SPS @ SPS.T)
                    den2 += ddmat[i, j]

        if not self._npopt:
            den = np.sqrt(2*den2)   # Factor of 2 because of *real* (not complex) data
            Q = Q / den

        else:
            # Calculate the Neyman-Pearson-optimal statistic
            pass

        return num/den, chi_tot, Q

    def compute_os(self, params):
        os, _, _ = self._get_compressed_coordinates(params)
        return os

    def get_fixedpar_os_distribution(self, params, ds_min=-5, ds_max=20, cutoff=1e-6, limit=100, epsabs=1e-6):
        """For given parameters, get the OS distribution"""
        _, _, Q = self._get_compressed_coordinates(params)
        eigen_values = sl.eigvalsh(Q)

        xs = np.linspace(ds_min, ds_max, 1000)

        # pdf = gx2pdf(eigen_values, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)
        cdf = gx2cdf(eigen_values, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)

        return xs, cdf


    def get_fixedpar_pval(self, params, cutoff=1e-6, limit=100, epsabs=1e-6):
        os, _, Q = self._get_compressed_coordinates(params)
        eigen_values = sl.eigvalsh(Q)
        cdf_val = gx2cdf(eigen_values, [os], cutoff=cutoff, limit=limit, epsabs=epsabs)[0]

        return 1-cdf_val
