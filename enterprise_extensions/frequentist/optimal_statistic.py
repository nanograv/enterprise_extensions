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


def imhof(u, x, eigen_values, output="cdf"):
    theta = (
        0.5 * np.sum(np.arctan(eigen_values[:, np.newaxis] * u), axis=0) - 0.5 * x * u
    )
    rho = np.prod((1.0 + (eigen_values[:, np.newaxis] * u) ** 2) ** 0.25, axis=0)

    rv = np.sin(theta) / (u * rho) if output == "cdf" else np.cos(theta) / rho

    return rv


def _select_gx2_eigenvalues(eigen_values, cutoff=1e-6):
    """Select eigenvalues for Imhof generalized-χ² integration.

    Parameters
    ----------
    eigen_values : array_like
        Eigenvalues of the quadratic-form matrix (e.g. from ``eigvalsh``,
        ascending).
    cutoff : float
        If ``cutoff > 1``, keep the ``int(cutoff)`` eigenvalues with largest
        absolute value. Otherwise keep eigenvalues with ``|λ| > cutoff``.
    """
    e = np.asarray(eigen_values, dtype=float).ravel()
    if e.size == 0:
        return e
    if cutoff > 1:
        k = min(int(cutoff), e.size)
        # eigvalsh is ascending: do not use e[:k] (that keeps the *smallest*).
        idx = np.argsort(np.abs(e))[::-1][:k]
        return e[idx]
    return e[np.abs(e) > cutoff]


def gx2pdf(eigen_values, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 PDF as a function of sx, based off of eigenvalues 'eigen_values'"""

    eigen_values = _select_gx2_eigenvalues(eigen_values, cutoff=cutoff)

    return np.array(
        [
            sint.quad(
                lambda u: float(imhof(u, x, eigen_values, output="pdf")),
                0,
                np.inf,
                limit=limit,
                epsabs=epsabs,
            )[0]
            / (2 * np.pi)
            for x in xs
        ]
    )


def gx2cdf(eigr, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 CDF as a function of sx, based off of eigenvalues 'eigr'"""

    eigen_values = _select_gx2_eigenvalues(eigr, cutoff=cutoff)

    return np.array(
        [
            0.5
            - sint.quad(
                lambda u: float(imhof(u, x, eigen_values)),
                0,
                np.inf,
                limit=limit,
                epsabs=epsabs,
            )[0]
            / np.pi
            for x in xs
        ]
    )


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


def ensure_2d_covmat(mat):
    """Make sure that the covariance matrix is 2D"""
    return mat if len(mat.shape) == 2 else np.diag(mat)


class DetectionStatistic(object):
    """
    Class for the Detection Statistic as used in the p-value paper.

    This class is specifically made for classical hypothesis testing. It
    requires an enterprise object for H0 and for H1. For now we assume that we
    keep the white noise fixed and that the only difference between the two
    hypotheses is in the way we model the Phi prior matrix.

    :param pta_h0:  The enterprise PTA object for the null hypothesis.
    :param pta_hs:  The enterprise PTA object for the signal hypothesis.
    :param dstype:  The type of detection statistic to use
                    DF, DFCC, NP, NPMV  -- default: DFCC (traditional OS)

    References:
     - Section 9, van Haasteren (2025), https://arxiv.org/abs/2506.10811
    """

    def __init__(
        self,
        pta_h0,
        pta_hs,
        dstype="DFCC",
    ):
        """Initialize the Detection statistic object."""
        # set up cache
        self._set_cache_parameters(pta_h0, pta_hs)
        self._np_stat, self._inc_auto_terms = self._get_dstype(dstype=dstype)

    def _get_dstype(self, dstype="DFCC"):
        """Set the type of detection statistic

        :param dstype:  The type of detection statistic
                        DF, DFCC, NP, NPMV

        :return:  (np_stat, inc_auto_terms)
        """
        key = str(dstype).upper()
        if key == "DF":
            return False, True
        if key == "DFCC":
            return False, False
        if key == "NP":
            return True, True
        if key == "NPMV":
            return True, False
        raise ValueError(
            "Unknown dstype={!r}; expected one of DF, DFCC, NP, NPMV".format(dstype)
        )

    def _set_cache_parameters(self, pta_h0, pta_hs):
        """Set the cache parameters according to the Section 9 in van Haasteren (2025)

        :param pta_h0:  The enterprise PTA object for the null hypothesis
        :param pta_hs:  The enterprise PTA object for the signal hypothesis.

        """
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
        self.pta_hs = pta_hs

        # Calculate lists of H0 quantities (11 seconds, only need it once)
        Tmat = pta_h0.get_basis({})  # List of 2D matrices
        self.Ndiag = pta_h0.get_ndiag({})  # Objects for sqrtsolve
        NT = [
            nd.sqrtsolve(tm) for (nd, tm) in zip(self.Ndiag, Tmat)
        ]  # List of 2D matrices
        self.G_T = [
            sl.svd(nt, full_matrices=False)[0] for nt in NT
        ]  # List of 2D matrices
        self.R = [gt.T @ nt for (gt, nt) in zip(self.G_T, NT)]  # List of 2D matrices

        # We do this here so we can avoid calculating the mask later
        # 1D array of all parameters
        xs = np.array(
            [par_val for p in pta_h0.params for par_val in np.atleast_1d(p.sample())]
        )
        pd = pta_hs.map_params(xs)
        Phi_0 = [
            ensure_2d_covmat(p) for p in pta_h0.get_phi(pd)
        ]  # Phi matrix of H0 -- 2D arrays
        BigPhiDiff = pta_hs.get_phi(pd) - sl.block_diag(*Phi_0)

        # Get only the non-zero elements of the BigPhiDiff matrix for selections later
        self.par_msk = np.sum(np.abs(BigPhiDiff), axis=1) > 0  # Mask for BigPhiDiff
        par_inds_offset = np.cumsum([0] + [len(p) for p in Phi_0])
        par_inds_start = par_inds_offset[:-1]
        par_inds_end = par_inds_offset[1:]
        par_inds_slices = [
            np.arange(p_start, p_end)
            for (p_start, p_end) in zip(par_inds_start, par_inds_end)
        ]
        self.par_psr_msk = [
            self.par_msk[slc] for slc in par_inds_slices
        ]  # Mask per psr for Phi_0 and Tmat

    def _get_compressed_coordinates(
        self, params, normalize_Q=True, force_include_auto=False
    ):
        """Returns detection statistic, chi, and Q for the given parameters.

        :param params:  The parameters to use for the calculation.
        :param normalize_Q:  Whether to normalize the Q matrix or not.
        :param force_include_auto:
            If True, always keep auto-correlation (i=j) blocks in Q. Used when
            the full deflection operator is needed (e.g. NP transform, or H_S
            covariance for ROC). If False, DFCC zeros auto blocks; NP/NPMV
            leave auto handling to ``deflection_to_np``.
        :return:  (ds, chi_tot, Q) with ds = chi^T Q chi after optional renorm

        """

        # These quantities have to be re-calculated for new hyperparameters
        Phi_0 = [
            ensure_2d_covmat(p) for p in self.pta_h0.get_phi(params)
        ]  # Phi matrix of H0 -- 2D arrays

        # This is a BIG matrix, but it's sparse. Not using that right now though
        # It's currently 0.4 secdonds for NG15
        BigPhiDiff = self.pta_hs.get_phi(params) - sl.block_diag(
            *Phi_0
        )  # 2D prior diff array

        # Inverse Noise matrix
        # List of matrix inverses -- 2D arrays
        C2i_0 = [inv_RPR(p, r) for (r, p) in zip(self.R, Phi_0)]

        # Get the Square-Root (we take it from the inv for numerical stability)
        C2i_0_svd = []
        for c in C2i_0:
            try:
                c_svd = sl.svd(c, full_matrices=True)
            except sl.LinAlgError:
                # GESVD is more numerically stable, but slower
                c_svd = sl.svd(c, full_matrices=True, lapack_driver="gesvd")

            C2i_0_svd.append(c_svd)

        # Select only non-singular values # Singular values -- 1D arrays
        C2i_sqrt_sing = [
            np.array([(np.sqrt(sv) if np.abs(sv) > 1e-10 else 0.0) for sv in s[1]])
            for s in C2i_0_svd
        ]
        # L matrix -- 2D arrays
        L_0 = [
            svd[0] @ np.diag(s) @ svd[0].T for (svd, s) in zip(C2i_0_svd, C2i_sqrt_sing)
        ]

        # Transformation 1: # List of 1D arrays (the weighted data)
        Nres = [
            nd.sqrtsolve(r - rp)
            for (nd, r, rp) in zip(
                self.Ndiag, self.pta_h0.get_residuals(), self.pta_h0.get_delay(params)
            )
        ]

        # Transformation 2: # List of 1D arrays (transformed data)
        self.GTNr = [gt.T @ nr for (gt, nr) in zip(self.G_T, Nres)]

        # Transformation 3:
        # From now also construct the filter transform, because it is of manaeable size
        LGNr = [
            l @ gnr for (l, gnr) in zip(L_0, self.GTNr)
        ]  # List of 1D arrays (transformed data)
        S3 = [
            l_bi @ r for (l_bi, r) in zip(L_0, self.R)
        ]  # List of 2D arrays (Q transformer)

        # Slice BigPhiDiff, because we only want non-zero items!
        PhiDiff = BigPhiDiff[self.par_msk, :][:, self.par_msk]

        # A = L_B^{-1} G^T_T T^{prime} = L_B^{-1} @ R
        # T^{prime} = L_N^{-1} Tmat
        # P_T T^{prime} = T^{prime}   ===> P_T = G_T G_T^T
        # S3m = S3 @ G_F (is same thing as selecting the columns of S3)
        # S3 matrix with only the relevant columns
        S3m = [s3[:, msk] for (msk, s3) in zip(self.par_psr_msk, S3)]

        # Need to swap the projector S3m = S3 @ G_F = P_A @ S3 @ G_F = U_A U_A^T
        U_A = [sl.svd(s3m, full_matrices=False)[0][:, : s3m.shape[1]] for s3m in S3m]

        # Transformation 4:
        # So now the data is:
        ULGNr = [ua.T @ lgnr for (ua, lgnr) in zip(U_A, LGNr)]
        S4 = [ua.T @ s3m for (ua, s3m) in zip(U_A, S3m)]

        # For testing, we could also use different coordinates:
        chi = ULGNr  # Whitened data
        chi_tot = np.concatenate(chi)  # ''
        S = S4  # Q transform
        Phi = PhiDiff  # H1-H0 difference

        # build the list of block‐sizes and cumulative indices
        block_sizes = [s.shape[0] for s in S]
        idx = np.cumsum([0] + block_sizes)
        self._idx = idx

        npsrs = len(block_sizes)

        # slice PhiDiff into npsrs×npsrs little blocks
        Phi_blocks = [
            [Phi[idx[i] : idx[i + 1], idx[j] : idx[j + 1]] for j in range(npsrs)]
            for i in range(npsrs)
        ]

        # Build full deflection filter Q (auto + cross). Drop auto later if DFCC.
        Q = np.zeros_like(Phi)
        for i, Si in enumerate(S):
            for j, Sj in enumerate(S):
                SPS = Si @ Phi_blocks[i][j] @ Sj.T
                Q[idx[i] : idx[i + 1], idx[j] : idx[j + 1]] = SPS

        # DFCC (deflection, cross-only): zero pulsar auto blocks before renorm.
        # NP/NPMV keep full Q here; auto policy is applied in deflection_to_np.
        remove_auto = (
            (not force_include_auto)
            and (not self._np_stat)
            and (not self._inc_auto_terms)
        )
        if remove_auto:
            self._zero_block_diagonal(Q)

        num = float(chi_tot.dot(Q @ chi_tot))
        den2 = float(np.trace(Q @ Q.T))
        # Factor of 2 because of *real* (not complex) data
        den = np.sqrt(2.0 * den2) if den2 > 0.0 else 1.0

        if normalize_Q:
            Q = Q / den

        return num / den, chi_tot, Q

    def _zero_block_diagonal(self, M):
        """Zero the (a==b) block-diagonal using self._idx as block boundaries (in place)."""
        for i in range(len(self._idx) - 1):
            ii0, ii1 = self._idx[i], self._idx[i + 1]
            M[ii0:ii1, ii0:ii1] = 0.0

        return M

    def deflection_to_np(self, Q, remove_auto_terms=True):
        """
        Transform what we have into a Neyman-Pearson-Weighted-optimal statistic
        This is all allowed here at this stage, because B commutes with (B + I)
        That means they have a common set of eigenvectors, and we can use the
        same low-rank basis for B and B + I. That was really really fortunate
        and cool!

        :param Q: The deflection-optimal detection statistic filter
        :param remove_auto_terms: If True, remove the auto-terms from the filter
        :return: The Neyman-Pearson optimal detection statistic filter
        """
        cfB = sl.cho_factor(Q + np.identity(len(Q)), lower=True)
        BBi = sl.cho_solve(cfB, Q)

        if remove_auto_terms:
            # Only keep the cross-terms
            self._zero_block_diagonal(BBi)

        BBBi = np.dot(BBi, BBi)
        den = np.sqrt(2 * np.trace(BBBi))
        return BBi / den

    def get_deflection_coordinates(self, params, normalize_Q=True):
        """Return the deflection-optimal detection statistic and filter.

        For DFCC (``_inc_auto_terms`` False), pulsar auto blocks of Q are zeroed
        before renormalization. NP/NPMV should use ``get_np_coordinates``.

        :param params:  The parameters to use for the calculation.
        :param normalize_Q:  Whether to normalize the Q matrix or not.
        :return:  (ds, chi_tot, Q)

        """
        return self._get_compressed_coordinates(params, normalize_Q=normalize_Q)

    def get_np_coordinates(self, params):
        """Return the Neyman-Pearson optimal detection statistic

        The Neyman-Pearson optimal statistic is easily derived from the
        deflection-optimal statistic. So just use that relationship and
        re-normalize the filter

        :param params:  The parameters to use for the calculation.
        :return:  (ds, chi_tot, Qnp)
        """
        # Full deflection Q (with auto); NP/NPMV auto policy applied next.
        _, chi_tot, Q = self._get_compressed_coordinates(
            params, normalize_Q=False, force_include_auto=True
        )
        Qnp = self.deflection_to_np(Q, remove_auto_terms=not self._inc_auto_terms)
        ds = np.sum(chi_tot * np.dot(Qnp, chi_tot))

        return ds, chi_tot, Qnp

    def compute_os(self, params):
        """Get the optimal statistic value / coordinates

        :param params:  The parameters to use for the calculation.

        """

        if self._np_stat:
            return self.get_np_coordinates(params)[0]
        else:
            return self.get_deflection_coordinates(params)[0]

    def get_fixedpar_os_distribution(
        self,
        params,
        ds_min=-10,
        ds_max=20,
        cutoff=1e-6,
        limit=100,
        epsabs=1e-6,
        Q=None,
        kind="cdf",
    ):
        """For given parameters, get the OS distribution PDF/CDF under H0

        :param params:  The parameters to use for the calculation.
        :param ds_min:  The minimum value of the OS distribution.
        :param ds_max:  The maximum value of the OS distribution.
        :param cutoff:  If >1, keep that many largest-|λ| eigenvalues; else |λ|>cutoff
        :param limit:   An upper bound on the number of subintervals used
                        in the adaptive integration algorithm
        :param epsabs:  The absolute error tolerance for the integration
        :param Q:       The Q matrix to use for the calculation. If None,
                        it will be computed from the parameters.
        :param kind:    The kind of distribution to return, either 'cdf' or 'pdf'

        :return: A tuple of (ds, dist) where ds are the detection statistic values

        """
        if Q is None:
            # compute_os returns only a scalar; fetch the filter explicitly.
            if self._np_stat:
                _, _, Q = self.get_np_coordinates(params)
            else:
                _, _, Q = self.get_deflection_coordinates(params)
        eigen_values = sl.eigvalsh(Q)

        xs = np.linspace(ds_min, ds_max, 1000)

        if kind == "cdf":
            dist = gx2cdf(eigen_values, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)
        elif kind == "pdf":
            dist = gx2pdf(eigen_values, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)
        else:
            raise ValueError("Parameter 'kind' has to be cdf/pdf")

        return xs, dist

    def get_roc_curve(
        self,
        params,
        ds_min=-10,
        ds_max=20,
        cutoff=1e-6,
        limit=100,
        epsabs=1e-6,
        calc_pdf=False,
    ):
        """For given parameters, get the ROC curve

        :param params:  The parameters to use for the calculation.
        :param ds_min:  The minimum value of the OS distribution.
        :param ds_max:  The maximum value of the OS distribution.
        :param cutoff:  If >1, keep that many largest-|λ| eigenvalues; else |λ|>cutoff
        :param limit:   An upper bound on the number of subintervals used
                        in the adaptive integration algorithm
        :param epsabs:  The absolute error tolerance for the integration
        :param calc_pdf: Whether to calculate the PDF as well. If False,
                        only the CDF will be calculated.

        :return: A tuple of (ds, cdf_h0, cdf_hs) if only CDFs are calculated
                 A tuple of (ds, pdf_h0, cdf_h0, pdf_hs, cdf_hs) if also PFDs

        Notes
        -----
        Under H_S the compressed data have covariance I+B with B the *full*
        deflection operator (auto + cross). Auto-term removal for DFCC/NPMV is
        applied only to the filter Q used as the detection statistic, not to
        that covariance. The optional ``xs -= tr(Q_H0)`` shift when auto terms
        are included is a plotting convenience and is not used in
        ``get_fixedpar_pval``.
        """
        # Full deflection B for H_S whitening covariance C = I + B.
        _, chi_tot, Q_full = self._get_compressed_coordinates(
            params, normalize_Q=False, force_include_auto=True
        )
        C = Q_full + np.identity(len(chi_tot))
        L = sl.cholesky(C, lower=True)

        if self._np_stat:
            Q = self.deflection_to_np(
                Q_full, remove_auto_terms=not self._inc_auto_terms
            )
        else:
            Q = np.array(Q_full, copy=True)
            if not self._inc_auto_terms:
                # DFCC: cross-correlation filter only
                self._zero_block_diagonal(Q)
            den2 = float(np.trace(Q @ Q.T))
            den = np.sqrt(2.0 * den2) if den2 > 0.0 else 1.0
            Q = Q / den

        # Whiten filter under H_S (already white under H_0)
        QH0 = Q
        QHS = L.T @ Q @ L

        xs, cdf_h0 = self.get_fixedpar_os_distribution(
            params,
            ds_min=ds_min,
            ds_max=ds_max,
            cutoff=cutoff,
            limit=limit,
            epsabs=epsabs,
            Q=QH0,
        )

        _, cdf_hs = self.get_fixedpar_os_distribution(
            params,
            ds_min=ds_min,
            ds_max=ds_max,
            cutoff=cutoff,
            limit=limit,
            epsabs=epsabs,
            Q=QHS,
        )

        # Plotting convenience for auto-inclusive stats (not used in p-values)
        if self._inc_auto_terms:
            xs = xs - np.trace(QH0)

        if calc_pdf:
            _, pdf_h0 = self.get_fixedpar_os_distribution(
                params,
                ds_min=ds_min,
                ds_max=ds_max,
                cutoff=cutoff,
                limit=limit,
                epsabs=epsabs,
                Q=QH0,
                kind="pdf",
            )

            _, pdf_hs = self.get_fixedpar_os_distribution(
                params,
                ds_min=ds_min,
                ds_max=ds_max,
                cutoff=cutoff,
                limit=limit,
                epsabs=epsabs,
                Q=QHS,
                kind="pdf",
            )

            return xs, pdf_h0, cdf_h0, pdf_hs, cdf_hs

        else:
            return xs, cdf_h0, cdf_hs

    def get_fixedpar_pval(self, params, cutoff=1e-6, limit=200, epsabs=1e-9):
        """calculate the p-value for the OS under H0

        :param params:  The parameters to use for the calculation.
        :param ds_min:  The minimum value of the OS distribution.
        :param ds_max:  The maximum value of the OS distribution.
        :param cutoff:  Only eigenvalues above this value are included
        :param limit:   An upper bound on the number of subintervals used
                        in the adaptive integration algorithm
        :param epsabs:  The absolute error tolerance for the integration

        :return: The p-value for the OS under H0, which is 1 - CDF(OS)

        """
        if self._np_stat:
            os, _, Q = self.get_np_coordinates(params)
        else:
            os, _, Q = self.get_deflection_coordinates(params)

        eigen_values = sl.eigvalsh(Q)
        cdf_val = gx2cdf(eigen_values, [os], cutoff=cutoff, limit=limit, epsabs=epsabs)[
            0
        ]

        return 1 - cdf_val
