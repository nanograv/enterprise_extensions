# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate as interp
from enterprise import constants as const
from enterprise.signals import signal_base


@signal_base.function
def param_hd_orf(pos1, pos2, a=1.5, b=-0.25, c=0.5, diag=1.0):
    """
    Pre-factor parametrized Hellings & Downs spatial correlation function.

    :param: a, b, c:
        coefficients of H&D-like curve [default=1.5,-0.25,0.5].
    :param diag:
        Auto-correlation term

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    if np.all(pos1 == pos2):
        return diag
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        params = [a, b, c]
        return params[0] * omc2 * np.log(omc2) + params[1] * omc2 + params[2]


@signal_base.function
def spline_orf(pos1, pos2, params, diag=1.0):
    """
    Agnostic spline-interpolated spatial correlation function. Spline knots
    are placed at edges, zeros, and minimum of H&D curve. Changing locations
    will require manual intervention to create new function.

    :param: params
        spline knot amplitudes.
    :param diag:
        Auto-correlation term

    Reference: Taylor, Gair, Lentati (2013), https://arxiv.org/abs/1210.6014
    Author: S. R. Taylor (2020)

    """
    if np.all(pos1 == pos2):
        return diag
    else:
        # spline knots placed at edges, zeros, and minimum of H&D
        spl_knts = (
            np.array([1e-3, 25.0, 49.3, 82.5, 121.8, 150.0, 180.0]) * np.pi / 180.0
        )
        omc2_knts = (1 - np.cos(spl_knts)) / 2
        finterp = interp.interp1d(omc2_knts, params, kind="cubic")

        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return finterp(omc2)


@signal_base.function
def interp_orf(pos1, pos2, params, diag=1.0, bins=None):
    """
    Approximation of the spatial correlation function at angles
    across angular separation space with linear interpolation in between.

    :param params:
        inter-pulsar correlation bin amplitudes that are interpolated over
    :param diag:
        Auto-correlation term

    Reference: Goncharov et al. (2021), https://arxiv.org/abs/2107.12112
    Author: B. Goncharov (2021)

    """
    if np.all(pos1 == pos2):
        return diag
    else:
        # angles/bins in angsep space
        if bins is None:
            bins = (
                np.array([1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 150.0, 180.0])
                * np.pi
                / 180.0
            )
        else:
            bins = bins * np.pi / 180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        return np.interp(angsep, bins, params)


@signal_base.function
def bin_orf(pos1, pos2, params, diag=1.0, bins=None):
    """
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.

    :param: params
        inter-pulsar correlation bin amplitudes.
    :param diag:
        Auto-correlation term

    Author: S. R. Taylor (2020)

    """
    if np.all(pos1 == pos2):
        return diag
    else:
        # angles/bins in angsep space
        if bins is None:
            bins = (
                np.array([1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 150.0, 180.0])
                * np.pi
                / 180.0
            )
        else:
            bins = bins * np.pi / 180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        idx = np.digitize(angsep, bins)
        return params[idx - 1]


@signal_base.function
def freq_hd(pos1, pos2, params):
    """
    Frequency-dependent Hellings & Downs spatial correlation function.
    Implemented as a model that only enforces H&D inter-pulsar correlations
    after a certain number of frequencies in the spectrum. The first set of
    frequencies are uncorrelated.

    :param: params
        params[0] is the number of components in the stochastic process.
        params[1] is the frequency at which to start the H&D inter-pulsar
        correlations (indexing from 0).

    Reference: Taylor et al. (2017), https://arxiv.org/abs/1606.09180
    Author: S. R. Taylor (2020)

    """
    nfreq = params[0]
    orf_ifreq = params[1]
    if np.all(pos1 == pos2):
        return np.ones(2 * nfreq)
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        hd_coeff = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
        hd_coeff *= np.ones(2 * nfreq)
        hd_coeff[: 2 * orf_ifreq] = 0.0
        return hd_coeff


signal_base.function


def legendre_orf(pos1, pos2, params, diag=1.0):
    """
    Legendre polynomial spatial correlation function.
    Assumes process normalization when autocorrelation signature is set to 1.
    A separate function is needed to use a "split likelihood" model with this
    Legendre process when the autocorrelation signature is set to zero. To be used
    in a "split likelihood" model with an additional common uncorrelated red process.
    The latter is necessary to regularize the overall Phi covariance matrix.

    :param: params
        Legendre polynomial amplitudes describing the Legendre series approximation
        to the inter-pulsar correlation signature.
        H&D coefficients are a_0=0, a_1=0, a_2=0.3125, a_3=0.0875, ...

    :param: diag
        float or parameter to set the diagonal auto correlation terms
        default: 1 (auto correlation enabled)
        1e-20 (zero diagonal cross correlation only,
        to be used in a "split likelihood" model with
        an additional common uncorrelated red process)
        enterprise parameter to be sampled

    Reference: Gair et al. (2014), https://arxiv.org/abs/1406.4664
    Author: S. R. Taylor (2020)

    """
    if np.all(pos1 == pos2):
        return diag
    else:
        costheta = np.dot(pos1, pos2)
        orf = np.polynomial.legendre.legval(costheta, params)
        return orf


@signal_base.function
def chebyshev_orf(pos1, pos2, params, diag=1.0):
    """
    Chebyshev polynomial decomposition of the spatial correlation function.

    :param: diag
        float or parameter to set the diagonal autocorrelation terms
        default: 1 (autocorrelation enabled)
        1e-20 (zero diagonal cross correlation only,
        to be used in a "split likelihood" model with
        an additional common uncorrelated red process)
        enterprise parameter to be sampled

    Reference: Chen et al. (2021), https://arxiv.org/abs/2110.13184
    Author: S. Chen (2021)
    """
    if np.all(pos1 == pos2):
        return diag
    else:
        zij = np.arccos(np.dot(pos1, pos2))
        x = (zij - 0.5 * np.pi) * 2.0 / np.pi
        c1, c2, c3, c4 = params
        ch = c1 + c2 * x + c3 * (2.0 * x * x - 1.0) + c4 * (4.0 * x**3 - 3.0 * x)
        if -1.0 < ch < 1.0:
            return ch
        else:
            return 100.0


@signal_base.function
def hd_orf(pos1, pos2, diag=1.0):
    """Hellings & Downs spatial correlation function."""
    if np.all(pos1 == pos2):
        return diag
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5


@signal_base.function
def dipole_orf(pos1, pos2, diag=1.0):
    """Dipole spatial correlation function."""
    if np.all(pos1 == pos2):
        return diag + 1e-5
    else:
        return np.dot(pos1, pos2)


@signal_base.function
def monopole_orf(pos1, pos2, diag=1.0):
    """Monopole spatial correlation function."""
    if np.all(pos1 == pos2):
        return diag + 1e-5
    else:
        return 1.0


@signal_base.function
def param_monopole_orf(pos1, pos2, c=1.0, diag=1.0):
    """
    Parametrized Monopole spatial correlation function.

    :param: c:
        coefficient of the Monopole correlation normalization

    :param: diag
        float or parameter to set the diagonal auto correlation terms
        default: 1 (auto correlation enabled)
        1e-20 (zero diagonal cross correlation only,
        to be used in a "split likelihood" model with
        an additional common uncorrelated red process)
        enterprise parameter to be sampled

    Author: S. Chen (2022)
    """
    if np.all(pos1 == pos2):
        if diag is not None:
            return diag + 1e-5
        else:
            return c + 1e-5
    else:
        return c


@signal_base.function
def param_multiple_corr_orf(pos1, pos2, mp=0.0, dp=0.0, hd=1.0, diag=1.0):
    """
    Parametrized multiple component spatial correlation function.

    :param mp:
        coefficient of the Monopole correlation normalization

    :param dp:
        coefficient of the Dipole correlation normalization

    :param hd:
        coefficient of the Hellings & Downs correlation normalization

    :param: diag
        float or parameter to set the diagonal auto correlation terms
        default: 1 (auto correlation enabled)
        1e-20 (zero diagonal cross correlation only,
        to be used in a "split likelihood" model with
        an additional common uncorrelated red process)
        enterprise parameter to be sampled

    Author: S. Chen (2023)
    """
    if np.all(pos1 == pos2):
        return diag + 1e-5
    else:
        c = np.dot(pos1, pos2)
        omc2 = (1 - c) / 2
        hd_corr = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
        return mp + dp * c + hd * hd_corr


@signal_base.function
def generalized_gwpol_orf(pos1, pos2, hd=1.0, st=0.0, vl=0.0, diag=1.0):
    """General GW Correlations. The VL mode is not normalized."""
    if np.all(pos1 == pos2):
        return diag + 1e-5
    else:
        c = np.dot(pos1, pos2)
        omc2 = (1 - c) / 2
        hd_corr = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

        st_corr = 1 / 8 * (3.0 + c)

        vl_corr = 3 * np.log10(2 / (1 - c)) - 4 * c - 3

        return hd * hd_corr + st * st_corr + vl * vl_corr


@signal_base.function
def anis_orf(pos1, pos2, params, **kwargs):
    """Anisotropic GWB spatial correlation function."""

    anis_basis = kwargs["anis_basis"]
    psrs_pos = kwargs["psrs_pos"]
    lmax = kwargs["lmax"]

    psr1_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos1)][0]
    psr2_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos2)][0]

    clm = np.zeros((lmax + 1) ** 2)
    clm[0] = 2.0 * np.sqrt(np.pi)
    if lmax > 0:
        clm[1:] = params

    return sum(
        clm[ii] * basis
        for ii, basis in enumerate(
            anis_basis[: (lmax + 1) ** 2, psr1_index, psr2_index]
        )
    )


@signal_base.function
def gw_monopole_orf(pos1, pos2):
    """
    GW-monopole Correlations. This phenomenological correlation pattern can be
    used in Bayesian runs as the simplest type of correlations.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1 / 2


@signal_base.function
def gw_dipole_orf(pos1, pos2):
    """
    GW-dipole Correlations.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1 / 2 * np.dot(pos1, pos2)


@signal_base.function
def st_orf(pos1, pos2):
    """
    Scalar tensor correlations as induced by the breathing polarization mode of gravity.
    Author: N. Laal (2020)
    """
    if np.all(pos1 == pos2):
        return 1
    else:
        return 1 / 8 * (3.0 + np.dot(pos1, pos2))


@signal_base.function
def gt_orf(pos1, pos2, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.

    Author: N. Laal (2020)

    """
    if np.all(pos1 == pos2):
        return 1
    else:
        k = 1 / 2 * (1 - np.dot(pos1, pos2))
        return 1 / 8 * (3 + np.dot(pos1, pos2)) + (1 - tau) * 3 / 4 * k * np.log(k)


@signal_base.function
def generalized_gwpol_psd(
    f,
    log10_A_tt=-15,
    log10_A_st=-15,
    alpha_tt=-2 / 3,
    alpha_alt=-1,
    log10_A_vl=-15,
    log10_A_sl=-15,
    kappa=0,
    p_dist=1.0,
):
    """
    General powerlaw spectrum allowing for existence of all possible modes of gravity as
    predicted by a general metric spacetime theory and generated by a binary system.
    The SL and VL modes' powerlaw relations are not normalized.

    :param: f
        A list of considered frequencies
    :param: log10_A_tt
        Amplitude of the tensor transverse mode
    :param: log10_A_st
        Amplitude of the scalar transverse mode
    :param: log10_A_vl
        Amplitude of the vector longitudinal mode
    :param: log10_A_sl
        Amplitude of the scalar longitudinal mode
    :param: kappa
        Relative amplitude of dipole radiation over quadrupolar radiation
    :param: p_dist
        Pulsar distance in kpc
    :param: alpha_tt
        spectral index of the TT mode.
    :param: alpha_alt
        spectral index of the non-Einsteinian modes.

    Reference: Cornish et al. (2017), https://arxiv.org/abs/1712.07132
    Author: S. R. Taylor, N. Laal (2020)

    """

    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    euler_e = 0.5772156649
    pdist = p_dist * const.kpc / const.c

    orf_aa_tt = (2 / 3) * np.ones(len(f))
    orf_aa_st = (2 / 3) * np.ones(len(f))
    orf_aa_vl = 2 * np.log(4 * np.pi * f * pdist) - 14 / 3 + 2 * euler_e
    orf_aa_sl = (
        np.pi**2 * f * pdist / 4 - np.log(4 * np.pi * f * pdist) + 37 / 24 - euler_e
    )

    prefactor = (1 + kappa**2) / (1 + kappa**2 * (f / const.fyr) ** (-2 / 3))
    gwpol_amps = 10 ** (2 * np.array([log10_A_tt, log10_A_st, log10_A_vl, log10_A_sl]))
    gwpol_factors = np.array(
        [
            orf_aa_tt * gwpol_amps[0],
            orf_aa_st * gwpol_amps[1],
            orf_aa_vl * gwpol_amps[2],
            orf_aa_sl * gwpol_amps[3],
        ]
    )

    S_psd = (
        prefactor
        * (
            gwpol_factors[0, :] * (f / const.fyr) ** (2 * alpha_tt)
            + np.sum(gwpol_factors[1:, :], axis=0) * (f / const.fyr) ** (2 * alpha_alt)
        )
        / (8 * np.pi**2 * f**3)
    )

    return S_psd * np.repeat(df, 2)
