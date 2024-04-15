#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feb 2019, Bence Becsy (NANOGrav):   created file, implemented FeStatistics

Dec 2023, Kathrin Grunthal (EPTA):  code not suitable for Sherman-Morrison

Jan 2024, Kathrin Grunthal (EPTA):  corrected matrix operations and implementation
                                    code works with Sherman-Morrison

Mar 2024, Kathrin Grunthal (EPTA):  assure backward-compatibility
                                    remove nuisance code
"""

import numpy as np
import scipy.linalg as sl
from enterprise.signals import (
    gp_signals,
    parameter,
    selections,
    signal_base,
    utils,
    white_signals,
)
from enterprise_extensions import blocks
from . import cgw_model


class FeStat(object):
    """
    Class for the Fe-statistic.

    :param psrs: List of `enterprise` Pulsar instances.
    :param params: Dictionary of noise parameters.

    """

    def __init__(
        self, psrs, params=None, custom_models={}, inc_crn=False, orf=None, pta=None
    ):
        if pta is None:
            print("Creating a PTA with TM and WN")

            efac = parameter.Constant()
            equad = parameter.Constant()
            ef = white_signals.MeasurementNoise(efac=efac)
            eq = white_signals.EquadNoise(log10_equad=equad)

            tm = gp_signals.TimingModel(use_svd=True)

            s = eq + ef + tm

            model = []
            for p in psrs:
                model.append(s(p))
            self.pta = signal_base.PTA(model)

            # set white noise parameters
            if params is None:
                print("No noise dictionary provided!")
            else:
                self.pta.set_default_params(params)

        else:
            self.pta = pta

        self.psrs = psrs
        self.params = params
        self.Nvecs = None

    # --------------------------------------------------------------------------
    def compute_Fe(
        self,
        f0,
        gw_skyloc,
        brave=False,
        maximized_parameters=False,
        sky_scrambles=False,
        psrs_theta_phi=None,
    ):
        """
        Computes the Fe-statistic (see Ellis, Siemens, Creighton 2012).

        :param f0: GW frequency
        :param gw_skyloc: 2x{number of sky locations} array containing [theta, phi] for each queried sky location,
                          where theta=pi/2-DEC, phi=RA,
                          for singlge sky location use gw_skyloc= np.array([[theta,],[phi,]])
        :param brave: Skip sanity checks in linalg for speedup if True.
        :param maximized_parameters: Calculate maximized extrinsic parameters if True.

        :returns:
            fstat: value of the Fe-statistic
        :if maximized_parameters=True also returns:
            inc_max: Maximized value of inclination
            psi_max: Maximized value of polarization angle
            phase0_max: Maximized value of initial fhase
            h_max: Maximized value of amplitude

        """

        tref = 53000 * 86400

        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()

        if self.Nvecs is None:
            self.Nvecs = self.pta.get_ndiag(self.params)

        n_psr = len(self.psrs)
        N = np.zeros((n_psr, 4))
        M = np.zeros((n_psr, 4, 4))

        for idx, (psr, Nvec, TNT, phiinv, T) in enumerate(
            zip(self.psrs, self.Nvecs, TNTs, phiinvs, Ts)
        ):
            # calculate non-changing terms for the inner product
            # to avoid redundant calculations in innerProduct_rr
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            ntoa = len(psr.toas)

            A = np.zeros((4, ntoa))
            A[0, :] = 1 / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * (psr.toas - tref))
            A[1, :] = 1 / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * (psr.toas - tref))
            A[2, :] = 1 / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * (psr.toas - tref))
            A[3, :] = 1 / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * (psr.toas - tref))

            ip1 = innerProduct_rr(
                A[0, :], psr.residuals, Nvec, T, TNT, Sigma, brave=brave
            )
            ip2 = innerProduct_rr(
                A[1, :], psr.residuals, Nvec, T, TNT, Sigma, brave=brave
            )
            ip3 = innerProduct_rr(
                A[2, :], psr.residuals, Nvec, T, TNT, Sigma, brave=brave
            )
            ip4 = innerProduct_rr(
                A[3, :], psr.residuals, Nvec, T, TNT, Sigma, brave=brave
            )

            N[idx, :] = np.array([ip1, ip2, ip3, ip4])

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(4):
                for kk in range(4):
                    M[idx, jj, kk] = innerProduct_rr(
                        A[jj, :], A[kk, :], Nvec, T, TNT, Sigma, brave=brave
                    )

        fstat = np.zeros(gw_skyloc.shape[1])
        if maximized_parameters:
            inc_max = np.zeros(gw_skyloc.shape[1])
            psi_max = np.zeros(gw_skyloc.shape[1])
            phase0_max = np.zeros(gw_skyloc.shape[1])
            h_max = np.zeros(gw_skyloc.shape[1])

        for j, gw_pos in enumerate(gw_skyloc.T):
            NN = np.copy(N)
            MM = np.copy(M)
            for idx, psr in enumerate(self.psrs):
                if sky_scrambles:
                    if psrs_theta_phi is None:
                        ptheta = np.arccos(np.random.uniform(-1.0, 1.0))
                        pphi = np.random.uniform(0.0, 2 * np.pi)
                    else:
                        ptheta = psrs_theta_phi[idx, 0]
                        pphi = psrs_theta_phi[idx, 1]
                    pos = np.array(
                        [
                            np.cos(pphi) * np.sin(ptheta),
                            np.sin(pphi) * np.sin(ptheta),
                            np.cos(ptheta),
                        ]
                    )
                    F_p, F_c, _ = utils.create_gw_antenna_pattern(
                        pos, gw_pos[0], gw_pos[1]
                    )
                else:
                    F_p, F_c, _ = utils.create_gw_antenna_pattern(
                        psr.pos, gw_pos[0], gw_pos[1]
                    )
                NN[idx, :] *= np.array([F_p, F_p, F_c, F_c])
                MM[idx, :, :] *= np.array(
                    [
                        [F_p**2, F_p**2, F_p * F_c, F_p * F_c],
                        [F_p**2, F_p**2, F_p * F_c, F_p * F_c],
                        [F_p * F_c, F_p * F_c, F_c**2, F_c**2],
                        [F_p * F_c, F_p * F_c, F_c**2, F_c**2],
                    ]
                )

            N_sum = np.sum(NN, axis=0)
            M_sum = np.sum(MM, axis=0)

            # take inverse of M
            Minv = np.linalg.pinv(M_sum)

            fstat[j] = 0.5 * np.dot(N_sum, np.dot(Minv, N_sum))

            if maximized_parameters:
                a_hat = np.dot(Minv, N_sum)

                A_p = np.sqrt(
                    (a_hat[0] + a_hat[3]) ** 2 + (a_hat[1] - a_hat[2]) ** 2
                ) + np.sqrt((a_hat[0] - a_hat[3]) ** 2 + (a_hat[1] + a_hat[2]) ** 2)
                A_c = np.sqrt(
                    (a_hat[0] + a_hat[3]) ** 2 + (a_hat[1] - a_hat[2]) ** 2
                ) - np.sqrt((a_hat[0] - a_hat[3]) ** 2 + (a_hat[1] + a_hat[2]) ** 2)
                AA = A_p + np.sqrt(A_p**2 - A_c**2)
                # AA = A_p + np.sqrt(A_p**2 + A_c**2)

                # inc_max[j] = np.arccos(-A_c/AA)
                inc_max[j] = np.arccos(A_c / AA)

                two_psi_max = np.arctan2(
                    (A_p * a_hat[3] - A_c * a_hat[0]), (A_c * a_hat[2] + A_p * a_hat[1])
                )

                psi_max[j] = 0.5 * np.arctan2(np.sin(two_psi_max), -np.cos(two_psi_max))

                # convert from [-pi, pi] convention to [0,2*pi] convention
                if psi_max[j] < 0:
                    psi_max[j] += np.pi

                # correcting weird problem of degeneracy (psi-->pi-psi/2 and phi0-->2pi-phi0 keep everything the same)
                if psi_max[j] > np.pi / 2:
                    psi_max[j] += -np.pi / 2

                half_phase0 = -0.5 * np.arctan2(
                    A_p * a_hat[3] - A_c * a_hat[0], A_c * a_hat[1] + A_p * a_hat[2]
                )

                phase0_max[j] = np.arctan2(
                    -np.sin(2 * half_phase0), np.cos(2 * half_phase0)
                )

                # convert from [-pi, pi] convention to [0,2*pi] convention
                if phase0_max[j] < 0:
                    phase0_max[j] += 2 * np.pi

                zeta = np.abs(AA) / 4  # related to amplitude, zeta=M_chirp^(5/3)/D
                h_max[j] = zeta * 2 * (np.pi * f0) ** (2 / 3) * np.pi ** (1 / 3)

        if maximized_parameters:
            return fstat, inc_max, psi_max, phase0_max, h_max
        else:
            return fstat


# ==============================================================================
def innerProduct_rr(x, y, Nvec, Tmat, TNT, Sigma, brave=False):
    r"""
    Compute inner product using rank-reduced
    approximations for red noise/jitter
    Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

    :param x: vector timeseries 1
    :param y: vector timeseries 2
    :param Nvec: white noise Sherman Morrison object
    :param Tmat: Modified design matrix including red noise/jitter
    :param TNT: T^T N^{-1} T matrix from pta object
    :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)

    :return: inner product (x|y)
    """

    TNy = Nvec.solve(y, left_array=Tmat)
    TNx = Nvec.solve(x, left_array=Tmat)  # later used only in transposed version
    xNy = Nvec.solve(y, left_array=x)

    if brave:
        cf = sl.cho_factor(Sigma, check_finite=False)
        SigmaTNy = sl.cho_solve(cf, TNy, check_finite=False)
    else:
        cf = sl.cho_factor(Sigma)
        SigmaTNy = sl.cho_solve(cf, TNy)

    return xNy - TNx.T @ SigmaTNy
