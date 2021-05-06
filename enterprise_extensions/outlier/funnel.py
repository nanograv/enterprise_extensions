#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains class for funnel transformation object, which performs a non-centered
reparametrization on an existing likelihood object.

Class methods include the funnel transformation and its auxilliary functions,
and computing the log likelihood and gradient in the transformed coordinates.

Requirements:
    numpy
    scipy
"""


import numpy as np
import scipy.linalg as sl

from .base import ptaLikelihood
from .choleskyext_omp import cython_dL_update_omp


class Funnel(ptaLikelihood):
    """This class implements a funnel transformation, which is a non-centered
    reparametrization of the coordinates to combat a potential sampling issue
    referenced as `Neal's funnel`.

    For more information about Neal's funnel, check
    Neal, Radford M. 2003. “Slice Sampling.” Annals of Statistics 31 (3): 705–67.

    :param enterprise_pintpulsar: `enterprise.PintPulsar` object (with drop_pintpsr=False)
    """
    def __init__(self, enterprise_pintpulsar):
        """Constructor method
        """
        super(Funnel, self).__init__(enterprise_pintpulsar)

        self.funnelmin = None
        self.funnelmax = None
        self.funnelstart = None

        self.Zmask_M = None
        self.Zmask_F = None
        self.Zmask_U = None

        self.ZNZ = None
        self.ZNyvec = None

        self.fnlslc = None
        self.fnl_Beta_inv = None
        self.fnl_Sigma = None
        self.fnl_L = None
        self.fnl_Li = None
        self.fnl_mu = None
        self.fnl_dL_M = None
        self.fnl_dL_tj = None

        self.log_jacob = None
        self.funnel_gradient = None

        self.init_funnel_model()


    def init_funnel_model(self):
        """Run initiliazation functions for funnel transformation object
        """
        self.setFunnelAuxiliary()
        self.getLowLevelZmask()
        self.lowLevelStart()
        self.initFunnelBounds()


    def lowLevelStart(self):
        """Set funnel transform object parameter start vector by setting all
        low-level parameter entries equal to 0.1
        """
        lowlevelpars = ['timingmodel', 'fouriermode', 'jittermode']

        if self.basepstart is None:
            return

        pstart = self.basepstart.copy()

        for _, sig in self.signals.items():
            if sig['type'] in lowlevelpars:
                msk = sig['msk']
                pstart[msk] = 0.1

        self.funnelstart = pstart


    def setFunnelAuxiliary(self):
        """Compute auxilliary vector and matrix quantities necessary for
        performing funnel transformation
        """
        Nvec = self.psr.toaerrs**2
        ZNyvec = np.dot(self.Zmat.T, self.psr.residuals / Nvec)
        ZNZ = np.dot(self.Zmat.T / Nvec, self.Zmat)

        self.ZNZ = ZNZ
        self.ZNyvec = ZNyvec


    def full_forward(self, x):
        """Apply funnel transformation to parameter vector

        :param x: Parameter vector to be transformed
        :return: Parameter vector with funnel transform
        """
        p = np.atleast_2d(x.copy())
        p[0, self.fnlslc] = np.dot(self.fnl_L.T, p[0, self.fnlslc] - self.fnl_mu)
        return p.reshape(x.shape)


    def full_backward(self, p):
        """Undo the funnel transformation

        :param p: Parameter vector under funnel transform
        :return: Parameter vector in original coordinates
        """
        x = np.atleast_2d(p.copy())
        x[0, self.fnlslc] = np.dot(self.fnl_Li.T, x[0, self.fnlslc]) + self.fnl_mu
        return x.reshape(p.shape)


    def multi_full_backward(self, p):
        """Undo the interval transformation for 2D array of samples

        :param p: Array of parameter vectors under interval transform
        :return: Array of parameter vectors in original coordinates
        """
        x = np.atleast_2d(p.copy())
        for ii, xx in enumerate(x):
            self.funnelTransform(xx)
            x[ii, self.fnlslc] = np.dot(self.fnl_Li.T, x[ii, self.fnlslc]) + self.fnl_mu
        return x.reshape(p.shape)



    def initFunnelBounds(self):
        """Forward transform the minimum and maximum parameter vectors
        """
        pmin = self.basepmin.copy()
        pmax = self.basepmax.copy()

        self.funnelTransform(pmin)
        self.funnelmin = self.full_forward(pmin)

        self.funnelTransform(pmax)
        self.funnelmax = self.full_forward(pmax)


    def getLowLevelMask(self):
        """Return array of indices corresponding to low-level parameters in
        the parameter vector

        :return: Array of vector indices
        """
        slc = np.array([], dtype=np.int)

        if 'timingmodel' in self.signals.keys():
            pslc = self.signals['timingmodel']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'fouriermode' in self.signals.keys():
            pslc = self.signals['fouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'dmfouriermode' in self.signals.keys():
            pslc = self.signals['dmfouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'jittermode' in self.signals.keys():
            pslc = self.signals['jittermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))

        return slc


    def getLowLevelZmask(self):
        """Set index masks for different sets of low-level parameters in the
        Z matrix
        """
        slc = np.array([], dtype=np.int)

        if 'timingmodel' in self.signals.keys():
            npars = self.signals['timingmodel']['numpars']
            self.Zmask_M = slice(0, npars)
            slc = np.append(slc, np.arange(0, npars))
        if 'fouriermode' in self.signals.keys():
            npars = self.signals['fouriermode']['numpars']
            if slc.size > 0:
                self.Zmask_F = slice(slc[-1]+1, slc[-1]+1+npars)
                slc = np.append(slc, np.arange(slc[-1]+1, slc[-1]+1+npars))
            else:
                self.Zmask_F = slice(0, npars)
                slc = np.append(slc, np.arange(0, npars))
        if 'jittermode' in self.signals.keys():
            npars = self.signals['jittermode']['numpars']
            if slc.size:
                self.Zmask_U = slice(slc[-1]+1, slc[-1]+1+npars)
                slc = np.append(slc, np.arange(slc[-1]+1, slc[-1]+1+npars))
            else:
                self.Zmask_U = slice(0, npars)
                slc = np.append(slc, np.arange(0, npars))


    def getBetaInv(self):
        """Return the inverse of the diagonal Beta matrix. Beta is the
        concatenation of the Phi and J vectors.

        :return: Inverse of the diagonal of Beta matrix
        """
        Beta_inv_diag = np.zeros(len(self.ZNZ))

        if 'fouriermode' in self.ptaparams.keys():
            phivec = self.Phivec

            Beta_inv_diag[self.Zmask_F] = 1.0 / phivec

        if 'dmfouriermode' in self.ptaparams.keys():
            pass

        if 'jittermode' in self.ptaparams.keys():
            Beta_inv_diag[self.Zmask_U] = 1.0 / self.Jvec

        return Beta_inv_diag


    def getSigma(self, Beta_inv_diag):
        """Return Sigma, L, and Li, where Sigma = Li Li^T.
        Off-diagonal elements of Sigma_inv come from the matrix product ZNZ,
        and diagonal elements are the sum of the diagonal of ZNZ and the
        diagonal of Beta_inv. L is then the Cholesky decomposition of
        Sigma_inv

        :param Beta_inv_diag: Diagonal of the inverse Beta matrix
        :return Sigma: Sigma matrix
        :return L: Cholesky decomposition of Sigma_inv
        :return Li: Solution x to Ax = I, where A is the triangular matrix
            found from `L`, and I is the identity matrix
        """
        Sigma_inv = np.copy(self.ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)

        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)

        L = sl.cholesky(Sigma_inv, lower=True)
        Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)
        cf = (L, True)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv))), L, Li


    def funnelTransform(self, parameters, set_hyper_params=True, calc_gradient=True):
        """Perform funnel transformation, and compute the log Jacobian and
        gradient for the transformation

        :param parameters: Vector of signal parameters
        :param set_hyper_params: Update hyperparameter values before doing
            then transformation, default is True
        :param calc_gradient: Include gradient calculation, default is True
        """
        if set_hyper_params:
            self.set_hyperparameters(parameters, calc_gradient=calc_gradient)

        self.fnlslc = self.getLowLevelMask()

        self.fnl_Beta_inv = self.getBetaInv()
        self.fnl_Sigma, self.fnl_L, self.fnl_Li = self.getSigma(self.fnl_Beta_inv)
        self.fnl_mu = np.dot(self.fnl_Sigma, self.ZNyvec)

        log_jacob = 0.0
        gradient = np.zeros_like(parameters)

        lowlevel_pars = np.dot(self.fnl_Li.T, parameters[self.fnlslc])
        self.fnl_dL_M, self.fnl_dL_tj = cython_dL_update_omp(self.fnl_L, self.fnl_Li, lowlevel_pars)

        log_jacob += np.sum(np.log(np.diag(self.fnl_Li)))

        if 'fouriermode' in self.ptaparams.keys():

            for key, d_Phivec_d_param in self.d_Phivec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_F] = self.fnl_Beta_inv[self.Zmask_F]**2 * d_Phivec_d_param

                gradient[key] += np.sum(self.fnl_dL_tj[self.Zmask_F] * BdB[self.Zmask_F])

        if 'jittermode' in self.ptaparams.keys():

            for key, d_Jvec_d_param in self.d_Jvec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_U] = self.fnl_Beta_inv[self.Zmask_U]**2 * d_Jvec_d_param

                gradient[key] += np.sum(self.fnl_dL_tj[self.Zmask_U] * BdB[self.Zmask_U])

        self.log_jacob = log_jacob
        self.funnel_gradient = gradient


    def dxdp_nondiag(self, parameters, ll_grad, set_hyper_params=False):
        """Return Jacobian of interval transformation (non-diagonal
        derivative of x wrt p)

        :param parameters: Vector of signal parameters
        :param ll_grad: Gradient of log likelihood for original parameters
        :param set_hyper_params: Boolean to decide whether or not to update
           hyperparameter auxilliary quantities (like Phi and N vectors) using
           input `parameters`. Default is False
        :return: Log of the Jacobian and gradient
        """
        if set_hyper_params:
            self.set_hyperparameters(parameters)

        ll_grad2 = np.atleast_2d(ll_grad)
        extra_grad = np.zeros_like(ll_grad2)
        extra_grad[:, :] = np.copy(ll_grad2)
        pslc_tot = self.fnlslc
        ll_grad2_psr = ll_grad2[:, pslc_tot]

        extra_grad[:, pslc_tot] = np.dot(self.fnl_Li, ll_grad2_psr.T).T

        if 'fouriermode' in self.ptaparams.keys():

            for key, d_Phivec_d_p in self.d_Phivec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_F] = \
                        self.fnl_Beta_inv[self.Zmask_F]**2 * d_Phivec_d_p

                # dxdp for Sigma
                dxdhp = np.dot(self.fnl_Li.T, np.dot(self.fnl_dL_M[:, self.Zmask_F],
                                                     BdB[self.Zmask_F]))
                extra_grad[:, key] += np.sum(dxdhp[None, :] * \
                                             ll_grad2_psr[:, :], axis=1)

                # dxdp for mu
                WBWv = np.dot(self.fnl_Sigma[:, self.Zmask_F],
                              self.fnl_Beta_inv[self.Zmask_F]**2 *
                              d_Phivec_d_p * self.fnl_mu[self.Zmask_F])
                extra_grad[:, key] += np.sum(ll_grad2_psr * \
                                             WBWv[None, :], axis=1)

        if 'jittermode' in self.ptaparams.keys():

            for key, d_Jvec_d_p in self.d_Jvec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_U] = \
                        self.fnl_Beta_inv[self.Zmask_U]**2 * \
                        d_Jvec_d_p
                # dxdp for Sigma
                dxdhp = np.dot(self.fnl_Li.T,
                               np.dot(self.fnl_dL_M[:, self.Zmask_U],
                                      BdB[self.Zmask_U]))
                extra_grad[:, key] += np.sum(dxdhp[None, :] * ll_grad2_psr[:, :], axis=1)

                # dxdp for mu
                WBWv = np.dot(self.fnl_Sigma[:, self.Zmask_U],
                              self.fnl_Beta_inv[self.Zmask_U]**2 *
                              d_Jvec_d_p * self.fnl_mu[self.Zmask_U])
                extra_grad[:, key] += np.sum(ll_grad2_psr *
                                             WBWv[None, :], axis=1)



        return extra_grad.reshape(ll_grad.shape)


    def funnel_loglikelihood_grad(self, parameters):
        """Return the log likelihood and gradient for the funnel transformed
        coordinates
        :param parameters: Vector of signal parameters
        :return: Log likelihood and gradient
        """
        self.funnelTransform(parameters)
        basepars = self.full_backward(parameters)

        ll, ll_grad = self.base_loglikelihood_grad(basepars)
        lj, lj_grad = self.log_jacob, self.funnel_gradient

        lp = ll + lj
        lp_grad = lj_grad + self.dxdp_nondiag(parameters, ll_grad)

        return lp, lp_grad
