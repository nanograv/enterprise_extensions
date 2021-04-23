#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains classes defining both interval coordinate transformations and
whitening coordinate transformations. The interval transformation builds
directly from the funnel transformation in `funnel.py`, whereas the whitening
transformation can be performed independent of other transformation.

Requirements:
    numpy
    scipy.linalg
"""


import numpy as np
import scipy.linalg as sl

from .funnel import Funnel


class Interval(Funnel):
    """This class implements an interval coordinate transformation for model
    hyperparameters. It inherits from the :class: `funnel.Funnel`, with the
    interval being defined by the min and max parameter vectors under the
    funnel transformation.

    As it stands, this class must necessarily be built on top of a funnel
    transform object, and cannot skip that step. Work may be done in the
    future to allow this transformation to be done independent of other
    coordinate transforamtions in this package.

    :param enterprise_pintpulsar: `enterprise.PintPulsar` object (with drop_pintpsr=False)
    :param pstart: Starting parameter vector under interval transformation. It
        is the eventual first input to the NUTS sampler
    :param msk: Vector of booleans for identifying hyperparameters
    :param a: Minimum parameter vector for interval transform
    :param b: Maximum parameter vector for interval transform
    """
    def __init__(self, enterprise_pintpulsar):
        """Constructor method
        """
        super(Interval, self).__init__(enterprise_pintpulsar)

        self.pstart = None

        self.msk = None

        self.a = None
        self.b = None

        self.init_interval_model()


    def init_interval_model(self):
        """Run initiliazation functions for interval transformation object
        """
        self.hypMask()
        self.initIntervalBounds()


    def initIntervalBounds(self):
        """Set min and max bounds of interval transformation and forward
        transform the starting parameter vector
        """
        self.a = self.funnelmin
        self.b = self.funnelmax
        self.pstart = self.forward(self.funnelstart)


    def hypMask(self):
        """Create boolean mask vector. Entries are True for corresponding
        hyperparameter indices, False otherwise
        """
        lowlevelpars = ['timingmodel', 'fouriermode', 'jittermode']
        msk = [True] * len(self.basepstart)
        for _, sig in self.signals.items():
            if sig['type'] in lowlevelpars:
                msk[sig['msk']] = [False] * sig['numpars']

        self.msk = np.array(msk)

    def forward(self, x):
        """Apply interval transformation to parameter vector

        :param x: Parameter vector to be transformed
        :return: Parameter vector with interval transform
        """
        p = np.atleast_2d(x.copy())
        posinf, neginf = (self.a == x), (self.b == x)
        m = self.msk & ~(posinf | neginf)
        p[:, m] = np.log((p[:, m] - self.a[m]) / (self.b[m] - p[:, m]))
        p[:, posinf] = np.inf
        p[:, neginf] = -np.inf
        return p.reshape(x.shape)


    def backward(self, p):
        """Undo the interval transformation

        :param p: Parameter vector under interval transform
        :return: Parameter vector in original coordinates
        """
        x = np.atleast_2d(p.copy())
        m = self.msk
        x[:, m] = (self.b[m] - self.a[m]) * np.exp(x[:, m]) \
                   / (1 + np.exp(x[:, m])) + self.a[m]
        return x.reshape(p.shape)


    def dxdp(self, p):
        """Return Jacobian of interval transformation (derivative of x wrt p)

        :param p: Vector of signal parameters
        :return: Jacobian of transformation evaluted at given signal
            parameters
        """
        pp = np.atleast_2d(p)
        m = self.msk
        d = np.ones_like(pp)
        d[:, m] = (self.b[m]-self.a[m])*np.exp(pp[:, m])/(1+np.exp(pp[:, m]))**2
        return d.reshape(p.shape)


    def logjacobian_grad(self, p):
        """Return the log of the Jacobian with gradient evaluted at the given
        signal parameters

        :param p: Vector of signal parameters
        :return: Log of the Jacobian and gradient
        """
        m = self.msk
        lj = np.sum(np.log(self.b[m]-self.a[m]) + p[m] -
                    2*np.log(1.0+np.exp(p[m])))

        lj_grad = np.zeros_like(p)
        lj_grad[m] = (1 - np.exp(p[m])) / (1 + np.exp(p[m]))
        return lj, lj_grad


    def full_loglikelihood_grad(self, parameters):
        """Return the log likelihood and gradient for the interval transformed
        coordinates
        :param parameters: Vector of signal parameters
        :return: Log likelihood and gradient
        """
        funnelpars = self.backward(parameters)
        ll, ll_grad = self.funnel_loglikelihood_grad(funnelpars)
        lj, lj_grad = self.logjacobian_grad(parameters)

        lp = ll + lj
        lp_grad = ll_grad * self.dxdp(parameters) + lj_grad

        return lp, lp_grad


class whitenedLikelihood():
    """This class implements an coordinate transformation that whitenes the
    parameter space. The coordinate basis is changed to one whose covariance
    is the identity matrix.

    :param likob: Likelihood object to whiten
    :param parameters: Approximate maximum of posterior
    :param hessian: Hessian of the parameter space
    :param ch: Cholesky decomposition of the Hessian
    :param chi: Solution x to Ax = I, where A is the triangular matrix found
        from `ch`, and I is the identity matrix
    :param lj: Log jacobian of the whitening transformation
    """
    def __init__(self, likob, parameters, hessian):
        """Constructor method
        """
        self.likob = likob
        self.mu = parameters.copy()

        self.calc_invsqrt(hessian)

    def calc_invsqrt(self, hessian):
        """Calculate the inverse square root, given the Hessian matrix

        :param hessian: Hessian matrix of the parameter space
        """
        try:
            # Try Cholesky
            self.ch = sl.cholesky(hessian, lower=True)

            # Fast solve
            self.chi = sl.solve_triangular(self.ch, np.eye(len(self.ch)),
                                           trans=0, lower=True)
            self.lj = np.sum(np.log(np.diag(self.chi)))
        except sl.LinAlgError:
            # Cholesky fails. Try eigh
            try:
                eigval, eigvec = sl.eigh(hessian)

                if not np.all(eigval > 0):
                    # Try SVD here? Or just regularize?
                    raise sl.LinAlgError("Eigh thinks hessian is not positive definite")

                self.ch = eigvec * np.sqrt(eigval)
                self.chi = (eigvec / np.sqrt(eigval)).T
                self.lj = -0.5*np.sum(np.log(eigval))
            except sl.LinAlgError:
                U, s, Vt = sl.svd(hessian)

                if not np.all(s > 0):
                    raise sl.LinAlgError("SVD thinks hessian is not positive definite")

                self.ch = U * np.sqrt(s)
                self.chi = (U / np.sqrt(s)).T
                self.lj = -0.5*np.sum(np.log(s))


    def forward(self, x):
        """Forward transform the parameter vector to whitened coordinates

        :param x: Parameter vector to be transformed
        :return: Whitened parameter vector
        """
        p = np.atleast_2d(x.copy()) - self.mu
        p = np.dot(self.ch.T, p.T).T
        return p.reshape(x.shape)


    def backward(self, p):
        """Backward transform whitened parameters to original coordinates

        :param p: Whitened parameter vector
        :return: Parameter vector in original coordinates
        """
        x = np.atleast_2d(p.copy())
        x = np.dot(self.chi.T, x.T).T + self.mu
        return x.reshape(p.shape)


    def logjacobian_grad(self, parameters):
        """Return the log of the Jacobian with gradient evaluted at the given
        signal parameters

        :param parameters: Vector of signal parameters
        :return: Log of the Jacobian and gradient
        """
        lj = self.lj
        lj_grad = np.zeros_like(parameters)
        return lj, lj_grad


    def loglikelihood_grad(self, parameters):
        """Return the log likelihood and gradient in the whitened coordinates

        :param parameters: Vector of signal parameters
        :return: Log likelihood and gradient
        """
        x = self.backward(parameters)
        lp, lp_grad = self.likob.full_loglikelihood_grad(x)
        lj, lj_grad = self.logjacobian_grad(parameters)
        grad = np.dot(self.chi, lp_grad) + lj_grad

        return lp + lj, grad
