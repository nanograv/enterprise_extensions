#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains base class for computing the log likeihood and gradient for a single
pulsar, including an outlier parameter to detect outlying TOAs. Any coordinate
transformations to be applied build off of this base class.

Class methods include computing white noise vectors (N and J), red noise
vectors (Phi), outlier parameters, updating deterministic signals, and
computing the log likelihood and gradient.

This class is a near copy of the ptaLikelihood class found in piccard
(https://github.com/vhaasteren/piccard), and the methods have only been
updated as necessary to work with enterprise Pulsar objects instead of the
data structures in piccard.

Requirements:
    numpy
"""


import numpy as np

from .jitterext import cython_Uj
from .pulsar import OutlierPulsar
import enterprise_extensions.outlier.utils as ut


class ptaLikelihood(OutlierPulsar):
    """This class serves as a base class for computing the log likelihood and
    gradient for a single pulsar. It contains methods to initialize and compute
    any auxilliary quantities needed for likelihood calculation, and includes
    an additional hyperparameter, corresponding to an outlier 'signal', to
    include with the noise parameters. When sampled using HMC, NUTS, or other
    gradient-based Monte Carlo samplers, this additional parameter can be used
    to detect outlying TOAs in the dataset.

    :param enterprise_pintpulsar: `enterprise.PintPulsar` object (with drop_pintpsr=False)
    """
    def __init__(self, enterprise_pintpulsar):
        """Constructor method
        """
        super(ptaLikelihood, self).__init__(enterprise_pintpulsar)

        self.basepmin = None
        self.basepmax = None
        self.basepstart = None


        self.outlier_prob = None
        self.detresiduals = None
        self.outlier_sig_dict = dict()
        self.d_Pb_ind = None

        self.d_L_d_b = None
        self.d_Pr_d_b = None


        self.initBounds()


    def initBounds(self):
        """Set parameter vector minimum, maximum, and start values by building
        from the :class: `OutlierPulsar` signal dictionary
        """
        pmin = []
        pmax = []
        pstart = []
        for _, sig in self.signals.items():
            pmin.extend(sig['pmin'])
            pmax.extend(sig['pmax'])
            pstart.extend(sig['pstart'])

        self.basepmin = np.array(pmin)
        self.basepmax = np.array(pmax)
        self.basepstart = np.array(pstart)


    def updateParams(self, parameters):
        """Update parameter name:value dictionary with new values

        :param parameters: Vector of signal parameters
        """
        for key, value in self.ptadict.items():
            self.ptaparams[key] = parameters[value]


    def setWhiteNoise(self, calc_gradient=True):
        """Compute white noise vectors for EFAC, EQUAD, and ECORR signals, and
        optionally calculate their derivatives.

        :param calc_gradient: Include gradient calculation, default is True
        """
        self.Nvec[:] = 0
        self.Jvec[:] = 0

        ef = self.efac_sig
        eq = self.equad_sig
        ec = self.ecorr_sig

        self.Nvec[:] = ef.get_ndiag(self.ptaparams) + eq.get_ndiag(self.ptaparams)

        if ec:
            for param in ec.param_names:
                pequadsqr = 10**(2*self.ptaparams[param])
                self.Jvec += self.signals[param]['Jvec'] * pequadsqr


        if calc_gradient:
            if ef:
                for param in ef.param_names:
                    self.d_Nvec_d_param[self.ptadict[param]] = 2 * \
                                                               self.signals[param]['Nvec'] * \
                                                               self.ptaparams[param]
            if eq:
                for param in eq.param_names:
                    self.d_Nvec_d_param[self.ptadict[param]] = self.signals[param]['Nvec'] * \
                                                               2 * np.log(10) * \
                                                               10**(2*self.ptaparams[param])
            if ec:
                for param in ec.param_names:
                    self.d_Jvec_d_param[self.ptadict[param]] = self.signals[param]['Jvec'] * \
                                                               2 * np.log(10) * \
                                                               10**(2*self.ptaparams[param])


    def setPhi(self, calc_gradient=True):
        """Compute red noise Phi matrix for log10Amp and spectral index
        parameters, and optionally calculate their derivatives.

        :param calc_gradient: Include gradient calculation, default is True
        """
        self.Phivec[:] = 0

        rn = self.rn_sig
        log10A = self.ptaparams[self.pname + '_rn_log10_A']
        gamma = self.ptaparams[self.pname + '_rn_gamma']
        sTmax = self.psr.toas.max() - self.psr.toas.min()

        self.Phivec[:] = rn.get_phi(self.ptaparams)

        if calc_gradient:
            d_mat = ut.d_powerlaw(log10A, gamma, sTmax, self.Ffreqs)
            for key, _ in self.ptaparams.items():
                if key.endswith('log10_A'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:, 0]
                elif key.endswith('gamma'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:, 1]


    def setOutliers(self):
        """Set outlier probability parameter and its corresponding index in the
        parameter vector.
        """
        for key, param in self.ptaparams.items():
            if key.endswith('outlierprob'):
                self.outlier_prob = param
                self.d_Pb_ind = [self.ptadict[key]]


    def setDetSources(self, parameters, calc_gradient=True):
        """Update the deterministic signals given a parameter vector, and
        optionally calculate their derivatives.

        :param parameters: Vector of signal parameters
        :param calc_gradient: Include gradient calculation, default is True
        """
        d_L_d_b = np.zeros_like(parameters)
        d_Pr_d_b = np.zeros_like(parameters)
        self.outlier_sig_dict = dict()

        self.detresiduals = self.psr.residuals.copy()

        for _, sig in self.signals.items():
            sparams = parameters[sig['msk']]

            if sig['type'] == 'bwm':
                pass
            elif sig['type'] == 'timingmodel':
                self.detresiduals -= np.dot(self.Mmat_g, sparams)
            elif sig['type'] == 'fouriermode':
                self.detresiduals -= np.dot(self.Fmat, sparams)
            elif sig['type'] == 'jittermode':
                self.detresiduals -= cython_Uj(sparams, self.Uindslc, len(self.detresiduals))

        if calc_gradient:
            pulsarind = 0
            if pulsarind not in self.outlier_sig_dict:
                self.outlier_sig_dict[pulsarind] = []
            for _, sig in self.signals.items():
                parslice = sig['msk']
                sparams = parameters[parslice]

                if sig['type'] == 'bwm':
                    pass
                elif sig['type'] == 'timingmodel':
                    d_L_d_xi = np.zeros(self.Mmat_g.shape[1])

                    d_L_d_b_o = self.Mmat_g.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))

                    d_L_d_b[parslice] = d_L_d_xi
                elif sig['type'] == 'fouriermode':
                    d_L_d_xi = np.zeros(self.Fmat.shape[1])
                    phivec = self.Phivec.copy()

                    d_L_d_b_o = self.Fmat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))

                    d_Pr_d_xi = -sparams / phivec
                    d_L_d_b[parslice] = d_L_d_xi
                    d_Pr_d_b[parslice] = d_Pr_d_xi
                elif sig['type'] == 'jittermode':
                    d_L_d_xi = np.zeros(self.Umat.shape[1])

                    d_L_d_b_o = self.Umat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))

                    d_Pr_d_xi = -sparams / self.Jvec
                    d_L_d_b[parslice] = d_L_d_xi
                    d_Pr_d_b[parslice] = d_Pr_d_xi

        self.d_L_d_b = d_L_d_b
        self.d_Pr_d_b = d_Pr_d_b


    def set_hyperparameters(self, parameters, calc_gradient=True):
        """Wrapper function to update all hyperparameters (white/red noise,
        outliers) and their associated vector and matrix quantities.

        :param parameters: Vector of signal parameters
        :param calc_gradient: Include gradient calculation, default is True
        """
        self.updateParams(parameters)

        self.setPhi(calc_gradient=calc_gradient)
        self.setWhiteNoise(calc_gradient=calc_gradient)
        self.setOutliers()


    def base_loglikelihood_grad(self, parameters, set_hyper_params=True, calc_gradient=True):
        """Return the log likelihood and gradient for the original,
        non-transformed coordinates.

        :param parameters: Vector of signal parameters
        :param set_hyper_params: Update hyperparameter values before computing
            log likelihood, default is True
        :param calc_gradient: Include gradient calculation, default is True
        :return: Log likelihood and gradient
        """
        if set_hyper_params:
            self.set_hyperparameters(parameters, calc_gradient=calc_gradient)
            self.setDetSources(parameters, calc_gradient=calc_gradient)

        d_L_d_b, d_Pr_d_b = self.d_L_d_b, self.d_Pr_d_b
        gradient = np.zeros_like(d_L_d_b)

        bBb = np.zeros_like(0, dtype=float)
        ldB = np.zeros_like(0, dtype=float)
        logl_outlier = np.zeros_like(0, dtype=float)

        P0 = self.P0
        Pb = self.outlier_prob

        lln = self.detresiduals**2 / self.Nvec
        lld = np.log(self.Nvec) + np.log(2*np.pi)
        logL0 = -0.5*lln -0.5*lld
        bigL0 = (1. - Pb) * np.exp(logL0)
        bigL = bigL0 + Pb/P0
        logl_outlier += np.sum(np.log(bigL))

        for pslc, d_L_d_b_o in self.outlier_sig_dict[0]:
            gradient[pslc] += np.sum(d_L_d_b_o * bigL0[None, :]/bigL[None, :], axis=1)

        for pbind in self.d_Pb_ind:
            gradient[pbind] += np.sum((-np.exp(logL0)+1.0/P0)/bigL)

        for key, d_Nvec_d_p in self.d_Nvec_d_param.items():
            d_L_d_b_o = 0.5*(self.detresiduals**2 * d_Nvec_d_p / \
                             self.Nvec**2 - d_Nvec_d_p / self.Nvec)
            gradient[key] += np.sum(d_L_d_b_o * bigL0/bigL)

        if 'fouriermode' in self.ptaparams.keys():
            pslc = self.signals['fouriermode']['msk']

            bsqr = parameters[pslc]**2
            phivec = self.Phivec # + Svec[fslc]

            bBb += np.sum(bsqr / phivec)
            ldB += np.sum(np.log(phivec))

            gradient[pslc] += d_Pr_d_b[pslc]

            for key, d_Phivec_d_p in self.d_Phivec_d_param.items():
                gradient[key] += 0.5 * np.sum(bsqr * d_Phivec_d_p / phivec**2)
                gradient[key] -= 0.5 * np.sum(d_Phivec_d_p / phivec)

        if 'dmfouriermode' in self.ptaparams.keys():
            pass

        if 'jittermode' in self.ptaparams.keys():
            pslc = self.signals['jittermode']['msk']

            bsqr = parameters[pslc]**2
            jvec = self.Jvec

            bBb += np.sum(bsqr / jvec)
            ldB += np.sum(np.log(jvec))

            gradient[pslc] += d_Pr_d_b[pslc]

            for key, d_Jvec_d_p in self.d_Jvec_d_param.items():
                gradient[key] += 0.5 * np.sum(bsqr * d_Jvec_d_p / jvec**2)
                gradient[key] -= 0.5 * np.sum(d_Jvec_d_p / jvec)

        ll = np.sum(logl_outlier) - 0.5*np.sum(bBb) - 0.5*np.sum(ldB)

        return ll, gradient
