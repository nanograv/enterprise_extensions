#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base pulsar class that contains pulsar timing information, white and red noise
vectors and bases, outlier parameter, and signal/parameter dictionaries.

Class methods include adding individual signals to the pulsar model, setting
auxilliary quantities for use in likelihood calculations, orthogonalizing the
pulsar design matrix, and loading both a libstempo and enterprise pulsa
object.

Functions such as those that add signals to the model, and especially the
`orthogonal_designmatrix` function are based almost entirely off of similar
functions found in piccard (https://github.com/vhaasteren/piccard), and have
only been adapted to cooperate with enterprise Pulsar objects.

Requirements:
    collections
    numpy
    scipy.linalg
    libstempo
    enterprise
"""

import sys
from collections import OrderedDict
import numpy as np
import scipy.linalg as sl

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import selections, white_signals, gp_signals, utils
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix, quant2ind

import pint.residuals as resid
import pint.toa as toa
import pint.models as model
import astropy.units as u

import enterprise_extensions.outlier.utils as ut


class OutlierPulsar():
    """This class builds off of (but does not directly inherit from) the PintPulsar
    class in enterprise. :class: `OutlierPulsar` takes a PintPulsar object as
    input, and can be referenced with the .psr
    attribute. Information already contained in the PintPulsar object, such as
    TOAs, residuals, design matrix, etc., can be accessed as normal with
    commands such as .psr.Mmat, .psr.toas, etc.

    What this class does is add additional attributes for white/red noise
    vectors, basis matrices, derivatives, and signal dictionaries for use in
    single-pulsar likelihood and gradient calculations.

    Included in this class is the choice to add an outlier parameter alongside
    existing noise parameters, which can then be used to perform single-pulsar
    outlier-analysis.

    :param enterprise_pintpulsar: `enterprise.PintPulsar` object (with drop_pintpsr=False)
    :param selection: enterprise Selection object for separating TOAs into
        groups, default is by_backend
    :param nfreqcomps: Number of Fourier modes for red noise signal, default
        is 30
    """
    def __init__(self, enterprise_pintpulsar, 
                 selection=selections.Selection(selections.by_backend),
                 nfreqcomps=30):
        """Constructor method
        """
        #self.parfile = parfile
        #self.timfile = timfile
        self.selection = selection

        # Initialize enterprise PintPulsar object
        #self.ltpsr = None
        self.ephem = None
        self.F0 = None
        self.P0 = None

        self.psr = None
        self.pname = None

        # Initialize white noise vectors and gradients
        self.Nvec = None
        self.Jvec = None
        self.d_Nvec_d_param = dict()
        self.d_Jvec_d_param = dict()

        # Initialize red noise and jitter mode vectors and gradients
        self.Phivec = None
        self.d_Phivec_d_param = dict()

        # Orthogonalized design matrix
        self.Mmat_g = None

        # Initialize Fourier design matrix
        self.nfreqcomps = nfreqcomps
        self.Fmat = None
        self.Ffreqs = None

        # Initialize ECORR exploder matrix
        self.Umat = None
        self.Uindslc = None

        # Initialize Zmat
        self.Zmat = None

        # Enterprise signals
        self.efac_sig = None
        self.equad_sig = None
        self.ecorr_sig = None
        self.rn_sig = None
        self.tm_sig = None

        # Signal and parameter dictionaries
        self.signals = dict()
        self.ptadict = dict()
        self.ptaparams = dict()

        self.init_hierarchical_model(enterprise_pintpulsar)

    def init_hierarchical_model(self, PintPulsar):
        """Run initialization functions for loading :class: `OutlierPulsar`
        object, and set auxilliary quantities for likelihood calculation.

        :param PintPulsar: `enterprise.PintPulsar` object
        """
        # Define self.psr...
        self.load_ent_pintpsr(PintPulsar)

        self.orthogonal_designmatrix()
        self.set_Fmat_auxiliaries()
        self.set_Umat_auxiliaries()

        self.Nvec = np.zeros(len(self.psr.toas))
        self.Jvec = np.zeros(self.Umat.shape[1])
        self.Phivec = np.zeros(2 * self.nfreqcomps)

        self.loadSignals()
        self.setZmat()


    def orthogonal_designmatrix(self, lowlevelparams=['rednoise', 'design', 'jitter']):
        """Orthogonalize the pulsar design matrix (Mmat). Design matrix is
        othogonalized (using SVD) in groups, corresponding to different sets
        of timing model parameters. Each set is dealt with separately, and
        then merged together to create the final design matrix for use in
        likelihood calcualtions (Mmat_g)

        :param lowlevelparams: List of low level (non-hyperparameter) paramter
            groups, default is ['rednoise', 'design', 'jitter']
        """
        F_list = ['Offset', \
                'LAMBDA', 'BETA', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
                'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'TASC', 'EPS1', 'EPS2', \
                'XDOT', 'PBDOT', 'KOM', 'KIN', 'EDOT', 'MTOT', 'SHAPMAX', \
                'GAMMA', 'X2DOT', 'XPBDOT', 'E2DOT', 'OM_1', 'A1_1', 'XOMDOT', \
                'PMLAMBDA', 'PMBETA', 'PX', 'PB', 'A1', 'E', 'ECC', \
                'T0', 'OM', 'OMDOT', 'SINI', 'A1', 'M2']
        F_front_list = ['JUMP', 'F']
        D_list = ['DM', 'DM1', 'DM2', 'DM3', 'DM4']

        Mmask_F = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmask_D = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmask_U = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmat_g = np.zeros(self.psr.Mmat.shape)
        for ii, par in enumerate(self.psr.fitpars):
            incrn = False
            for par_front in F_front_list:
                if par[:len(par_front)] == par_front:
                    incrn = True

            if (par in F_list or incrn) and 'rednoise' in lowlevelparams:
                Mmask_F[ii] = True

            if par in D_list and 'dm' in lowlevelparams:
                Mmask_D[ii] = True

        Mmask_M = np.array([1]*Mmat_g.shape[1], dtype=np.bool)
        if 'rednoise' in lowlevelparams:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_F))
        if 'dm' in lowlevelparams:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_D))
        if 'jitter' in lowlevelparams:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_U))

        # Create orthogonals for all of these
        if np.sum(Mmask_F) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_F], full_matrices=False)
            Mmat_g[:, Mmask_F] = U

        if np.sum(Mmask_D) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_D], full_matrices=False)
            Mmat_g[:, Mmask_D] = U

        if np.sum(Mmask_U) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_U], full_matrices=False)
            Mmat_g[:, Mmask_U] = U

        if np.sum(Mmask_M) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_M], full_matrices=False)
            Mmat_g[:, Mmask_M] = U

        self.Mmat_g = Mmat_g


    def set_Umat_auxiliaries(self):
        """Set quantization matrix and array of observing epoch start/stop
        indices for TOAs after filtering out short (<4 TOAs) epochs
        """
        Umat, _ = create_quantization_matrix(self.psr.toas)
        Uind = quant2ind(Umat)

        self.Umat, self.Uindslc = ut.set_Uindslc(Umat, Uind)


    def set_Fmat_auxiliaries(self):
        """Set Fourier design matrix and array of corresponding Fourier mode
        frequencies
        """
        Fmat, self.Ffreqs = createfourierdesignmatrix_red(self.psr.toas)
        self.Fmat = np.zeros_like(Fmat)
        self.Fmat[:, 1::2] = Fmat[:, ::2]
        self.Fmat[:, ::2] = Fmat[:, 1::2]


    def setZmat(self):
        """Set Z matrix, which is the combination of basis matrices for timing
        model, fourier mode, and jitter mode parameters
        """
        if 'timingmodel' in self.ptaparams.keys():
            Zmat = self.Mmat_g.copy()
        if 'fouriermode' in self.ptaparams.keys():
            Zmat = np.append(Zmat, self.Fmat, axis=1)
        if 'jittermode' in self.ptaparams.keys():
            Zmat = np.append(Zmat, self.Umat, axis=1)

        self.Zmat = Zmat


    def load_ent_pintpsr(self,epp):
        """Load enterprise PintPulsar object, then re-sort based on available flags.

        Parameters
        ==========
        epp: `enterprise.PintPulsar` object
    
        """
        # Most NANOGrav releases (except 5 yr; -be) include -f flags for all TOAs.
        try:
            flags = epp.flags['f']
        except:
            flags = epp.flags['be']
   
        isort = ut.argsortTOAs(epp._toas, flags)
        self.ephem = epp.model.EPHEM.value
        self.F0 = epp.model.F0.value
        self.P0 = 1.0 / self.F0
        epp.to_pickle() # write pickle object and delete pint_toas/model.
        self.pname = epp.name
        epp._isort = isort
        self.psr = epp


    def loadSignals(self, incEfac=True, incEquad=True, incEcorr=True,
                    incRn=True, incOut=True, incTiming=True, incFourier=True,
                    incJitter=True):
        """Generate dictionary of signal information and dictionary of signal
        parameter names and their values. The parameter names will match
        exactly to enterprise signal `param_names` attributes.

        :param incEfac: Add EFAC signal to model, default is True
        :param incEquad: Add EQUAD signal to model, default is True
        :param incEcorr: Add ECORR signal to model, default is True
        :param incRn: Add red noise signal to model, default is True
        :param incOut: Add outlier parameter signal to model, default is True
        :param incTiming: Add timing model signal to model, default is True
        :param incFourier: Add fourier mode signals to model, default is True
        :param incJitter: Add jitter mode signals to model, default is True
        """
        if incEfac:
            self.signals.update(self.add_efac())
        if incEquad:
            self.signals.update(self.add_equad())
        if incEcorr:
            self.signals.update(self.add_ecorr())
        if incRn:
            self.signals.update(self.add_rn())
        if incOut:
            self.signals.update(self.add_outlier())
        if incTiming:
            self.signals.update(self.add_timingmodel())
        if incFourier:
            self.signals.update(self.add_fourier())
        if incJitter:
            self.signals.update(self.add_jitter())

        index = 0
        for key, sig in self.signals.items():
            sig['pmin'] = np.array(sig['pmin'])
            sig['pmax'] = np.array(sig['pmax'])
            sig['pstart'] = np.array(sig['pstart'])
            sig['index'] = index
            sig['msk'] = slice(sig['index'], sig['index']+sig['numpars'])
            index += sig['numpars']

        for key, sig in self.signals.items():
            if sig['type'] == 'rn':
                self.ptaparams.update(dict(zip(sig['name'], sig['pstart'])))
            else:
                self.ptaparams[sig['name']] = sig['pstart'][0]

        for ii, key in enumerate(self.ptaparams.keys()):
            if key not in ['timingmodel', 'fouriermode', 'jittermode']:
                self.ptadict[key] = ii


    def add_efac(self):
        """Return dictionary containing EFAC white noise signal
        attributes

        :return: OrderedDict of EFAC signal
        """
        efac_dct = dict()
        efac = parameter.Uniform(0.001, 5.0)
        ef = white_signals.MeasurementNoise(efac=efac, selection=self.selection)
        self.efac_sig = ef(self.psr)
        for ii, param in enumerate(self.efac_sig.param_names):
            Nvec = self.psr.toaerrs**2 * self.efac_sig._masks[ii]
            newsignal = OrderedDict({'type': 'efac',
                                     'name': param,
                                     'pmin': [0.001],
                                     'pmax': [5.0],
                                     'pstart': [1.0],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Nvec': Nvec})
            efac_dct.update({param : newsignal})

        return efac_dct


    def add_equad(self):
        """Return dictionary containing EQUAD white noise signal
        attributes

        :return: OrderedDict of EQUAD signal
        """
        equad_dct = dict()
        equad = parameter.Uniform(-10.0, -4.0)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=self.selection)
        self.equad_sig = eq(self.psr)
        for ii, param in enumerate(self.equad_sig.param_names):
            Nvec = np.ones_like(self.psr.toaerrs) * self.equad_sig._masks[ii]
            newsignal = OrderedDict({'type': 'equad',
                                     'name': param,
                                     'pmin': [-10.0],
                                     'pmax': [-4.0],
                                     'pstart': [-6.5],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Nvec': Nvec})
            equad_dct.update({param : newsignal})

        return equad_dct


    def add_ecorr(self):
        """Return dictionary containing correlated white noise (ECORR) signal
        attributes

        :return: OrderedDict of ECORR signal
        """
        ecorr_dct = dict()
        ecorr = parameter.Uniform(-10.0, -4.0)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=self.selection)
        self.ecorr_sig = ec(self.psr)
        for ii, param in enumerate(self.ecorr_sig.param_names):
            Nvec = np.ones_like(self.psr.toaerrs) * self.ecorr_sig._masks[ii]
            Jvec = np.array(np.sum(Nvec * self.Umat.T, axis=1) > 0.0, dtype=np.double)
            newsignal = OrderedDict({'type': 'ecorr',
                                     'name': param,
                                     'pmin': [-10.0],
                                     'pmax': [-4.0],
                                     'pstart': [-6.5],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Jvec': Jvec})
            ecorr_dct.update({param : newsignal})

        return ecorr_dct


    def add_rn(self):
        """Return dictionary containing red noise power-law signal attributes

        :return: OrderedDict of red noise signal
        """
        log10_A = parameter.Uniform(-20.0, -10.0)
        gamma = parameter.Uniform(0.02, 6.98)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl, components=self.nfreqcomps, name='rn')
        newsignal = OrderedDict({'type': 'rn',
                                 'name': [self.pname + '_rn_log10_A', self.pname + '_rn_gamma'],
                                 'pmin': [-20.0, 0.02],
                                 'pmax': [-10.0, 6.98],
                                 'pstart': [-14.5, 3.51],
                                 'interval': [True, True],
                                 'numpars': 2})
        self.rn_sig = rn(self.psr)
        return {'rn': newsignal}


    def add_timingmodel(self):
        """Return dictionary containing timing model signal attributes

        :return: OrderedDict of timing model signal
        """
        tm = gp_signals.TimingModel(use_svd=False)
        npars = self.psr.Mmat.shape[1]
        newsignal = OrderedDict({'type': 'timingmodel',
                                 'name': 'timingmodel',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-10]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})
        self.tm_sig = tm(self.psr)
        return {'timingmodel': newsignal}


    def add_fourier(self):
        """Return dictionary containing fourier modes signal attributes

        :return: OrderedDict of fourier modes signal
        """
        npars = 2 * self.nfreqcomps
        newsignal = OrderedDict({'type': 'fouriermode',
                                 'name': 'fouriermode',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-9]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})

        return {'fouriermode': newsignal}


    def add_jitter(self):
        """Return dictionary containing jitter modes signal attributes

        :return: OrderedDict of jitter modes signal
        """
        npars = len(self.Jvec)
        newsignal = OrderedDict({'type': 'jittermode',
                                 'name': 'jittermode',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-9]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})

        return {'jittermode': newsignal}


    def add_outlier(self):
        """Return dictionary containing outlier signal attributes

        :return: OrderedDict of outlier signal
        """
        newsignal = OrderedDict({'type': 'outlier',
                                 'name': self.pname + '_outlierprob',
                                 'pmin': [0.0],
                                 'pmax': [1.0],
                                 'pstart': [0.001],
                                 'interval': [True],
                                 'numpars': 1})
        return {'outlier': newsignal}
