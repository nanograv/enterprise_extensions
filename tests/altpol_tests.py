#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for altpol functions in e_e Code.
"""

import json
import logging
import os
import pickle

import enterprise.signals.parameter as parameter
import numpy as np
import pytest
from enterprise.signals import gp_signals, signal_base

from enterprise_extensions import model_orfs, models
from enterprise_extensions.frequentist import optimal_statistic as optstat

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')


psr_names = ['J0613-0200', 'J1713+0747', 'J1909-3744']

with open(datadir+'/ng11yr_noise.json', 'r') as fin:
    noise_dict = json.load(fin)


@pytest.fixture
def nodmx_psrs(caplog):
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng9yr_nodmx_DE436_epsr.pkl'.format(p), 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


def test_model_general_alt_correlations(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m=models.model_general(nodmx_psrs, noisedict=noise_dict,
                           orf='hd,gw_monopole,gw_dipole,st,gt,dipole,monopole')
    assert hasattr(m, 'get_lnlikelihood')


def test_model_2a_altpol_spectrum(nodmx_psrs, caplog):

    log10_A_tt = parameter.LinearExp(-18, -12)('log10_A_tt')
    log10_A_st = parameter.LinearExp(-18, -12)('log10_A_st')
    log10_A_vl = parameter.LinearExp(-18, -15)('log10_A_vl')
    log10_A_sl = parameter.LinearExp(-18, -16)('log10_A_sl')
    kappa = parameter.Uniform(0, 15)('kappa')
    p_dist = parameter.Normal(1.0, 0.2)
    pl = model_orfs.generalized_gwpol_psd(log10_A_tt=log10_A_tt, log10_A_st=log10_A_st,
                                          log10_A_vl=log10_A_vl, log10_A_sl=log10_A_sl,
                                          kappa=kappa, p_dist=p_dist, alpha_tt=-2/3, alpha_alt=-1)

    s = models.white_noise_block(vary=False, inc_ecorr=True)
    s += models.red_noise_block(prior='log-uniform')
    s += gp_signals.FourierBasisGP(spectrum=pl, name='gw')
    s += gp_signals.TimingModel()

    m = signal_base.PTA([s(psr) for psr in nodmx_psrs])
    m.set_default_params(noise_dict)
    for param in m.params:
        if 'gw_p_dist' in str(param):
            # get pulsar name and distance
            psr_name = str(param).split('_')[0].strip('"')
            psr_dist = [p._pdist for p in nodmx_psrs if psr_name in p.name][0]

            # edit prior settings
            param._prior = parameter.Normal(mu=psr_dist[0],
                                            sigma=psr_dist[1])
            param._mu = psr_dist[0]
            param._sigma = psr_dist[1]

    assert hasattr(m, 'get_lnlikelihood')


"""
Tests for altpol functions in OS Code.
"""


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.fixture
def pta_model2a(nodmx_psrs, caplog):
    m2a=models.model_2a(nodmx_psrs, noisedict=noise_dict)
    return m2a


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_os(nodmx_psrs, pta_model2a):
    orfs = ['hd', 'gw_monopole', 'gw_dipole', 'st', 'dipole', 'monopole']
    for orf in orfs:
        OS = optstat.OptimalStatistic(psrs=nodmx_psrs, pta=pta_model2a, orf=orf)
        assert hasattr(OS, 'Fmats')
        OS.compute_os()
        chain = np.zeros((10, len(pta_model2a.params)+4))
        for ii in range(10):
            entry = [par.sample() for par in pta_model2a.params]
            entry.extend([OS.pta.get_lnlikelihood(entry)-OS.pta.get_lnprior(entry),
                          OS.pta.get_lnlikelihood(entry),
                          0.5, 1])
            chain[ii, :] = np.array(entry)
        OS.compute_noise_marginalized_os(chain, param_names=OS.pta.param_names, N=10)
        OS.compute_noise_maximized_os(chain, param_names=OS.pta.param_names)
