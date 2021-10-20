#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle

import numpy as np
import pytest

from enterprise_extensions import models
from enterprise_extensions.frequentist import optimal_statistic as optstat

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')


psr_names = ['J0613-0200', 'J1713+0747', 'J1909-3744']

with open(datadir+'/ng11yr_noise.json', 'r') as fin:
    noise_dict = json.load(fin)


@pytest.fixture
def dmx_psrs(caplog):

    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng9yr_dmx_DE436_epsr.pkl'.format(p), 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.fixture
def pta_model2a(dmx_psrs, caplog):
    m2a=models.model_2a(dmx_psrs, noisedict=noise_dict)
    return m2a


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_os(nodmx_psrs, pta_model2a):
    OS = optstat.OptimalStatistic(psrs=nodmx_psrs, pta=pta_model2a)
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
