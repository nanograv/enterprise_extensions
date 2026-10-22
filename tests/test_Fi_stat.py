#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package 'Fi-statistic.'"""

import json
import logging
import os

import numpy as np
import pytest

from enterprise_extensions import models
from enterprise_extensions.frequentist import Fi_statistic as Fi_stat
from enterprise.pulsar import FeatherPulsar

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
        psrs.append(FeatherPulsar.read_feather(datadir+'/{0}_ng9yr_dmx_DE436_epsr.feather'.format(p)))

    return psrs


@pytest.fixture
def nodmx_psrs(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        psrs.append(FeatherPulsar.read_feather(datadir+'/{0}_ng9yr_nodmx_DE436_epsr.feather'.format(p)))

    return psrs


@pytest.fixture
def pta_model2a(dmx_psrs, caplog):
    # filterwarnings must not decorate fixtures (pytest>=8 hard-fails).
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict, tnequad=True)
    return m2a


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_Fi(nodmx_psrs, pta_model2a):
    pmaxes = [p.toas.max() for p in nodmx_psrs]
    pmins = [p.toas.min() for p in nodmx_psrs]
    maxMJD = max(pmaxes)/3600/24
    minMJD = min(pmins)/3600/24
    t0_mid = minMJD + (maxMJD - minMJD)/2

    Fi_obj = Fi_stat.FiStat(psrs=nodmx_psrs, noisedict=noise_dict, pta=pta_model2a)
    Fi_obj.compute_Fi(t0=t0_mid)

    chain = np.zeros((10, len(pta_model2a.params)+4))
    for ii in range(10):
        entry = [par.sample() for par in pta_model2a.params]
        entry.extend([Fi_obj.pta.get_lnlikelihood(entry)-Fi_obj.pta.get_lnprior(entry),
                      Fi_obj.pta.get_lnlikelihood(entry),
                      0.5, 1])
        chain[ii, :] = np.array(entry)
    Fi_obj.compute_maximum_likelihood_Fi(epoch=t0_mid, noisedict=noise_dict, chain=chain)

    Nts = 20
    t0_range = np.linspace(minMJD, maxMJD, Nts)
    Fi_obj.compute_noise_marginalized_Fi(epochs=t0_range, noisedict=noise_dict, chain=chain, Nitr=10)
