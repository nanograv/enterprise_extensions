#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle

import pytest

from enterprise_extensions import models, hypermodel

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')
outdir = os.path.join(testdir, 'test_out')

psr_names = ['J0613-0200', 'J1713+0747', 'J1909-3744']

with open(datadir+'/ng11yr_noise.json', 'r') as fin:
    noise_dict = json.load(fin)


@pytest.fixture
def dmx_psrs(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng9yr_dmx_DE436_epsr.pkl'.format(p), 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_hypermodel(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    m3a = models.model_3a(dmx_psrs, noisedict=noise_dict)
    ptas = {0: m2a, 1: m3a}
    hm = hypermodel.HyperModel(ptas)
    assert hasattr(hm, 'get_lnlikelihood')
    assert 'gw_log10_A' in hm.param_names
    assert 'nmodel' in hm.param_names


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_hyper_sampler(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    m3a = models.model_3a(dmx_psrs, noisedict=noise_dict)
    ptas = {0: m2a, 1: m3a}
    hm = hypermodel.HyperModel(ptas)
    samp = hm.setup_sampler(outdir=outdir, human='tester')
    assert hasattr(samp, "sample")
    paramfile = os.path.join(outdir, "pars.txt")
    assert os.path.isfile(paramfile)
    with open(paramfile, "r") as f:
        params = [line.rstrip('\n') for line in f]
    for ptapar, filepar in zip(hm.param_names, params):
        assert ptapar == filepar
    x0 = hm.initial_sample()
    assert len(x0) == len(hm.param_names)
