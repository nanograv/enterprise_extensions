#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import pytest
import pickle, json, os
import logging
from enterprise_extensions import models, sampler

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')
outdir = os.path.join(testdir, 'test_out')

psr_names = ['J0613-0200','J1713+0747','J1909-3744']

with open(datadir+'/ng11yr_noise.json','r') as fin:
    noise_dict = json.load(fin)

@pytest.fixture
def dmx_psrs(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng9yr_dmx_DE436_epsr.pkl'.format(p),'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_setup_sampler(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    samp = sampler.setup_sampler(m2a, outdir=outdir, human='tester')
    assert hasattr(samp, "sample")
    assert os.path.isfile("tmp/pars.txt")
    with open("tmp/pars.txt", "r") as f:
        params = [line.rstrip('\n') for line in f]
    assert m2a.param_names == params

