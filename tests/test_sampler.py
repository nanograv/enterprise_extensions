#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle

import pytest

from enterprise_extensions import models, sampler

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
def test_jumpproposal(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    jp = sampler.JumpProposal(m2a)
    assert jp.draw_from_prior.__name__ == 'draw_from_prior'
    assert jp.draw_from_signal_prior.__name__ == 'draw_from_signal_prior'
    assert (jp.draw_from_par_prior('J1713+0747').__name__ ==
            'draw_from_J1713+0747_prior')
    assert (jp.draw_from_par_log_uniform({'gw': (-20, -10)}).__name__ ==
            'draw_from_gw_log_uniform')
    assert (jp.draw_from_signal('red noise').__name__ ==
            'draw_from_red noise_signal')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_setup_sampler(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    samp = sampler.setup_sampler(m2a, outdir=outdir, human='tester')
    assert hasattr(samp, "sample")
    paramfile = os.path.join(outdir, "pars.txt")
    assert os.path.isfile(paramfile)
    with open(paramfile, "r") as f:
        params = [line.rstrip('\n') for line in f]
    for ptapar, filepar in zip(m2a.param_names, params):
        assert ptapar == filepar
