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
from enterprise_extensions.frequentist import chi_squared as chisqr

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')


psr_names = ['J0613-0200', 'J1944+0907']

with open(datadir+'/ng11yr_noise.json', 'r') as fin:
    noise_dict = json.load(fin)


@pytest.fixture
def dmx_psrs(caplog):
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng11yr_dmx_DE436_epsr.pkl'.format(p), 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.fixture
def pta_model1(dmx_psrs, caplog):
    m2a=models.model_1(dmx_psrs, noisedict=noise_dict, tnequad=True)
    return m2a


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_chisqr(dmx_psrs, pta_model1):
    chi2 = chisqr.get_chi2(pta_model1, noise_dict)
    dof = 0
    dof += np.sum([p.toas.size for p in dmx_psrs])
    dof -= np.sum([len(p.fitpars) for p in dmx_psrs])
    dof -= len(pta_model1.param_names)
    red_chi2 = chi2/dof
    print(red_chi2)
    rchi2 = chisqr.get_reduced_chi2(pta_model1, noise_dict)
    assert rchi2 == red_chi2
    assert np.isclose(1.0, rchi2, atol=0.01)
