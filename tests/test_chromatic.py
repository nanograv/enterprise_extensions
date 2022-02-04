#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions.chromatic` submodule."""

import logging
import os
import pickle

import numpy as np
import pytest

from enterprise_extensions.chromatic import solar_wind as sw

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')


psr_names = ['J0613-0200', 'J1944+0907']


@pytest.fixture
def nodmx_psrs(caplog):
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir+'/{0}_ng11yr_nodmx_DE436_epsr.pkl'.format(p), 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_sw_r_to_p(nodmx_psrs):
    p0 = nodmx_psrs[0]
    dt_sw1 = sw.solar_wind_r_to_p(p0.toas, p0.freqs, p0.planetssb,
                                  p0.sunssb, p0.pos_t,
                                  n_earth=5, power=2, log10_ne=False)

    dt_sw2 = sw.solar_wind(p0.toas, p0.freqs, p0.planetssb,
                           p0.sunssb, p0.pos_t, n_earth=5)
    assert all(np.isclose(dt_sw1, dt_sw2, atol=1e-8))
