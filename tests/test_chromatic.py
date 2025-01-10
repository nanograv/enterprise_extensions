#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions.chromatic` submodule."""

import logging
import os
import pickle

import numpy as np
import pytest

from enterprise_extensions.chromatic import solar_wind as sw
from enterprise_extensions.chromatic import construct_chromatic_cached_parts
from enterprise_extensions.chromatic import createfourierdesignmatrix_chromatic_with_additional_caching
from enterprise.signals import parameter, gp_bases, signal_base, gp_priors, gp_signals, white_signals, selections

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

psr_names = ['J0613-0200', 'J1944+0907']


@pytest.fixture
def nodmx_psrs(caplog):
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir + f'/{p}_ng11yr_nodmx_DE436_epsr.pkl', 'rb') as fin:
            psrs.append(pickle.load(fin))

    return psrs


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_sw_r_to_p(nodmx_psrs):
    p0 = nodmx_psrs[0]
    dt_sw1 = sw.solar_wind_r_to_p(
        p0.toas, p0.freqs, p0.planetssb, p0.sunssb, p0.pos_t,
        n_earth=5, power=2, log10_ne=False
    )

    dt_sw2 = sw.solar_wind(
        p0.toas, p0.freqs, p0.planetssb, p0.sunssb, p0.pos_t, n_earth=5
    )
    assert all(np.isclose(dt_sw1, dt_sw2, atol=1e-8))


def test_chromatic_fourier_basis_varied_idx(nodmx_psrs):
    """Test the set up of variable index chromatic bases and make sure that the caching is the same as no caching"""
    p0 = nodmx_psrs[0]
    idx = parameter.Uniform(2.5, 7)
    uncached_basis = gp_bases.createfourierdesignmatrix_chromatic(
        p0.toas, p0.freqs, nmodes=100, idx=idx
    )
    fmat_red, Ffreqs, nus = construct_chromatic_cached_parts(p0.toas, p0.freqs, nmodes=100)
    cached_basis = createfourierdesignmatrix_chromatic_with_additional_caching(
        fmat_red=fmat_red, Ffreqs=Ffreqs, fref_over_radio_freqs=nus, idx=idx
    )
    pr = gp_priors.powerlaw(log10_A=parameter.Uniform(-18, -11), gamma=parameter.Uniform(1, 7))
    uncached = gp_signals.BasisGP(priorFunction=pr, basisFunction=uncached_basis, name="chrom_gp")
    cached = gp_signals.BasisGP(priorFunction=pr, basisFunction=cached_basis, name="chrom_gp")
    pr = gp_priors.powerlaw_genmodes(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
    basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
    rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
    efac = parameter.Normal(1.0, 0.1)
    backend = selections.Selection(selections.by_backend)
    equad = parameter.Uniform(-8.5, -5)
    wn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=backend, name=None)
    mod1 = uncached + rn + wn
    mod2 = cached + rn + wn
    uncached_pta = signal_base.PTA([mod1(p0)])
    cached_pta = signal_base.PTA([mod2(p0)])

    # check that both of the chromatic bases have the chromatic index as a parameter
    msg = "chromatic index missing from pta parameter list"
    assert "J0613-0200_chrom_gp_idx" in uncached_pta.param_names, msg
    assert "J0613-0200_chrom_gp_idx" in cached_pta.param_names, msg

    # test to make sure the likelihood evaluations agree for 10 calls
    msg = "the likelihood from cached chromatic basis disagrees with the uncached chromatic basis likelihood"
    x0 = [np.hstack([p.sample() for p in cached_pta.params]) for _ in range(10)]
    no_cache_lnlike = [uncached_pta.get_lnlikelihood(x0[i]) for i in range(10)]
    cache_lnlike = [cached_pta.get_lnlikelihood(x0[i]) for i in range(10)]
    assert np.all(no_cache_lnlike == cache_lnlike), msg

    # check that both the cached and the uncached basis yield the same basis
    msg = "the cached chromatic basis does not match the uncached chromatic basis"
    assert np.all(uncached_pta.get_basis(params={})[0] == cached_pta.get_basis(params={})[0]), msg