#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle
import inspect
import numpy as np

import pytest

from enterprise.pulsar import Pulsar

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


def test_timing_hypermodel(caplog):
    caplog.set_level(logging.CRITICAL)
    t2_psr = Pulsar(datadir+'/J1640+2224_ng9yr_dmx_DE421.par',
                    datadir+'/J1640+2224_ng9yr_dmx_DE421.tim',
                    ephem='DE421', clk=None, drop_t2pulsar=False)

    nltm_params_1 = []
    ltm_params_1 = []
    nltm_params_2 = []
    ltm_params_2 = []
    tm_param_dict = {}
    for par in t2_psr.fitpars:
        if par == "Offset":
            ltm_params_1.append(par)
            ltm_params_2.append(par)
        elif "DMX" in par:
            ltm_params_2.append(par)
        elif "JUMP" in par:
            ltm_params_2.append(par)
        elif "FD" in par:
            ltm_params_2.append(par)
        elif par == "SINI":
            nltm_params_2.append("COSI")
        else:
            nltm_params_1.append(par)
            nltm_params_2.append(par)

        if par in ['XDOT', 'PBDOT']:
            par_val = np.double(t2_psr.t2pulsar.vals()[t2_psr.t2pulsar.pars().index(par)])
            par_sigma = np.double(t2_psr.t2pulsar.errs()[t2_psr.t2pulsar.pars().index(par)])
            if np.log10(par_sigma) > -10.0:
                lower = par_val - 50 * par_sigma * 1e-12
                upper = par_val + 50 * par_sigma * 1e-12
                # lower = pbdot - 5 * pbdot_sigma * 1e-12
                # upper = pbdot + 5 * pbdot_sigma * 1e-12
                tm_param_dict[par] = {
                    "prior_mu": par_val,
                    "prior_sigma": par_sigma * 1e-12,
                    "prior_lower_bound": lower,
                    "prior_upper_bound": upper,
                }

    model_args = inspect.getfullargspec(models.model_singlepsr_noise)
    model_keys = model_args[0][1:]
    model_vals = model_args[3]
    model_kwargs_1 = dict(zip(model_keys, model_vals))
    model_kwargs_2 = dict(zip(model_keys, model_vals))
    model_kwargs_1.update(
        {
            "tm_var": True,
            "tm_linear": False,
            "tm_param_list": nltm_params_1,
            "ltm_list": ltm_params_1,
            "tm_param_dict": tm_param_dict,
            "tm_prior": "Uniform",
            "normalize_prior_bound": 50.0,
            "fit_remaining_pars": True,
        }
    )
    model_kwargs_2.update(
        {
            "tm_var": True,
            "tm_linear": False,
            "tm_param_list": nltm_params_2,
            "ltm_list": ltm_params_2,
            "tm_param_dict": tm_param_dict,
            "tm_prior": "Uniform",
            "normalize_prior_bound": 50.0,
            "fit_remaining_pars": True,
        }
    )

    ptas = {0: models.model_singlepsr_noise(t2_psr, **model_kwargs_1), 1: models.model_singlepsr_noise(t2_psr, **model_kwargs_2)}
    hm = hypermodel.HyperModel(ptas)
    samp = hm.setup_sampler(outdir=outdir, human='tester', timing=True, psr=t2_psr)
    assert hasattr(samp, "sample")
    paramfile = os.path.join(outdir, "pars.txt")
    assert os.path.isfile(paramfile)
    with open(paramfile, "r") as f:
        params = [line.rstrip('\n') for line in f]
    for ptapar, filepar in zip(hm.param_names, params):
        assert ptapar == filepar
    x0 = hm.initial_sample(tm_params_orig=t2_psr.tm_params_orig, tm_param_dict=tm_param_dict, zero_start=True)
    assert len(x0) == len(hm.param_names)
