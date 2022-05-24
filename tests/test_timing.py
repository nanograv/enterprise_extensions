#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""
# TODO: Add PINT timing_block test
# TODO: Add Wideband tests

import logging
import os
import pytest
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base

from enterprise_extensions import timing as tm
from enterprise_extensions import sampler
from enterprise_extensions.blocks import white_noise_block

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")

psr_names = ['J1640+2224']


@pytest.fixture
def t2_psr(caplog):
    caplog.set_level(logging.CRITICAL)
    psr = Pulsar(datadir+f'/{psr_names[0]}_ng9yr_dmx_DE421.par',
                 datadir+f'/{psr_names[0]}_ng9yr_dmx_DE421.tim',
                 ephem='DE421', clk=None, drop_t2pulsar=False)

    return psr


@pytest.fixture
def pint_psr(caplog):
    caplog.set_level(logging.CRITICAL)
    psr = Pulsar(datadir+f'/{psr_names[0]}_ng9yr_dmx_DE421.par',
                 datadir+f'/{psr_names[0]}_ng9yr_dmx_DE421.tim',
                 ephem='DE421', clk=None, drop_pintpsr=False, timing_package='pint')

    return psr


def test_timing_block(t2_psr, caplog):
    nltm_params = []
    ltm_params = []
    tm_param_dict = {}
    for par in t2_psr.fitpars:
        if par == "Offset":
            ltm_params.append(par)
        elif par in ['XDOT', 'PBDOT']:
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
        else:
            nltm_params.append(par)

    tm.timing_block(t2_psr,
                    tm_param_list=nltm_params,
                    ltm_list=ltm_params,
                    prior_type="uniform",
                    prior_sigma=2.0,
                    prior_lower_bound=-5.0,
                    prior_upper_bound=5.0,
                    tm_param_dict=tm_param_dict,
                    fit_remaining_pars=True,
                    wideband_kwargs={},)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_tm_delay_t2(t2_psr, caplog):
    nltm_params = []
    ltm_params = []
    tm_param_dict = {}
    for par in t2_psr.fitpars:
        if par == "Offset":
            ltm_params.append(par)
        elif par in ['XDOT', 'PBDOT']:
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
        elif par == "PX":
            # Use the priors from Vigeland and Vallisneri 2014
            tm_param_dict[par] = {
                "prior_mu": np.double(t2_psr.t2pulsar.vals()[t2_psr.t2pulsar.pars().index(par)]),
                "prior_sigma": np.double(
                    t2_psr.t2pulsar.errs()[t2_psr.t2pulsar.pars().index(par)]
                ),
                "prior_type": "dm_dist_px_prior",
            }
        else:
            nltm_params.append(par)

    s = tm.timing_block(
        t2_psr,
        tm_param_list=nltm_params,
        ltm_list=ltm_params,
        prior_type="bounded-normal",
        prior_sigma=2.0,
        prior_lower_bound=-5.0,
        prior_upper_bound=5.0,
        tm_param_dict=tm_param_dict,
        fit_remaining_pars=True,
        wideband_kwargs={},
    )

    s += white_noise_block(vary=True, inc_ecorr=True, tnequad=True)

    pta = signal_base.PTA(s(t2_psr))

    psampler = sampler.setup_sampler(
        pta, outdir='./outdir_tests', resume=False, timing=True)

    x0_list = []
    for p in pta.params:
        if "timing" in p.name:
            if "DMX" in p.name:
                p_name = ("_").join(p.name.split("_")[-2:])
            else:
                p_name = p.name.split("_")[-1]
            if t2_psr.tm_params_orig[p_name][-1] == "normalized":
                x0_list.append(np.double(0.0))
            else:
                if p_name in tm_param_dict.keys():
                    x0_list.append(np.double(tm_param_dict[p_name]["prior_mu"]))
                else:
                    x0_list.append(np.double(t2_psr.tm_params_orig[p_name][0]))
        else:
            x0_list.append(p.sample())
    x0 = np.asarray(x0_list)

    psampler.sample(
        x0,
        100,
        SCAMweight=30,
        AMweight=15,
        DEweight=30,
    )

    if os.path.isdir('./outdir_tests'):
        for file in os.listdir('./outdir_tests'):
            os.remove('./outdir_tests/'+file)

        os.removedirs('./outdir_tests')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_tm_delay_pint(pint_psr, caplog):
    nltm_params = []
    ltm_params = []
    tm_param_dict = {}
    for par in pint_psr.fitpars:
        if par == "Offset":
            ltm_params.append(par)
        else:
            nltm_params.append(par)
    s = tm.timing_block(
        pint_psr,
        tm_param_list=nltm_params,
        ltm_list=ltm_params,
        prior_type="bounded-normal",
        prior_sigma=2.0,
        prior_lower_bound=-5.0,
        prior_upper_bound=5.0,
        tm_param_dict=tm_param_dict,
        fit_remaining_pars=True,
        wideband_kwargs={},
    )

    s += white_noise_block(vary=True, inc_ecorr=True, tnequad=True)

    pta = signal_base.PTA(s(pint_psr))

    psampler = sampler.setup_sampler(
        pta, outdir='./outdir_tests', resume=False, timing=True)

    x0_list = []
    for p in pta.params:
        if "timing" in p.name:
            if "DMX" in p.name:
                p_name = ("_").join(p.name.split("_")[-2:])
            else:
                p_name = p.name.split("_")[-1]
            if pint_psr.tm_params_orig[p_name][-1] == "normalized":
                x0_list.append(np.double(0.0))
            else:
                if p_name in tm_param_dict.keys():
                    x0_list.append(np.double(tm_param_dict[p_name]["prior_mu"]))
                else:
                    x0_list.append(np.double(pint_psr.tm_params_orig[p_name][0]))
        else:
            x0_list.append(p.sample())
    x0 = np.asarray(x0_list)

    psampler.sample(
        x0,
        100,
        SCAMweight=30,
        AMweight=15,
        DEweight=30,
    )

    if os.path.isdir('./outdir_tests'):
        for file in os.listdir('./outdir_tests'):
            os.remove('./outdir_tests/'+file)

        os.removedirs('./outdir_tests')
