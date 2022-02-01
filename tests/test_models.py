#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle

import pytest
from enterprise import constants as const

from enterprise_extensions import model_utils, models

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')


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


def test_model_singlepsr_noise(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m=models.model_singlepsr_noise(nodmx_psrs[1])
    assert hasattr(m, 'get_lnlikelihood')


def test_model_singlepsr_noise_faclike(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    # default behaviour
    m=models.model_singlepsr_noise(nodmx_psrs[1],
                                   factorized_like=True, Tspan=10*const.yr)
    m.get_basis()
    assert 'gw_log10_A' in m.param_names
    assert 'J1713+0747_red_noise_log10_A' in m.param_names
    assert m.signals["J1713+0747_gw"]._labels[''][-1] == const.fyr

    # gw but no RN
    m=models.model_singlepsr_noise(nodmx_psrs[1], red_var=False,
                                   factorized_like=True, Tspan=10*const.yr)
    assert hasattr(m, 'get_lnlikelihood')
    assert 'gw_log10_A' in m.param_names
    assert 'J1713+0747_red_noise_log10_A' not in m.param_names


def test_model_singlepsr_noise_sw(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True,
                                   dm_sw_gp=True, swgp_basis='powerlaw')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True,
                                   dm_sw_gp=True, swgp_basis='periodic')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True,
                                   dm_sw_gp=True, swgp_basis='sq_exp')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model_singlepsr_noise_dip_cusp(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    dip_kwargs = {'dm_expdip': True,
                  'dmexp_sign': 'negative',
                  'num_dmdips': 2,
                  'dm_expdip_tmin': [54700, 57450],
                  'dm_expdip_tmax': [54850, 57560],
                  'dmdip_seqname': ['1st_ism', '2nd_ism'],
                  'dm_cusp': False,
                  'dm_cusp_sign': 'negative',
                  'dm_cusp_idx': [2, 4],
                  'dm_cusp_sym': False,
                  'dm_cusp_tmin': None,
                  'dm_cusp_tmax': None,
                  'num_dm_cusps': 2,
                  'dm_dual_cusp': True,
                  'dm_dual_cusp_tmin': [54700, 57450],
                  'dm_dual_cusp_tmax': [54850, 57560], }
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True,
                                   dm_sw_gp=True, **dip_kwargs)
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model_singlepsr_noise_chrom_nondiag(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m=models.model_singlepsr_noise(nodmx_psrs[0], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag')
    assert 'J0613-0200_chrom_gp_log10_sigma' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell2' not in m.param_names
    assert 'J0613-0200_chrom_gp_log10_alpha_wgt' not in m.param_names
    assert 'J0613-0200_chrom_gp_log10_p' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag')
    assert 'J1713+0747_chrom_gp_log10_sigma' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell2' not in m.param_names
    assert 'J1713+0747_chrom_gp_log10_alpha_wgt' not in m.param_names
    assert 'J1713+0747_chrom_gp_log10_p' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[2], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag')
    assert 'J1909-3744_chrom_gp_log10_sigma' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell2' not in m.param_names
    assert 'J1909-3744_chrom_gp_log10_alpha_wgt' not in m.param_names
    assert 'J1909-3744_chrom_gp_log10_p' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[0], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='periodic_rfband')
    assert 'J0613-0200_chrom_gp_log10_sigma' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell2' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_p' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='periodic_rfband')
    assert 'J1713+0747_chrom_gp_log10_sigma' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell2' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_p' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[2], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='periodic_rfband')
    assert 'J1909-3744_chrom_gp_log10_sigma' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell2' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_p' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_gam_p' in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[0], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp')
    assert 'J0613-0200_chrom_gp_log10_sigma' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_p' not in m.param_names
    assert 'J0613-0200_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp')
    assert 'J1713+0747_chrom_gp_log10_sigma' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_p' not in m.param_names
    assert 'J1713+0747_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[2], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp')
    assert 'J1909-3744_chrom_gp_log10_sigma' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_p' not in m.param_names
    assert 'J1909-3744_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[0], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp_rfband')
    assert 'J0613-0200_chrom_gp_log10_sigma' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_ell2' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J0613-0200_chrom_gp_log10_p' not in m.param_names
    assert 'J0613-0200_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp_rfband')
    assert 'J1713+0747_chrom_gp_log10_sigma' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_ell2' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J1713+0747_chrom_gp_log10_p' not in m.param_names
    assert 'J1713+0747_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[2], dm_var=True,
                                   dm_type=None, chrom_gp=True,
                                   chrom_gp_kernel='nondiag',
                                   chrom_kernel='sq_exp_rfband')
    assert 'J1909-3744_chrom_gp_log10_sigma' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_ell2' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_alpha_wgt' in m.param_names
    assert 'J1909-3744_chrom_gp_log10_p' not in m.param_names
    assert 'J1909-3744_chrom_gp_log10_gam_p' not in m.param_names
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model_singlepsr_noise_chrom_diag(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m=models.model_singlepsr_noise(nodmx_psrs[1], chrom_gp=True,
                                   chrom_gp_kernel='diag')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], chrom_gp=True,
                                   chrom_gp_kernel='diag',
                                   chrom_psd='turnover')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)
    m=models.model_singlepsr_noise(nodmx_psrs[1], chrom_gp=True,
                                   chrom_gp_kernel='diag',
                                   chrom_psd='spectrum')
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model_singlepsr_fact_like(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    psr = nodmx_psrs[1]
    Tspan = model_utils.get_tspan([psr])
    m=models.model_singlepsr_noise(psr, chrom_gp=True,
                                   chrom_gp_kernel='diag',
                                   factorized_like=False,
                                   Tspan=Tspan, fact_like_gamma=13./3,
                                   gw_components=5)
    assert hasattr(m, 'get_lnlikelihood')
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_modelbwmsglpsr(nodmx_psrs, caplog):
    nodmx_psr=nodmx_psrs[0]

    m=models.model_bwm_sglpsr(nodmx_psr)  # should I be testing the Log and Lookup Likelihoods?
    # If this test belongs in enterprise/tests instead, do
    # I need to include the lookup table in tests/data?
    assert hasattr(m, 'get_lnlikelihood')
    assert "ramp_log10_A" in m.param_names
    assert "ramp_t0" in m.param_names


def test_modelbwm(nodmx_psrs, caplog):
    m=models.model_bwm(nodmx_psrs)  # should I be testing the Log and Lookup Likelihoods?
    # If this test belongs in enterprise/tests instead, do
    # I need to include the lookup table in tests/data?
    assert hasattr(m, 'get_lnlikelihood')
    assert "bwm_log10_A" in m.param_names
    assert "bwm_t0" in m.param_names
    assert "bwm_phi" in m.param_names
    assert "bwm_pol" in m.param_names
    assert "bwm_costheta" in m.param_names


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model1(dmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m1=models.model_1(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m1, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a_pshift(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs, noisedict=noise_dict, pshift=True, pseed=42)
    assert hasattr(m2a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a_5gwb(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs, n_gwbfreqs=5, noisedict=noise_dict)
    assert hasattr(m2a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a_broken_plaw(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs, psd='broken_powerlaw', delta_common=0,
                        noisedict=noise_dict)
    assert hasattr(m2a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2b(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2b=models.model_2b(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2b, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2c(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2c=models.model_2c(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2c, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2d(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2d=models.model_2d(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2d, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a_pshift(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs, noisedict=noise_dict, pshift=True, pseed=42)
    assert hasattr(m3a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a_5rnfreqs(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs, n_rnfreqs=5, noisedict=noise_dict)
    assert hasattr(m3a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a_broken_plaw(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs, psd='broken_powerlaw', delta_common=0,
                        noisedict=noise_dict)
    assert hasattr(m3a, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3b(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3b=models.model_3b(dmx_psrs)
    assert hasattr(m3b, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3c(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3c=models.model_3c(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3c, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3d(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3d=models.model_3d(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3d, 'get_lnlikelihood')


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model_fdm(dmx_psrs, caplog):
    fdm=models.model_fdm(dmx_psrs, noisedict=noise_dict)
    assert hasattr(fdm, 'get_lnlikelihood')
