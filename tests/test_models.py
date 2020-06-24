#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import pytest
import pickle, json, os
import logging
from enterprise_extensions import models, model_utils, sampler

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


psr_names = ["J0613-0200", "J1713+0747", "J1909-3744"]

with open(datadir + "/ng11yr_noise.json", "r") as fin:
    noise_dict = json.load(fin)


@pytest.fixture
def dmx_psrs(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    psrs = []
    for p in psr_names:
        with open(datadir + "/{0}_ng9yr_dmx_DE436_epsr.pkl".format(p), "rb") as fin:
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
        with open(datadir + "/{0}_ng9yr_nodmx_DE436_epsr.pkl".format(p), "rb") as fin:
            psrs.append(pickle.load(fin))

    return psrs


def test_model_singlepsr_noise(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m = models.model_singlepsr_noise(nodmx_psrs[1])
    assert hasattr(m, "get_lnlikelihood")


def test_model_singlepsr_noise_sw(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m = models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True, dm_sw_gp=True)
    assert hasattr(m, "get_lnlikelihood")
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)

def test_model_singlepsr_noise_dip_cusp(nodmx_psrs,caplog):
    # caplog.set_level(logging.CRITICAL)
    dip_kwargs = {'dm_expdip':True,
                  'dmexp_sign': 'negative',
                  'num_dmdips':2,
                  'dm_cusp_idx':[2,4],
                  'dm_expdip_tmin':[54700,57450],
                  'dm_expdip_tmax':[54850,57560],
                  'dmdip_seqname':['1st_ism','2nd_ism'],
                  'dm_cusp':False,
                  'dm_cusp_sign':'negative',
                  'dm_cusp_idx':[2,4],
                  'dm_cusp_sym':False,
                  'dm_cusp_tmin':None,
                  'dm_cusp_tmax':None,
                  'num_dm_cusps':2,
                  'dm_dual_cusp':True,
                  'dm_dual_cusp_tmin':[54700,57450],
                  'dm_dual_cusp_tmax':[54850,57560],}
    m=models.model_singlepsr_noise(nodmx_psrs[1], dm_sw_deter=True,
                                   dm_sw_gp=True,**dip_kwargs)
    assert hasattr(m,'get_lnlikelihood')
    x0 = {pname:p.sample() for pname,p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)

def test_model_singlepsr_noise_chrom_nondiag(nodmx_psrs,caplog):
    # caplog.set_level(logging.CRITICAL)
    m = models.model_singlepsr_noise(nodmx_psrs[1], chrom_gp=True)
    assert hasattr(m, "get_lnlikelihood")
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model_singlepsr_noise_chrom_diag(nodmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m = models.model_singlepsr_noise(
        nodmx_psrs[1], chrom_gp=True, chrom_gp_kernel="diag"
    )
    assert hasattr(m, "get_lnlikelihood")
    x0 = {pname: p.sample() for pname, p in zip(m.param_names, m.params)}
    m.get_lnlikelihood(x0)


def test_model1(dmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m1 = models.model_1(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m1, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model1(dmx_psrs, caplog):
    # caplog.set_level(logging.CRITICAL)
    m1 = models.model_1(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m1, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model2a(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2a, "get_lnlikelihood")


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a_pshift(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs,noisedict=noise_dict,pshift=True,pseed=42)
    assert hasattr(m2a,'get_lnlikelihood')

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2a_5gwb(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m2a=models.model_2a(dmx_psrs, n_gwbfreqs=5, noisedict=noise_dict)
    assert hasattr(m2a,'get_lnlikelihood')

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model2b(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m2b = models.model_2b(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2b, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model2c(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2c = models.model_2c(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2c, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model2d(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m2d = models.model_2d(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m2d, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model3a(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3a = models.model_3a(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3a, "get_lnlikelihood")


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a_pshift(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs,noisedict=noise_dict,pshift=True,pseed=42)
    assert hasattr(m3a,'get_lnlikelihood')

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3a_5rnfreqs(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m3a=models.model_3a(dmx_psrs, n_rnfreqs=5, noisedict=noise_dict)
    assert hasattr(m3a,'get_lnlikelihood')

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model3b(dmx_psrs,caplog):
    caplog.set_level(logging.CRITICAL)
    m3b = models.model_3b(dmx_psrs)
    assert hasattr(m3b, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model3c(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3c = models.model_3c(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3c, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_model3d(dmx_psrs, caplog):
    caplog.set_level(logging.CRITICAL)
    m3d = models.model_3d(dmx_psrs, noisedict=noise_dict)
    assert hasattr(m3d, "get_lnlikelihood")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_jumpproposal(dmx_psrs, caplog):
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    jp = sampler.JumpProposal(m2a)
    assert jp.draw_from_prior.__name__ == "draw_from_prior"
    assert jp.draw_from_signal_prior.__name__ == "draw_from_signal_prior"
    assert jp.draw_from_par_prior("J1713+0747").__name__ == "draw_from_J1713+0747_prior"
    assert (
        jp.draw_from_par_log_uniform({"gw": (-20, -10)}).__name__
        == "draw_from_gw_log_uniform"
    )
    assert jp.draw_from_signal("red noise").__name__ == "draw_from_red noise_signal"
