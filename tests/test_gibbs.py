# -*- coding: utf-8 -*-

"""
Tests for the gibbs sampling code.
"""

import logging
import os
import pickle
import enterprise_extensions.GibbsMethod.BayesPower as BP
import pytest

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


psr_names = ["J0613-0200"]

datadir = os.path.join(testdir, 'data')


psr_names = ['J0613-0200']

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


def test_gibbs(nodmx_psrs, caplog):
    psr = nodmx_psrs[0]
    BPC = BP.BayesPower(
        psr=psr,
        Tspan=None,
        select="backend",
        white_vary=True,
        inc_ecorr=True,
        ecorr_type="kernel",
        noise_dict=None,
        tm_marg=False,
        freq_bins=10,
        rhomin=-9,
        rhomax=-4,
    )

    assert hasattr(BPC, "sample")

    psr = nodmx_psrs[0]
    BPC = BP.BayesPower(
                psr = psr,
                Tspan = None,
                select = 'backend',
                white_vary = True,
                inc_ecorr = True,
                ecorr_type = 'kernel',
                noise_dict = None,
                tm_marg = False,
                freq_bins = 10,
                rhomin = -9, rhomax = -4)

    assert hasattr(BPC, 'sample')







