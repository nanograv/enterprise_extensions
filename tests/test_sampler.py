#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import json
import logging
import os
import pickle

import pytest

from enterprise_extensions import models, sampler
from enterprise_extensions.empirical_distr import (
    make_empirical_distributions, make_empirical_distributions_KDE)

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


@pytest.fixture
def empirical_distribution_1d(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    with open(datadir+'/emp_dist_1d.pkl', 'rb') as fin:
        emp_dists = pickle.load(fin)

    return emp_dists


@pytest.fixture
def empirical_distribution_1d_kde(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    with open(datadir+'/emp_dist_samples.pkl', 'rb') as fin:
        emp_dists = pickle.load(fin)

    return emp_dists


@pytest.fixture
def empirical_distribution_2d(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    with open(datadir+'/emp_dist_2d.pkl', 'rb') as fin:
        emp_dists = pickle.load(fin)

    return emp_dists


@pytest.fixture
def empirical_distribution_2d_kde(caplog):
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    caplog.set_level(logging.CRITICAL)
    with open(datadir+'/emp_dist_2d_kde.pkl', 'rb') as fin:
        emp_dists = pickle.load(fin)

    return emp_dists


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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_extend_emp_dists_1d(dmx_psrs, caplog):
    with open(datadir+'/emp_dist_samples.pkl', 'rb') as fin:
        tmp_data = pickle.load(fin)

    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    new_dist = make_empirical_distributions(m2a, tmp_data['names'], tmp_data['names'],
                                            tmp_data['samples'], save_dists=False)
    # run extend when edges match priors
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    # change priors so they don't match edges of
    # empirical distribution
    for ii in range(len(tmp_data['names'])):
        m2a.params[ii].prior._defaults['pmin'] -= 0.1
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    assert len(new_dist) == 6
    for i in range(6):
        assert new_dist[i]._edges[0] <= m2a.params[i].prior._defaults['pmin']
        assert new_dist[i]._edges[-1] >= m2a.params[i].prior._defaults['pmax']


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_extend_emp_dists_2d(dmx_psrs, caplog):
    with open(datadir+'/emp_dist_samples.pkl', 'rb') as fin:
        tmp_data = pickle.load(fin)
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    parnames = [[tmp_data['names'][0], tmp_data['names'][1]],
                [tmp_data['names'][2], tmp_data['names'][3]],
                [tmp_data['names'][4], tmp_data['names'][5]]]
    new_dist = make_empirical_distributions(m2a, parnames, tmp_data['names'],
                                            tmp_data['samples'], save_dists=False)
    # case 1, edges match priors
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    # case 2, edges don't match priors (set priors to be different)
    for ii in range(len(tmp_data['names'])):
        m2a.params[ii].prior._defaults['pmin'] -= 0.1
        m2a.params[ii].prior._defaults['pmax'] += 0.1
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    assert len(new_dist) == 3
    for i in range(3):
        k = 2 * i
        assert new_dist[i]._edges[0][0] <= m2a.params[k].prior._defaults['pmin']
        assert new_dist[i]._edges[0][-1] <= m2a.params[k].prior._defaults['pmax']
        assert new_dist[i]._edges[1][0] <= m2a.params[k + 1].prior._defaults['pmin']
        assert new_dist[i]._edges[1][-1] <= m2a.params[k + 1].prior._defaults['pmax']


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_extend_emp_dists_1d_kde(dmx_psrs, caplog):
    with open(datadir+'/emp_dist_samples.pkl', 'rb') as fin:
        tmp_data = pickle.load(fin)

    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    new_dist = make_empirical_distributions_KDE(m2a, tmp_data['names'], tmp_data['names'],
                                                tmp_data['samples'], save_dists=False)
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    for ii in range(len(tmp_data['names'])):
        m2a.params[ii].prior._defaults['pmin'] -= 0.1
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    assert len(new_dist) == 6
    for i in range(6):
        assert new_dist[i].minval <= m2a.params[i].prior._defaults['pmin']
        assert new_dist[i].maxval >= m2a.params[i].prior._defaults['pmax']


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_extend_emp_dists_2d_kde(dmx_psrs, caplog):

    with open(datadir+'/emp_dist_samples.pkl', 'rb') as fin:
        tmp_data = pickle.load(fin)
    m2a = models.model_2a(dmx_psrs, noisedict=noise_dict)
    parnames = [[tmp_data['names'][0], tmp_data['names'][1]],
                [tmp_data['names'][2], tmp_data['names'][3]],
                [tmp_data['names'][4], tmp_data['names'][5]]]
    new_dist = make_empirical_distributions_KDE(m2a, parnames, tmp_data['names'],
                                                tmp_data['samples'], save_dists=False)
    # case 1
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    # case 2
    for ii in range(len(tmp_data['names'])):
        m2a.params[ii].prior._defaults['pmin'] -= 0.1
        m2a.params[ii].prior._defaults['pmax'] += 0.1
    new_dist = sampler.extend_emp_dists(m2a, new_dist)
    assert len(new_dist) == 3
    for i in range(3):
        k = 2 * i
        assert new_dist[i].minvals[0] <= m2a.params[k].prior._defaults['pmin']
        assert new_dist[i].maxvals[0] <= m2a.params[k].prior._defaults['pmax']
        assert new_dist[i].minvals[1] <= m2a.params[k + 1].prior._defaults['pmin']
        assert new_dist[i].maxvals[1] <= m2a.params[k + 1].prior._defaults['pmax']
