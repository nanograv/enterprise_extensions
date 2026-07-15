#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_prior_draw_mode
----------------------------------

Tests for JumpProposal's honoring of Parameter.prior_draw_mode: a "joint"
vector parameter (correlated prior) must be replaced whole by a prior-draw
proposal, while a "component" vector parameter may still be replaced one
entry at a time.
"""

import numpy as np

from enterprise.signals import parameter
from enterprise.signals.parameter import Function
from enterprise_extensions.sampler import _draw_parameter_from_prior


def _joint_gaussian_parameter(mean, cov, *, fixed_sample=None):
    """A correlated 2D Gaussian UserParameter with prior_draw_mode='joint'."""
    prec = np.linalg.inv(cov)

    def _logprior(value):
        diff = np.asarray(value, dtype=float) - mean
        return float(-0.5 * diff @ prec @ diff)

    def _sampler(size=None):
        if fixed_sample is not None:
            return np.asarray(fixed_sample, dtype=float)
        return np.random.multivariate_normal(mean, cov)

    return parameter.UserParameter(
        logprior=Function(_logprior),
        sampler=_sampler,
        size=mean.size,
        prior_draw_mode="joint",
    )("timing_x")


def test_joint_vector_replaces_every_component():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    proposed = np.array([3.0, -4.0])
    param = _joint_gaussian_parameter(mean, cov, fixed_sample=proposed)

    q = np.array([0.1, 0.2])
    sl = slice(0, 2)

    _draw_parameter_from_prior(param, q, sl)

    np.testing.assert_allclose(q, proposed)


def test_joint_vector_hastings_correction_is_logp_old_minus_logp_new():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    proposed = np.array([3.0, -4.0])
    param = _joint_gaussian_parameter(mean, cov, fixed_sample=proposed)

    x_old = np.array([0.1, 0.2])
    q = x_old.copy()
    sl = slice(0, 2)

    _draw_parameter_from_prior(param, q, sl)
    lqxy = param.get_logpdf(x_old[sl]) - param.get_logpdf(q[sl])

    expected = param.get_logpdf(x_old) - param.get_logpdf(proposed)
    assert np.isclose(lqxy, expected)


def test_joint_vector_independence_proposal_recovers_target_moments():
    """An independence MH sampler using the joint prior draw as its sole
    proposal should recover the target mean/covariance, since proposing
    from the prior and correcting with lqxy = logp(old) - logp(new) gives
    an exact acceptance ratio of 1 for a likelihood-free target."""
    rng = np.random.default_rng(0)
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.8], [0.8, 1.5]])
    prec = np.linalg.inv(cov)

    def _logprior(value):
        diff = np.asarray(value, dtype=float) - mean
        return float(-0.5 * diff @ prec @ diff)

    def _sampler(size=None):
        return rng.multivariate_normal(mean, cov)

    param = parameter.UserParameter(
        logprior=Function(_logprior),
        sampler=_sampler,
        size=2,
        prior_draw_mode="joint",
    )("timing_x")

    sl = slice(0, 2)
    x = np.zeros(2)
    draws = []
    for _ in range(4000):
        q = x.copy()
        _draw_parameter_from_prior(param, q, sl)
        lqxy = param.get_logpdf(x[sl]) - param.get_logpdf(q[sl])
        # target is the prior itself (no likelihood): logpost = logprior
        log_alpha = (param.get_logpdf(q[sl]) - param.get_logpdf(x[sl])) + lqxy
        if np.log(rng.uniform()) < log_alpha:
            x = q
        draws.append(x.copy())

    draws = np.asarray(draws[500:])  # discard burn-in
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.3)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.6)


def test_component_vector_replaces_exactly_one_entry():
    proposed = np.array([9.0, 9.0, 9.0])
    param = parameter.UserParameter(
        prior=Function(lambda value: np.ones_like(np.asarray(value, dtype=float))),
        sampler=lambda size=None: proposed,
        size=3,
    )("free_vector")
    assert param.prior_draw_mode == "component"

    x_old = np.array([0.1, 0.2, 0.3])
    q = x_old.copy()
    sl = slice(0, 3)

    _draw_parameter_from_prior(param, q, sl)

    changed = ~np.isclose(q, x_old)
    assert changed.sum() == 1
    changed_idx = np.flatnonzero(changed)[0]
    assert q[changed_idx] == proposed[changed_idx]


def test_scalar_parameter_replaces_directly():
    param = parameter.Uniform(0.0, 1.0)("scalar")
    np.random.seed(0)

    x_old = np.array([0.3])
    q = x_old.copy()
    sl = slice(0, 1)

    _draw_parameter_from_prior(param, q, sl)

    assert q[0] != x_old[0]
    assert 0.0 <= q[0] <= 1.0


def test_size_one_vector_joint_draw_and_map_params():
    from enterprise.signals import signal_base

    mean = np.zeros(1)
    cov = np.array([[1.0]])
    proposed = np.array([2.5])
    param = _joint_gaussian_parameter(mean, cov, fixed_sample=proposed)
    assert param.size == 1

    class _FakePTAParams:
        def __init__(self, params):
            self.params = params

    fake_pta = _FakePTAParams([param])
    assert signal_base.PTA.param_names.fget(fake_pta) == ["timing_x_0"]

    mapped = signal_base.PTA.map_params(fake_pta, np.array([0.1]))
    assert isinstance(mapped["timing_x"], np.ndarray)
    assert mapped["timing_x"].shape == (1,)

    q = np.array([0.1])
    sl = slice(0, 1)
    _draw_parameter_from_prior(param, q, sl)
    np.testing.assert_allclose(q, proposed)
