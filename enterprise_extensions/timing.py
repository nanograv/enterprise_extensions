# -*- coding: utf-8 -*-

import json
import os
from collections import OrderedDict

import numpy as np
from enterprise.signals import deterministic_signals, parameter, signal_base
import scipy
import scipy.stats as sps
from scipy.stats import truncnorm


# Scipy defined RV for NE2001 DM Dist data.
defpath = os.path.dirname(__file__)
data_file = defpath + "/px_prior_1.txt"
px_prior = np.loadtxt(data_file)
px_hist = np.histogram(px_prior, bins=100, density=True)
px_rv = sps.rv_histogram(px_hist)


def BoundNormPrior(value, mu=0, sigma=1, pmin=-1, pmax=1):
    """Prior function for InvGamma parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    return truncnorm.pdf(value, loc=mu, scale=sigma, a=low, b=up)


def BoundNormSampler(mu=0, sigma=1, pmin=-1, pmax=1, size=None):
    """Sampling function for Uniform parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    return truncnorm.rvs(loc=mu, scale=sigma, a=low, b=up, size=size)


def BoundedNormal(mu=0, sigma=1, pmin=-1, pmax=1, size=None):
    """Class factory for bounded Normal parameters."""

    class BoundedNormal(parameter.Parameter):
        _prior = parameter.Function(
            BoundNormPrior, mu=mu, sigma=sigma, pmin=pmin, pmax=pmax,
        )
        _sampler = staticmethod(BoundNormSampler)
        _size = size
        _mu = mu
        _sigma = sigma
        _pmin = pmin
        _pmax = pmax

        def __repr__(self):
            return f"{self.name}: BoundedNormal({mu},{sigma}, [{pmin},{pmax}])" + ("" if self._size is None else f"[{self._size}]")

    return BoundedNormal


# NE2001 DM Dist data prior.
def NE2001DMDist_Prior(value):
    """Prior function for NE2001DMDist parameters."""
    return px_rv.pdf(value)


def NE2001DMDist_Sampler(size=None):
    """Sampling function for NE2001DMDist parameters."""
    return px_rv.rvs(size=size)


def NE2001DMDist_Parameter(size=None):
    """Class factory for NE2001DMDist parameters."""

    class NE2001DMDist_Parameter(parameter.Parameter):
        _size = size
        _typename = parameter._argrepr("NE2001DMDist")
        _prior = parameter.Function(NE2001DMDist_Prior)
        _sampler = staticmethod(NE2001DMDist_Sampler)

    return NE2001DMDist_Parameter


def get_default_physical_tm_priors():
    """Fills dictionary with physical bounds on timing parameters
    """
    physical_tm_priors = {}
    physical_tm_priors["E"] = {"pmin": 0.0, "pmax": 0.9999}
    physical_tm_priors["ECC"] = {"pmin": 0.0, "pmax": 0.9999}
    physical_tm_priors["SINI"] = {"pmin": 0.0, "pmax": 1.0}
    physical_tm_priors["COSI"] = {"pmin": 0.0, "pmax": 1.0}
    physical_tm_priors["PX"] = {"pmin": 0.0}
    physical_tm_priors["M2"] = {"pmin": 1e-10}

    return physical_tm_priors


def get_astrometric_priors(astrometric_px_file="../parallaxes.json"):
    # astrometric_px_file = '../parallaxes.json'
    astrometric_px = {}
    with open(astrometric_px_file) as pxf:
        astrometric_px = json.load(pxf)
        pxf.close()

    return astrometric_px


def get_prior(
    prior_type,
    prior_sigma,
    prior_lower_bound=-5.0,
    prior_upper_bound=5.0,
    mu=0.0,
    num_params=None,
):
    """Returns the requested prior for a parameter.

    :param prior_type: prior on timing parameters.
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    :param mu: Sets the mean/central value of prior if bounded normal is selected
    :param num_params: number of timing parameters assigned to prior. Default is None (ie. only one)
    """
    if prior_type.lower() == "bounded-normal":
        if mu < prior_lower_bound:
            mu = np.mean([prior_lower_bound, prior_upper_bound])
            prior_sigma = np.std([prior_lower_bound, prior_upper_bound])
        return BoundedNormal(
            mu=mu,
            sigma=prior_sigma,
            pmin=prior_lower_bound,
            pmax=prior_upper_bound,
            size=num_params,
        )
    elif prior_type.lower() == "uniform":
        return parameter.Uniform(prior_lower_bound, prior_upper_bound, size=num_params)
    elif prior_type.lower() == "normal":
        return parameter.Normal(mu=mu, sigma=prior_sigma, size=num_params)
    elif prior_type.lower() == "dm_dist_px_prior":
        return NE2001DMDist_Parameter(size=num_params)
    else:
        raise ValueError(
            "prior_type can only be uniform, normal, bounded-normal, or dm_dist_px_prior, not ",
            prior_type,
        )

@signal_base.function
def tm_delay(residuals, t2pulsar, tmparams_orig, tmparams, which='all'):
    """
    Compute difference in residuals due to perturbed timing model.

    :param residuals: original pulsar residuals from Pulsar object
    :param t2pulsar: libstempo pulsar object
    :param tmparams_orig: dictionary of TM parameter tuples, (val, err)
    :param tmparams: new timing model parameters, rescaled to be in sigmas
    :param which: option to have all or only named TM parameters varied

    :return: difference between new and old residuals in seconds
    """

    if which == 'all':
        keys = tmparams_orig.keys()
    else:
        keys = which

    # grab original timing model parameters and errors in dictionary
    orig_params = np.array([tmparams_orig[key] for key in keys])

    # put varying parameters into dictionary
    tmparams_rescaled = np.atleast_1d(np.double(orig_params[:, 0] +
                                                tmparams * orig_params[:, 1]))
    tmparams_vary = OrderedDict(zip(keys, tmparams_rescaled))

    # set to new values
    t2pulsar.vals(tmparams_vary)
    new_res = np.double(t2pulsar.residuals().copy())

    # remember to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params[:, 0])))))

    # Sort the residuals
    isort = np.argsort(t2pulsar.toas(), kind='mergesort')

    return residuals[isort] - new_res[isort]

# Model component building blocks #


def timing_block(tmparam_list=['RAJ', 'DECJ', 'F0', 'F1',
                               'PMRA', 'PMDEC', 'PX']):
    """
    Returns the timing model block of the model
    :param tmparam_list: a list of parameters to vary in the model
    """
    # default 5-sigma prior above and below the parfile mean
    tm_params = parameter.Uniform(-5.0, 5.0, size=len(tmparam_list))

    # timing model
    tm_func = tm_delay(tmparams=tm_params, which=tmparam_list)
    tm = deterministic_signals.Deterministic(tm_func, name='timing model')

    return tm
