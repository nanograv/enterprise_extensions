# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from collections import defaultdict
from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from scipy.stats import truncnorm

import time


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
            BoundNormPrior, mu=mu, sigma=sigma, pmin=pmin, pmax=pmax
        )
        _sampler = staticmethod(BoundNormSampler)
        _size = size
        _mu = mu
        _sigma = sigma
        _pmin = pmin
        _pmax = pmax

        def __repr__(self):
            return "{}: BoundedNormal({},{}, [{},{}])".format(
                self.name, mu, sigma, pmin, pmax
            ) + ("" if self._size is None else "[{}]".format(self._size))

    return BoundedNormal


# timing model delay
@signal_base.function
def tm_delay(
    t2pulsar,
    tmparams_orig,
    param_dict,
    pos_params,
    pm_params,
    spin_params,
    kep_params,
    gr_params,
):
    """
    Compute difference in residuals due to perturbed timing model.

    :param residuals: original pulsar residuals from Pulsar object
    :param t2pulsar: libstempo pulsar object
    :param tmparams_orig: dictionary of TM parameter tuples, (val, err)
    :param tmparams: new timing model parameters, rescaled to be in sigmas
    :param which: option to have all or only named TM parameters varied

    :return: difference between new and old residuals in seconds
    """
    """OUTLINE:
    take in parameters in par file
    save to dictionary
    Based on params in input param list, set parameter prior distribution
    Feed the priors and param list into tm_delay function
    """
    residuals = t2pulsar.residuals()

    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_category, tm_param_keys in param_dict.items():
        if tm_category == "pos":
            normed_params = pos_params
        elif tm_category == "pm":
            normed_params = pm_params
        elif tm_category == "spin":
            normed_params = spin_params
        elif tm_category == "kep":
            normed_params = kep_params
        elif tm_category == "gr":
            normed_params = gr_params
        else:
            normed_params = None
        for i, tm_param in enumerate(tm_param_keys):
            orig_params[tm_param] = tmparams_orig[tm_param][0]
            if isinstance(normed_params, (list, np.ndarray)):
                tm_params_rescaled[tm_param] = (
                    normed_params[i] * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
            elif isinstance(normed_params, (float, int)):
                tm_params_rescaled[tm_param] = (
                    normed_params * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
            else:
                raise ValueError(
                    "sampled parameters cannot by of type ", type(normed_params)
                )

    # set to new values
    t2pulsar.vals(tm_params_rescaled)
    new_res = t2pulsar.residuals()

    # remmeber to set values back to originals
    t2pulsar.vals(orig_params)

    # Return the time-series for the pulsar
    return new_res - residuals


# Model component building blocks #


def timing_block(
    tmparam_list=["RAJ", "DECJ", "F0", "F1", "PMRA", "PMDEC", "PX"],
    prior_type="bounded-normal",
    prior_sigma=2.0,
    prior_lower_bound=-3.0,
    prior_upper_bound=3.0,
):
    """
    Returns the timing model block of the model

    :param tmparam_list: a list of parameters to vary in the model
    :param prior_type: prior on timing parameters. Default is a bounded normal, can be "uniform"
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    """
    param_dict = defaultdict(list)
    for par in tmparam_list:
        if par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]:
            param_dict["pos"].append(par)
        elif par in ["PMDEC", "PMRA", "PMRV", "PMBETA", "PMLAMBDA"]:
            param_dict["pm"].append(par)
        elif par in ["F", "F0", "F1", "F2", "P", "P1"]:
            param_dict["spin"].append(par)
        elif par in [
            "PB",
            "T0",
            "A1",
            "OM",
            "E",
            "ECC",
            "EPS1",
            "EPS2",
            "EPS1DOT",
            "EPS2DOT",
            "FB",
            "SINI",
            "MTOT",
            "M2",
            "XDOT",
            "X2DOT",
            "EDOT",
        ]:
            param_dict["kep"].append(par)
        elif par in [
            "H3",
            "H4",
            "OMDOT",
            "OM2DOT",
            "XOMDOT",
            "PBDOT",
            "XPBDOT",
            "GAMMA",
            "PPNGAMMA",
            "DR",
            "DTHETA",
        ]:
            param_dict["gr"].append(par)
        else:
            if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
                param_dict["dmx"].append(par)
            else:
                print(par, " is not currently a modelled parameter.")

    # default 2-sigma prior above and below the parfile mean
    if len(param_dict["pos"]) != 0:
        if prior_type == "bounded-normal":
            pos_params = BoundedNormal(
                mu=0.0,
                sigma=prior_sigma,
                pmin=prior_lower_bound,
                pmax=prior_upper_bound,
                size=len(param_dict["pos"]),
            )
        elif prior_type == "uniform":
            pos_params = parameter.Uniform(
                prior_lower_bound, prior_upper_bound, size=len(param_dict["pos"])
            )
        else:
            raise ValueError(
                "prior_type can only be uniform or bounded-normal, not ", prior_type
            )
    else:
        pos_params = None

    if len(param_dict["pm"]) != 0:
        if prior_type == "bounded-normal":
            pm_params = BoundedNormal(
                mu=0.0,
                sigma=prior_sigma,
                pmin=prior_lower_bound,
                pmax=prior_upper_bound,
                size=len(param_dict["pm"]),
            )
        elif prior_type == "uniform":
            pm_params = parameter.Uniform(
                prior_lower_bound, prior_upper_bound, size=len(param_dict["pm"])
            )
        else:
            raise ValueError(
                "prior_type can only be uniform or bounded-normal, not ", prior_type
            )
    else:
        pm_params = None

    if len(param_dict["spin"]) != 0:
        if prior_type == "bounded-normal":
            spin_params = BoundedNormal(
                mu=0.0,
                sigma=prior_sigma,
                pmin=prior_lower_bound,
                pmax=prior_upper_bound,
                size=len(param_dict["spin"]),
            )
        elif prior_type == "uniform":
            spin_params = parameter.Uniform(
                prior_lower_bound, prior_upper_bound, size=len(param_dict["spin"])
            )
        else:
            raise ValueError(
                "prior_type can only be uniform or bounded-normal, not ", prior_type
            )
    else:
        spin_params = None
    if len(param_dict["kep"]) != 0:
        if prior_type == "bounded-normal":
            kep_params = BoundedNormal(
                mu=0.0,
                sigma=prior_sigma,
                pmin=prior_lower_bound,
                pmax=prior_upper_bound,
                size=len(param_dict["kep"]),
            )
        elif prior_type == "uniform":
            kep_params = parameter.Uniform(
                prior_lower_bound, prior_upper_bound, size=len(param_dict["kep"])
            )
        else:
            raise ValueError(
                "prior_type can only be uniform or bounded-normal, not ", prior_type
            )
    else:
        kep_params = None
    if len(param_dict["gr"]) != 0:
        if prior_type == "bounded-normal":
            gr_params = BoundedNormal(
                mu=0.0,
                sigma=prior_sigma,
                pmin=prior_lower_bound,
                pmax=prior_upper_bound,
                size=len(param_dict["gr"]),
            )
        elif prior_type == "uniform":
            gr_params = parameter.Uniform(
                prior_lower_bound, prior_upper_bound, size=len(param_dict["gr"])
            )
        else:
            raise ValueError(
                "prior_type can only be uniform or bounded-normal, not ", prior_type
            )
    else:
        gr_params = None

    # timing model
    tm_func = tm_delay(
        param_dict=param_dict,
        pos_params=pos_params,
        pm_params=pm_params,
        spin_params=spin_params,
        kep_params=kep_params,
        gr_params=gr_params,
    )
    tm = deterministic_signals.Deterministic(tm_func, name="timing_model")

    return tm
