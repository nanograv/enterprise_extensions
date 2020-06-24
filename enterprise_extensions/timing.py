# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os, glob, ephem
import numpy as np
from collections import defaultdict
from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from scipy.stats import truncnorm


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


def get_prior(
    prior_type, prior_sigma, prior_lower_bound, prior_upper_bound, num_params=None
):
    """
    Returns the requested prior for a parameter
    
    :param prior_type: prior on timing parameters.
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    :param num_params: number of timing parameters assigned to prior. Default is None (ie. only one)
    """
    if prior_type == "bounded-normal":
        return BoundedNormal(
            mu=0.0,
            sigma=prior_sigma,
            pmin=prior_lower_bound,
            pmax=prior_upper_bound,
            size=num_params,
        )
    elif prior_type == "uniform":
        return parameter.Uniform(prior_lower_bound, prior_upper_bound, size=num_params)
    else:
        raise ValueError(
            "prior_type can only be uniform or bounded-normal, not ", prior_type
        )


def get_par_errors(t2psr, par):
    """
    Prevents nans in errors for some pulsars

    :param psr: pulsar to pull error for
    :param par: parameter to pull error from par file
    """
    filename = t2psr.parfile.split("/")[-1]
    file = glob.glob("../../*/par/" + filename)[0]

    with open(file, "r") as f:
        for line in f.readlines():
            if par == "ELONG":
                # enterprise renames LAMBDA
                if line.split()[0] in [par, "LAMBDA"]:
                    error = ephem.degrees(line.split()[-1])
                    return error
            elif par == "ELAT":
                # enterprise renames BETA
                if line.split()[0] in [par, "BETA"]:
                    error = ephem.degrees(line.split()[-1])
                    return error
            else:
                if line.split()[0] == par:
                    error = ephem.degrees(line.split()[-1])
                    return error
                else:
                    raise ValueError(par, " not in file!")


# timing model delay
@signal_base.function
def tm_delay(t2pulsar, tmparams_orig, **kwargs):
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
    error_pos = {}
    for tm_scaled_key, tm_scaled_val in kwargs.items():
        tm_param = tm_scaled_key.split("_")[-1]
        orig_params[tm_param] = tmparams_orig[tm_param][0]
        # Section because there are incorrect handlings of errors for ecliptic coordinates, idk why
        if tm_param in ["ELONG", "LAMBDA"]:
            error_pos["ELONG"] = get_par_errors(t2pulsar, tm_param)
        elif tm_param in ["ELAT", "BETA"]:
            error_pos["ELAT"] = get_par_errors(t2pulsar, tm_param)

        if tm_param in ["ELONG", "LAMBDA", "ELAT", "BETA"] and error_pos.keys() >= {
            "ELAT",
            "ELONG",
        }:
            ec_errors = ephem.Ecliptic(
                error_pos["ELONG"]["err"], error_pos["ELAT"]["err"]
            )

            tm_params_rescaled["ELONG"] = (
                tm_scaled_val * np.double(ec_errors.lon) + tmparams_orig["ELONG"][0]
            )
            tm_params_rescaled["ELAT"] = (
                tm_scaled_val * np.double(ec_errors.lat) + tmparams_orig["ELAT"][0]
            )

        else:
            tm_params_rescaled[tm_param] = (
                tm_scaled_val * tmparams_orig[tm_param][1] + tmparams_orig[tm_param][0]
            )
            """
            # Making sanity checks
            if tm_param in ["E", "ECC"]:  # ,"SINI"]:
                if tm_params_rescaled[tm_param] <= 0.0:
                    tm_params_rescaled[tm_param] = 1e-9
                elif tm_params_rescaled[tm_param] >= 1.0:
                    tm_params_rescaled[tm_param] = 1.0 - 1e-9
            if tm_param in ["PX","M2"]:
                if tm_params_rescaled[tm_param] <= 0.0:
                    tm_params_rescaled[tm_param] = 1e-9
            """

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
    tm_delay_kwargs = {}
    for par in tmparam_list:
        if par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]:
            key_string = "pos_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
            )

        elif par in [
            "PMDEC",
            "PMRA",
            "PMELONG",
            "PMELAT",
            "PMRV",
            "PMBETA",
            "PMLAMBDA",
        ]:
            key_string = "pm_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
            )
        elif par in ["F", "F0", "F1", "F2", "P", "P1"]:
            key_string = "spin_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
            )
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
            key_string = "kep_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
            )
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
            key_string = "gr_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
            )
        else:
            if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
                key_string = "dmx_param_" + par
                tm_delay_kwargs[key_string] = get_prior(
                    prior_type, prior_sigma, prior_lower_bound, prior_upper_bound
                )
            else:
                print(par, " is not currently a modelled parameter.")

    # timing model
    tm_func = tm_delay(**tm_delay_kwargs)
    tm = deterministic_signals.Deterministic(tm_func, name="timing_model")

    return tm
