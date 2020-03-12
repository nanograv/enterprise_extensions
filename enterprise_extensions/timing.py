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


def get_default_physical_tm_priors():
    """
    "RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"
    "PMDEC", "PMRA", "PMELONG", "PMELAT", "PMRV", "PMBETA", "PMLAMBDA"
    "F", "F0", "F1", "F2", "P", "P1","PB","T0","A1","OM","EPS1","EPS2",
    "EPS1DOT","EPS2DOT","FB","MTOT","M2","XDOT","X2DOT","EDOT","H3",
    "H4","OMDOT","OM2DOT","XOMDOT","PBDOT","XPBDOT","GAMMA","PPNGAMMA",
    "DR","DTHETA"
    """
    default_tm_priors = {}
    default_tm_priors["E"] = {"pmin": 0.0, "pmax": 1.0}
    default_tm_priors["ECC"] = {"pmin": 0.0, "pmax": 1.0}
    default_tm_priors["SINI"] = {"pmin": 0.0, "pmax": 1.0}
    return


def get_astrometric_priors():
    astrometric_priors = {}


def get_par_errors(t2psr, par):
    """
    Prevents nans in errors for some pulsars

    :param psr: pulsar to pull error for
    :param par: parameter to pull error from par file
    """
    filename = t2psr.parfile.split("/")[-1]
    file = glob.glob("../../*/*/" + filename)[0]
    with open(file, "r") as f:
        for line in f.readlines():
            if par == "ELONG":
                # enterprise renames LAMBDA
                if line.split()[0] in [par, "LAMBDA"]:
                    error = line.split()[-1]
                    return error
            elif par == "ELAT":
                # enterprise renames BETA
                if line.split()[0] in [par, "BETA"]:
                    error = line.split()[-1]
                    return error
            else:
                if line.split()[0] == par:
                    error = line.split()[-1]
                    return error
                else:
                    raise ValueError(par, " not in file!")


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
    error_pos = {}
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
            # Section because there are incorrect handlings of errors for ecliptic coordinates, idk why
            if tm_param in ["ELONG", "LAMBDA"]:
                error_pos["ELONG"] = {
                    "err": get_par_errors(t2pulsar, tm_param),
                    "param_iter": i,
                }
            elif tm_param in ["ELAT", "BETA"]:
                error_pos["ELAT"] = {
                    "err": get_par_errors(t2pulsar, tm_param),
                    "param_iter": i,
                }

            if tm_param in ["ELONG", "LAMBDA", "ELAT", "BETA"] and error_pos.keys() >= {
                "ELAT",
                "ELONG",
            }:
                ec_errors = ephem.Ecliptic(
                    error_pos["ELONG"]["err"], error_pos["ELAT"]["err"]
                )

                tm_params_rescaled["ELONG"] = (
                    normed_params[error_pos["ELONG"]["param_iter"]]
                    * np.double(ec_errors.lon)
                    + tmparams_orig["ELONG"][0]
                )
                tm_params_rescaled["ELAT"] = (
                    normed_params[error_pos["ELAT"]["param_iter"]]
                    * np.double(ec_errors.lat)
                    + tmparams_orig["ELAT"][0]
                )
                # End of handling section
                """
                for key in error_pos.keys():
                    print(key,': ')
                    print(' Original Value: ', orig_params[key])
                    print(' normed_params: ', normed_params[error_pos[key]["param_iter"]])
                    if key == "ELONG":
                        print(' tmparam Errors: ',np.double(ec_errors.lon))
                    else:
                        print(' tmparam Errors: ',np.double(ec_errors.lat))
                    print(' tmparam Value: ',tmparams_orig[key][0])
                    print(' New Value: ',tm_params_rescaled[key])
                """
            else:
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
                # Making sanity checks
                if tm_param in ["E", "ECC"]:  # ,"SINI"]:
                    if tm_params_rescaled[tm_param] <= 0.0:
                        tm_params_rescaled[tm_param] = 1e-9
                    elif tm_params_rescaled[tm_param] >= 1.0:
                        tm_params_rescaled[tm_param] = 1.0 - 1e-9
                """
                if tm_param not in ["ELONG","ELAT"]:
                    print(tm_param,': ')
                    print(' Original Value: ', orig_params[tm_param])
                    print(' normed_params: ', normed_params)
                    print(' tmparam Errors: ',tmparams_orig[tm_param][1])
                    print(' tmparam Value: ',tmparams_orig[tm_param][0])
                    print(' New Value: ',tm_params_rescaled[tm_param])
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
    param_dict = defaultdict(list)
    for par in tmparam_list:
        if par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]:
            param_dict["pos"].append(par)
        elif par in [
            "PMDEC",
            "PMRA",
            "PMELONG",
            "PMELAT",
            "PMRV",
            "PMBETA",
            "PMLAMBDA",
        ]:
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
