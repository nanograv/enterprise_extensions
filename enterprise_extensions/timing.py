# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from collections import OrderedDict
import scipy.stats as sps
from scipy.stats import truncnorm

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_signals


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


# Scipy defined RV for NE2001 DM Dist data.
defpath = os.path.dirname(__file__)
data_file = defpath + "/px_prior_1.txt"
px_prior = np.loadtxt(data_file)
px_hist = np.histogram(px_prior, bins=100, density=True)
px_rv = sps.rv_histogram(px_hist)


def get_default_physical_tm_priors():
    """
    Fills dictionary with physical bounds on timing parameters
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
    with open(astrometric_px_file, "r") as pxf:
        astrometric_px = json.load(pxf)
        pxf.close()

    return astrometric_px


def get_prior(
    prior_type,
    prior_sigma,
    prior_lower_bound,
    prior_upper_bound,
    mu=0.0,
    num_params=None,
):
    """
    Returns the requested prior for a parameter
    :param prior_type: prior on timing parameters.
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    :param mu: Sets the mean/central value of prior if bounded normal is selected
    :param num_params: number of timing parameters assigned to prior. Default is None (ie. only one)
    """
    if prior_type == "bounded-normal":
        return BoundedNormal(
            mu=mu,
            sigma=prior_sigma,
            pmin=prior_lower_bound,
            pmax=prior_upper_bound,
            size=num_params,
        )
    elif prior_type == "uniform":
        return parameter.Uniform(prior_lower_bound, prior_upper_bound, size=num_params)
    elif prior_type == "dm_dist_px_prior":
        return NE2001DMDist_Parameter(size=num_params)
    else:
        raise ValueError(
            "prior_type can only be uniform, bounded-normal, or dm_dist_px_prior, not ",
            prior_type,
        )


def filter_Mmat(psr, ltm_list=[]):
    """Filters the pulsar's design matrix of parameters
    :param psr: Pulsar object
    :param ltm_list: a list of parameters that will linearly varied, default is to vary anything not in tm_param_list
    :return: A new pulsar object with the filtered design matrix
    """
    idx_lin_pars = [psr.fitpars.index(p) for p in psr.fitpars if p in ltm_list]
    psr.fitpars = list(np.array(psr.fitpars)[idx_lin_pars])
    psr._designmatrix = psr._designmatrix[:, idx_lin_pars]
    return psr


# timing model delay
@signal_base.function
def tm_delay(t2pulsar, tm_params_orig, **kwargs):
    """
    Compute difference in residuals due to perturbed timing model.

    :param t2pulsar: libstempo pulsar object
    :param tm_params_orig: dictionary of TM parameter tuples, (val, err)

    :return: difference between new and old residuals in seconds
    """
    residuals = np.longdouble(t2pulsar.residuals().copy())
    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_scaled_key, tm_scaled_val in kwargs.items():
        if "DMX" in tm_scaled_key:
            tm_param = "_".join(tm_scaled_key.split("_")[-2:])
        else:
            tm_param = tm_scaled_key.split("_")[-1]

        if tm_param == "COSI":
            orig_params["SINI"] = np.longdouble(tm_params_orig["SINI"][0])
        else:
            orig_params[tm_param] = np.longdouble(tm_params_orig[tm_param][0])

        if "physical" in tm_params_orig[tm_param]:
            # User defined priors are assumed to not be scaled
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - tm_scaled_val ** 2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(tm_scaled_val)
        else:
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                rescaled_COSI = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - rescaled_COSI ** 2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )

    # TODO: Find a way to not do this every likelihood call bc it doesn't change and it is in enterprise.psr._isort
    # Sort residuals by toa to match with get_detres() call
    isort = np.argsort(t2pulsar.toas(), kind="mergesort")
    t2pulsar.vals(tm_params_rescaled)
    new_res = np.longdouble(t2pulsar.residuals().copy())

    # remeber to set values back to originals
    t2pulsar.vals(orig_params)

    # Return the time-series for the pulsar
    return -(new_res[isort] - residuals[isort])


# Model component building blocks #


def timing_block(
    psr,
    tm_param_list=["F0", "F1"],
    ltm_list=["Offset"],
    prior_type="uniform",
    prior_mu=0.0,
    prior_sigma=2.0,
    prior_lower_bound=-5.0,
    prior_upper_bound=5.0,
    tm_param_dict={},
    fit_remaining_pars=True,
    wideband_kwargs={},
):
    """
    Returns the timing model block of the model
    :param psr: a pulsar object on which to construct the timing model
    :param tm_param_list: a list of parameters to vary nonlinearly in the model
    :param ltm_list: a list of parameters to vary linearly in the model
    :param prior_type: the function used for the priors ['uniform','bounded-normal']
    :param prior_mu: the mean/central vlaue for the prior if ``prior_type`` is 'bounded-normal'
    :param prior_sigma: the sigma for the prior if ``prior_type`` is 'bounded-normal'
    :param prior_lower_bound: the lower bound for the prior
    :param prior_upper_bound: the upper bound for the prior
    :param tm_param_dict: a dictionary of physical parameters for nonlinearly varied timing model parameters, used to sample in non-sigma-scaled parameter space
    :param fit_remaining_pars: fits any timing model parameter in the linear regime if not in ``tm_param_list`` or ``tm_param_dict``
    :param wideband_kwargs: extra kwargs for ``gp_signals.WidebandTimingModel``
    """
    # If param in tm_param_dict not in tm_param_list, add it
    for key in tm_param_dict.keys():
        if key not in tm_param_list:
            tm_param_list.append(key)

    physical_tm_priors = get_default_physical_tm_priors()

    # Get values and errors as pulled by libstempo from par file.
    ptypes = ["normalized" for ii in range(len(psr.t2pulsar.pars()))]
    psr.tm_params_orig = OrderedDict(
        zip(
            psr.t2pulsar.pars(),
            map(
                list,
                zip(
                    np.longdouble(psr.t2pulsar.vals()),
                    np.longdouble(psr.t2pulsar.errs()),
                    ptypes,
                ),
            ),
        )
    )
    # Check to see if nan or inf in pulsar parameter errors.
    # The refit will populate the incorrect errors, but sometimes
    # changes the values by too much, which is why it is done in this order.
    orig_vals = {p: v for p, v in zip(psr.t2pulsar.pars(), psr.t2pulsar.vals())}
    orig_errs = {p: e for p, e in zip(psr.t2pulsar.pars(), psr.t2pulsar.errs())}
    if np.any(np.isnan(psr.t2pulsar.errs())) or np.any(
        [err == 0.0 for err in psr.t2pulsar.errs()]
    ):
        eidxs = np.where(
            np.logical_or(np.isnan(psr.t2pulsar.errs()), psr.t2pulsar.errs() == 0.0)
        )[0]
        psr.t2pulsar.fit()
        for idx in eidxs:
            par = psr.t2pulsar.pars()[idx]
            psr.tm_params_orig[par][1] = np.longdouble(psr.t2pulsar.errs()[idx])
    psr.t2pulsar.vals(orig_vals)
    psr.t2pulsar.errs(orig_errs)

    tm_delay_kwargs = {}
    default_prior_params = [
        prior_mu,
        prior_sigma,
        prior_lower_bound,
        prior_upper_bound,
        prior_type,
    ]
    for par in tm_param_list:
        if par == "Offset":
            raise ValueError(
                "TEMPO2 does not support modeling the phase offset: 'Offset'."
            )
        elif par in tm_param_dict.keys():
            # Overwrite default priors if new ones defined for the parameter in tm_param_dict
            if par in psr.tm_params_orig.keys():
                psr.tm_params_orig[par][-1] = "physical"
                val, err, _ = psr.tm_params_orig[par]
            elif "COSI" in par and "SINI" in psr.tm_params_orig.keys():
                print("COSI added to tm_params_orig for to work with tm_delay.")
                sin_val, sin_err, _ = psr.tm_params_orig["SINI"]
                val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                err = np.longdouble(
                    np.sqrt((sin_err * sin_val) ** 2 / (1 - sin_val ** 2))
                )
                # psr.tm_params_orig["SINI"][-1] = "physical"
                psr.tm_params_orig[par] = [val, err, "physical"]
            else:
                raise ValueError(par, "not in psr.tm_params_orig.")

            if "prior_mu" in tm_param_dict[par].keys():
                prior_mu = tm_param_dict[par]["prior_mu"]
            else:
                prior_mu = default_prior_params[0]
            if "prior_sigma" in tm_param_dict[par].keys():
                prior_sigma = tm_param_dict[par]["prior_sigma"]
            else:
                prior_sigma = default_prior_params[1]
            if "prior_lower_bound" in tm_param_dict[par].keys():
                prior_lower_bound = tm_param_dict[par]["prior_lower_bound"]
            else:
                prior_lower_bound = np.float(val + err * prior_lower_bound)
            if "prior_upper_bound" in tm_param_dict[par].keys():
                prior_upper_bound = tm_param_dict[par]["prior_upper_bound"]
            else:
                prior_upper_bound = np.float(val + err * prior_upper_bound)

            if "prior_type" in tm_param_dict[par].keys():
                prior_type = tm_param_dict[par]["prior_type"]
            else:
                prior_type = default_prior_params[4]
        else:
            prior_mu = default_prior_params[0]
            prior_sigma = default_prior_params[1]
            prior_lower_bound = default_prior_params[2]
            prior_upper_bound = default_prior_params[3]
            prior_type = default_prior_params[4]

        if par in physical_tm_priors.keys():
            if par in tm_param_dict.keys():
                if "pmin" in physical_tm_priors[par].keys():
                    if prior_lower_bound < physical_tm_priors[par]["pmin"]:
                        prior_lower_bound = physical_tm_priors[par]["pmin"]
                if "pmax" in physical_tm_priors[par].keys():
                    if prior_upper_bound > physical_tm_priors[par]["pmax"]:
                        prior_upper_bound = physical_tm_priors[par]["pmax"]
            else:
                if par in psr.tm_params_orig.keys():
                    val, err, _ = psr.tm_params_orig[par]
                else:
                    # Switch for sampling in COSI, but using SINI in libstempo
                    if "COSI" in par and "SINI" in psr.tm_params_orig.keys():
                        print("COSI added to tm_params_orig for to work with tm_delay.")
                        sin_val, sin_err, _ = psr.tm_params_orig["SINI"]
                        val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                        err = np.longdouble(
                            np.sqrt((sin_err * sin_val) ** 2 / (1 - sin_val ** 2))
                        )
                        psr.tm_params_orig[par] = [val, err, "normalized"]
                    else:
                        raise ValueError("{} not in psr.tm_params_orig".format(par))

                if (
                    "pmin" in physical_tm_priors[par].keys()
                    and "pmax" in physical_tm_priors[par].keys()
                ):
                    if val + err * prior_lower_bound < physical_tm_priors[par]["pmin"]:
                        psr.tm_params_orig[par][-1] = "physical"
                        prior_lower_bound = physical_tm_priors[par]["pmin"]

                    if val + err * prior_upper_bound > physical_tm_priors[par]["pmax"]:
                        if psr.tm_params_orig[par][-1] != "physical":
                            psr.tm_params_orig[par][-1] = "physical"
                            # Need to change lower bound to a non-normed prior too
                            prior_lower_bound = np.float(val + err * prior_lower_bound)
                        prior_upper_bound = physical_tm_priors[par]["pmax"]
                    else:
                        if psr.tm_params_orig[par][-1] == "physical":
                            prior_upper_bound = np.float(val + err * prior_upper_bound)
                elif (
                    "pmin" in physical_tm_priors[par].keys()
                    or "pmax" in physical_tm_priors[par].keys()
                ):
                    if "pmin" in physical_tm_priors[par].keys():
                        if (
                            val + err * prior_lower_bound
                            < physical_tm_priors[par]["pmin"]
                        ):
                            psr.tm_params_orig[par][-1] = "physical"
                            prior_lower_bound = physical_tm_priors[par]["pmin"]
                            # Need to change lower bound to a non-normed prior too
                            prior_upper_bound = np.float(val + err * prior_upper_bound)
                    elif "pmax" in physical_tm_priors[par].keys():
                        if (
                            val + err * prior_upper_bound
                            > physical_tm_priors[par]["pmax"]
                        ):
                            psr.tm_params_orig[par][-1] = "physical"
                            prior_upper_bound = physical_tm_priors[par]["pmax"]
                            # Need to change lower bound to a non-normed prior too
                            prior_lower_bound = np.float(val + err * prior_lower_bound)

        tm_delay_kwargs[par] = get_prior(
            prior_type, prior_sigma, prior_lower_bound, prior_upper_bound, mu=prior_mu,
        )
    # timing model
    tm_func = tm_delay(**tm_delay_kwargs)
    tm = deterministic_signals.Deterministic(tm_func, name="timing_model")

    # filter design matrix of all but linear params
    if fit_remaining_pars:
        if not ltm_list:
            ltm_list = [p for p in psr.fitpars if p not in tm_param_list]
        filter_Mmat(psr, ltm_list=ltm_list)
        if any(["DMX" in x for x in ltm_list]) and wideband_kwargs:
            ltm = gp_signals.WidebandTimingModel(
                name="wideband_timing_model", **wideband_kwargs,
            )
        else:
            ltm = gp_signals.TimingModel(coefficients=False)
        tm += ltm

    return tm
