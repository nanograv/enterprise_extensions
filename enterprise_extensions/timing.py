# -*- coding: utf-8 -*-

import os
import json
import copy
import numpy as np

from collections import OrderedDict
import scipy
import scipy.stats as sps
from scipy.stats import truncnorm

import enterprise.constants as const
from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_signals

from pint.residuals import Residuals


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
    prior_lower_bound=-5.0,
    prior_upper_bound=5.0,
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


def dm_funk(adjusted_dmx_epochs, dm0, dm1, dm2):
    """Used to refit for DM0, DM1, and DM2.
    :param adjusted_toas: (dmxepochs-dmepoch) measured dmx epochs - the reference epoch for DM [s]
    :param dm0: dm enterprise parameters of the constant DM offset
    :param dm1: dm enterprise parameters of the first DM derivative
    :param dm2: dm enterprise parameters of the second DM derivative
    """
    # DM(t)=DM+DM1*(t-DMEPOCH)+DM2*(t-DMEPOCH)^2
    return dm0 + dm1 * adjusted_dmx_epochs + dm2 * (adjusted_dmx_epochs ** 2)


# DM delay
@signal_base.function
def dm_delay(toas, freqs, dm0=0, dm1=0, dm2=0, dmepoch=0, **kwargs):
    """Delay in DMX model of DM variations
    :param toas: Time-of-arrival measurements [s]
    :param freqs: observing frequencies [Hz]
    :param dm0: dm enterprise parameters of the constant DM offset
    :param dm1: dm enterprise parameters of the first DM derivative
    :param dm2: dm enterprise parameters of the second DM derivative
    :param dmepoch: the reference epoch for DM [s]
    """
    dmN = dm0 + dm1*(toas-dmepoch) + dm2*(toas-dmepoch)**2
    return dmN * freqs**2 / const.DM_K / 1e12


def dm_block(psr,
             dmepoch=None,
             prior_type="normal",
             prior_sigma=2.0,
             prior_lower_bound=-5.0,
             prior_upper_bound=5.0,
             dmx_data=None
             ):
    """
    Returns the quadratic dm model block of the model
    :param psr: a pulsar object on which to construct the timing model
    :param dmepoch: the reference epoch for DM [days]
    :param prior_type: the function used for the priors ['uniform', 'normal', 'bounded-normal']
    :param prior_sigma: the sigma for the prior if ``prior_type`` is 'bounded-normal'
    :param prior_lower_bound: the lower bound for the prior
    :param prior_upper_bound: the upper bound for the prior
    """
    if dmepoch is None:
        if hasattr(psr, "model"):
            if hasattr(psr.model, "DMEPOCH"):
                dmepoch = psr.model['DMEPOCH'].value
            elif hasattr(psr.model, "PEPOCH"):
                dmepoch = psr.model['PEPOCH'].value
            elif hasattr(psr.model, "POSEPOCH"):
                dmepoch = psr.model['POSEPOCH'].value
            else:
                raise ValueError("dmepoch must be assigned.")
        else:
            raise ValueError("dmepoch must be assigned.")
    DMEPOCH = dmepoch*24*3600

    # Make sure dmx_data is sorted
    if all(dmx_data["DMXEP"] != sorted(dmx_data["DMXEP"])):
        sorted_dmx_ep = np.asarray(sorted(dmx_data["DMXEP"]))
        sorted_dmx_vals = []
        for sx in sorted(dmx_data["DMXEP"]):
            sorted_dmx_vals.append(dmx_data["DMX_value"][np.where(dmx_data["DMXEP"]==sx)][0])
        sorted_dmx_vals = np.array(sorted_dmx_vals)
    else:
        sorted_dmx_ep = dmx_data["DMXEP"]
        sorted_dmx_vals = dmx_data["DMX_value"]

    #Get the dmx value nearest to the DMEPOCH
    dmx_DMEPOCH = sorted_dmx_vals[(np.abs(sorted_dmx_ep - dmepoch)).argmin()]
    #Fit a quadratic equation (given in dm_funk) to get central dm0, dm1, and dm2 values and their errors
    #We make dmx(DMEPOCH) = 0
    popt, pcov = scipy.optimize.curve_fit(dm_funk, sorted_dmx_ep*24*3600-DMEPOCH, sorted_dmx_vals-dmx_DMEPOCH)
    perr = np.sqrt(np.diag(pcov))

    dm0 = get_prior(prior_type, perr[0]*prior_sigma, mu=popt[0],
                    prior_lower_bound=popt[0]-prior_lower_bound*perr[0],
                    prior_upper_bound=popt[0]+prior_lower_bound*perr[0])
    dm1 = get_prior(prior_type, perr[1]*prior_sigma, mu=popt[1],
                    prior_lower_bound=popt[1]-prior_lower_bound*perr[1],
                    prior_upper_bound=popt[1]+prior_lower_bound*perr[1])
    dm2 = get_prior(prior_type, perr[2]*prior_sigma, mu=popt[2],
                    prior_lower_bound=popt[2]-prior_lower_bound*perr[2],
                    prior_upper_bound=popt[2]+prior_lower_bound*perr[2])

    # dm model
    dm_func = dm_delay(psr.toas, psr.freqs, dm0=dm0, dm1=dm1, dm2=dm2, dmepoch=DMEPOCH)

    dm = deterministic_signals.Deterministic(dm_func, name="dm_model")
    return dm


# timing model delay
@signal_base.function
def tm_delay(psr, **kwargs):
    """
    Compute difference in residuals due to perturbed timing model.

    :param psr: enterprise pulsar object

    :return: difference between new and old residuals in seconds
    """

    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_scaled_key, tm_scaled_val in kwargs.items():
        if "DMX" in tm_scaled_key:
            tm_param = "_".join(tm_scaled_key.split("_")[-2:])
        else:
            tm_param = tm_scaled_key.split("_")[-1]

        if tm_param == "COSI":
            orig_params["SINI"] = np.longdouble(psr.tm_params_orig["SINI"][0])
        else:
            orig_params[tm_param] = np.longdouble(psr.tm_params_orig[tm_param][0])

        if "physical" in psr.tm_params_orig[tm_param]:
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
                    tm_scaled_val * psr.tm_params_orig[tm_param][1]
                    + psr.tm_params_orig[tm_param][0]
                )
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - rescaled_COSI ** 2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(
                    tm_scaled_val * psr.tm_params_orig[tm_param][1]
                    + psr.tm_params_orig[tm_param][0]
                )

    if hasattr(psr, 'model'):
        new_model = copy.deepcopy(psr.model)
        # Set values to new sampled values
        new_model.set_param_values(tm_params_rescaled)
        # Get new residuals
        new_res = np.longdouble(Residuals(psr.pint_toas, new_model).resids_value)
    elif hasattr(psr, 't2pulsar'):
        # Set values to new sampled values
        psr.t2pulsar.vals(tm_params_rescaled)
        # Get new residuals
        new_res = np.longdouble(psr.t2pulsar.residuals())
        # Set values back to originals
        psr.t2pulsar.vals(orig_params)
    else:
        raise ValueError('Enterprise pulsar must keep either pint or t2pulsar. Use either drop_t2pulsar=False or drop_pintpsr=False when initializing the enterprise pulsar.')

    # Return the time-series for the pulsar
    # Sort residuals by toa to match with get_detres() call
    return psr.residuals - new_res[psr.isort]

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
    :param prior_type: the function used for the priors ['uniform', 'normal', 'bounded-normal']
    :param prior_mu: the mean/central value for the prior if ``prior_type`` is 'normal' or 'bounded-normal'
    :param prior_sigma: the sigma for the prior if ``prior_type`` is 'normal' or 'bounded-normal'
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

    if not hasattr(psr, 'tm_params_orig'):
        if hasattr(psr, 'model'):
            # Get values and errors as initialized by pint.
            psr.tm_params_orig = OrderedDict()
            for par in psr.fitpars:
                if hasattr(psr.model, par):
                    psr.tm_params_orig[par]=[getattr(psr.model, par).value,
                                             getattr(psr.model, par).uncertainty_value,
                                             "normalized"]
        elif hasattr(psr, 't2pulsar'):
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
        else:
            raise ValueError('Enterprise pulsar must keep either pint or t2pulsar. Use either drop_t2pulsar=False or drop_pintpsr=False when initializing the enterprise pulsar.')

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
                print("COSI added to tm_params_orig to work with tm_delay.")
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
                prior_lower_bound = float(val + err * prior_lower_bound)
            if "prior_upper_bound" in tm_param_dict[par].keys():
                prior_upper_bound = tm_param_dict[par]["prior_upper_bound"]
            else:
                prior_upper_bound = float(val + err * prior_upper_bound)

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
                            prior_lower_bound = float(val + err * prior_lower_bound)
                        prior_upper_bound = physical_tm_priors[par]["pmax"]
                    else:
                        if psr.tm_params_orig[par][-1] == "physical":
                            prior_upper_bound = float(val + err * prior_upper_bound)
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
                            prior_upper_bound = float(val + err * prior_upper_bound)
                    elif "pmax" in physical_tm_priors[par].keys():
                        if (
                            val + err * prior_upper_bound
                            > physical_tm_priors[par]["pmax"]
                        ):
                            psr.tm_params_orig[par][-1] = "physical"
                            prior_upper_bound = physical_tm_priors[par]["pmax"]
                            # Need to change lower bound to a non-normed prior too
                            prior_lower_bound = float(val + err * prior_lower_bound)

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
