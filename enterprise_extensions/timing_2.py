# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from collections import defaultdict
from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from scipy.stats import truncnorm


def BoundNormPrior(value, mu=0, sigma=1, pmin=-1, pmax=1):
    """Prior function for InvGamma parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    print("BoundNormPrior call")
    return truncnorm.pdf(value, loc=mu, scale=sigma, a=low, b=up)


def BoundNormSampler(mu=0, sigma=1, pmin=-1, pmax=1, size=None):
    """Sampling function for Uniform parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    print("BoundNormSampler call")
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
            return '{}: BoundedNormal({},{}, [{},{}])'.format(
                self.name, mu, sigma, pmin, pmax
            ) + ("" if self._size is None else "[{}]".format(self._size))

    return BoundedNormal


# timing model delay
@signal_base.function
def tm_delay(
    t2pulsar,
    tmparams_orig,
    param_dict,
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
    print('tm_delay!')
    #print(param_dict)
    residuals = t2pulsar.residuals()

    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_category, tm_params in param_dict.items():
        for tm_key, tm_val in tm_params.items():
            orig_params[tm_key] = tmparams_orig[tm_key][0]
            tm_params_rescaled[tm_key] = (tm_val * tmparams_orig[tm_key][1]
                + tmparams_orig[tm_key][0])
    """                
        if tm_category == "pos":
            for i, tm_param in enumerate(param_dict["pos"].keys()):
                orig_params[tm_param] = tmparams_orig[tm_param][0]
                tm_params_rescaled[tm_param] = (
                    tm_category[tm_param] * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
        elif tm_category == "pm":
            for i, tm_param in enumerate(param_dict["pm"].keys()):
                orig_params[tm_param] = tmparams_orig[tm_param][0]
                tm_params_rescaled[tm_param] = (
                    pm_params[i] * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
        elif tm_category == "spin":
            for i, tm_param in enumerate(param_dict["spin"].keys()):
                orig_params[tm_param] = tmparams_orig[tm_param][0]
                tm_params_rescaled[tm_param] = (
                    spin_params[i] * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
        elif tm_category == "kep":
            for i, tm_param in enumerate(param_dict["kep"].keys()):
                orig_params[tm_param] = tmparams_orig[tm_param][0]
                tm_params_rescaled[tm_param] = (
                    kep_params[i] * tmparams_orig[tm_param][1]
                    + tmparams_orig[tm_param][0]
                )
        elif tm_category == "gr":
            for i, tm_param in enumerate(param_dict["gr"].keys()):
                orig_params[tm_param] = tmparams_orig[tm_param][0]
                if isinstance(gr_params, (list, np.ndarray)):
                    tm_params_rescaled[tm_param] = (
                        gr_params[i] * tmparams_orig[tm_param][1]
                        + tmparams_orig[tm_param][0]
                    )
                else:
                    tm_params_rescaled[tm_param] = (
                        gr_params * tmparams_orig[tm_param][1]
                        + tmparams_orig[tm_param][0]
                    )
    """
    print(tm_params_rescaled)
    # set to new values
    t2pulsar.vals(tm_params_rescaled)
    new_res = t2pulsar.residuals()

    # remmeber to set values back to originals
    t2pulsar.vals(orig_params)

    # Return the time-series for the pulsar
    return new_res - residuals


# Model component building blocks #


def timing_block(tmparam_list=["RAJ", "DECJ", "F0", "F1", "PMRA", "PMDEC", "PX"],
    ):
    """
    Returns the timing model block of the model
    :param tmparam_list: a list of parameters to vary in the model
    """
    param_dict = {}
    for par in tmparam_list:
        if par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]:
            if "pos" not in param_dict.keys():
                param_dict["pos"] = {}   
            vars()['pos_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
            param_dict["pos"][par] = vars()['pos_params_{}'.format(par)]
        elif par in ["PMDEC", "PMRA", "PMRV", "PMBETA", "PMLAMBDA"]:
            if "pm" not in param_dict.keys():
                param_dict["pm"] = {}
            vars()['pm_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
            param_dict["pm"][par] = vars()['pm_params_{}'.format(par)]
        elif par in ["F", "F0", "F1", "F2", "P", "P1"]:
            if "spin" not in param_dict.keys():
                param_dict["spin"] = {}
            vars()['spin_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
            param_dict["spin"][par] = vars()['spin_params_{}'.format(par)]
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
            if "kep" not in param_dict.keys():
                    param_dict["kep"] = {}
            vars()['kep_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
            param_dict["kep"][par] = vars()['kep_params_{}'.format(par)]
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
            if "gr" not in param_dict.keys():
                param_dict["gr"] = {}
            vars()['gr_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
            param_dict["gr"][par] = vars()['gr_params_{}'.format(par)]
        else:
            if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
                if "dmx" not in param_dict.keys():
                    param_dict["dmx"] = {}
                vars()['dmx_params_{}'.format(par)] = BoundedNormal(mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)
                param_dict["dmx"][par] = vars()['dmx_params_{}'.format(par)]
            else:
                print(par, " is not currently a modelled parameter.")

    print(param_dict['spin']['F0'].name)
    # default 3-sigma prior above and below the parfile mean
    """if len(param_dict["pos"]) != 0:
                    for par in param_dict["pos"].keys():   
                        pos_params = BoundedNormal(
                            mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)(par)
                else:
                    pos_params = None
                if len(param_dict["pm"]) != 0:
                    for par in param_dict["pm"]:
                        pm_params = BoundedNormal(
                            mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)(par)
                else:
                    pm_params = None
                if len(param_dict["spin"]) != 0:
                    for par in param_dict["spin"]:
                        spin_params = BoundedNormal(
                            mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)(par)
                else:
                    spin_params = None
                if len(param_dict["kep"]) != 0:
                    for par in param_dict["kep"]:
                        kep_params = BoundedNormal(
                            mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)(par)
                else:
                    kep_params = None
                if len(param_dict["gr"]) != 0:
                    for par in param_dict["gr"]:
                        gr_params = BoundedNormal(
                            mu=0.0, sigma=2.0, pmin=-3.0, pmax=3.0)(par)
                else:
                    gr_params = None"""

    print('timing_block!')
    # timing model
    tm_func = tm_delay(
        param_dict=param_dict,
    )
    print('tm_func: ',tm_func)

    print(dir(tm_func))
    tm = deterministic_signals.Deterministic(tm_func, name="timing_model")
    print('tm: ',tm)
    return tm
