# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
from enterprise.signals import deterministic_signals, parameter, signal_base

# timing model delay


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

    # remmeber to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params[:, 0])))))

    # Return the time-series for the pulsar
    return new_res - residuals

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
