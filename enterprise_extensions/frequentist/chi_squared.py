# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl


def get_chi2(pta, xs):
    """Compute generalize chisq for pta:
        chisq = y^T (N + F phi F^T)^-1 y
              = y^T N^-1 y - y^T N^-1 F (F^T N^-1 F + phi^-1)^-1 F^T N^-1 y
    """

    params = xs if isinstance(xs, dict) else pta.map_params(xs)

    # chisq = y^T (N + F phi F^T)^-1 y
    #       = y^T N^-1 y - y^T N^-1 F (F^T N^-1 F + phi^-1)^-1 F^T N^-1 y

    TNrs = pta.get_TNr(params)
    TNTs = pta.get_TNT(params)
    phiinvs = pta.get_phiinv(params, logdet=True, method='cliques')

    chi2 = np.sum(ell[0] for ell in pta.get_rNr_logdet(params))

    if pta._commonsignals:
        raise NotImplementedError("get_chi2 does not support correlated signals")
    else:
        for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
            if TNr is None:
                continue

            phiinv, _ = pl
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            try:
                cf = sl.cho_factor(Sigma)
                expval = sl.cho_solve(cf, TNr)
            except sl.LinAlgError:  # pragma: no cover
                return -np.inf

            chi2 = chi2 - np.dot(TNr, expval)

    return chi2


def get_reduced_chi2(pta, xs):
    """
    Compute Generalized Reduced Chi Square for PTA using degrees of freedom
    (DOF), defined by dof= NTOAs - N Timing Parameters - N Model Params.
    """
    keys = [ky for ky in pta._signal_dict.keys() if 'timing_model' in ky]
    chi2 = get_chi2(pta, xs)
    degs = np.array([pta._signal_dict[ky].get_basis().shape for ky in keys])
    dof = np.sum(degs[:, 0]) - np.sum(degs[:, 1])
    dof -= len(pta.param_names)
    return chi2/dof
