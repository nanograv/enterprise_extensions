from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl
import json

#from enterprise_extensions import models
import enterprise_cw_funcs_from_git as models

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from . import Fe_statistic

def get_similarity_matrix(pta, params_list):
    phiinvs = pta.get_phiinv([], logdet=False)
    TNTs = pta.get_TNT([])
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag([])
    Nmats = [ Fe_statistic.make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

    n_sources = len(params_list)    

    res_model = [pta.get_delay(x) for x in params_list]

    S = np.zeros((n_sources,n_sources))
    for idx, (psr, Nmat, TNT, phiinv, T) in enumerate(zip(pta.pulsars, Nmats,
                                                          TNTs, phiinvs, Ts)):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)    
        for i in range(n_sources):
            for j in range(n_sources):
                delay_i = res_model[i][idx]
                delay_j = res_model[j][idx]
                S[i,j] += Fe_statistic.innerProduct_rr(delay_i, delay_j, Nmat, T, Sigma)
    return S

def get_match_matrix(pta, params_list):
    S = get_similarity_matrix(pta, params_list)
    
    M = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            M[i,j] = S[i,j]/np.sqrt(S[i,i]*S[j,j])
    return M
