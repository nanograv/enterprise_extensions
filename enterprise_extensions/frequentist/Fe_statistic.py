from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl

from enterprise_extensions import models

class FeStat(object):
    """
    Class for the Fe-statistic.
    :param psrs: List of `enterprise` Pulsar instances.
    """
    
    def __init__(self, psrs, params=None, wideband=False):
        
        # initialize standard model with fixed white noise and powerlaw red noise
        print('Initializing the model...')
        self.pta = models.model_cw(psrs, noisedict=params, rn_psd='powerlaw',
                                   ecc=False, psrTerm=False,
                                   bayesephem=False, wideband=wideband)
            
        self.psrs = psrs
        self.params = params
                                   
        self.Nmats = None



    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.pta.get_TNT(self.params)
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        #Get noise parameters for pta toaerr**2
        Nvecs = self.pta.get_ndiag(self.params)
        #Get the basis matrix
        Ts = self.pta.get_basis(self.params)
        
        Nmats = [ make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]
        
        return Nmats
