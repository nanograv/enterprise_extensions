from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl
import scipy.special

from enterprise_extensions import models


class FpStat(object):
    """
    Class for the Fp-statistic.

    :param psrs: List of `enterprise` Pulsar instances.
    :param psrTerm: Include the pulsar term in the CW signal model. Default=True
    :param bayesephem: Include BayesEphem model. Default=True
    """
    
    def __init__(self, psrs, params=None,
                 psrTerm=True, bayesephem=True, wideband=False, pta=None):
        
        if pta is None:
        
            # initialize standard model with fixed white noise and powerlaw red noise
            print('Initializing the model...')
            self.pta = models.model_cw(psrs, noisedict=params, rn_psd='powerlaw',
                                       ecc=False, psrTerm=psrTerm,
                                       bayesephem=bayesephem, wideband=wideband)

        else:
            self.pta = pta
                    
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
    
    def compute_Fp(self, f0):
        """
        Computes the Fp-statistic.

        :param f0: GW frequency

        :returns:
        fstat: value of the Fp-statistic at the given frequency
        """
        
        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        
        if self.Nmats == None:
            
            self.Nmats = self.get_Nmats()
        
        N = np.zeros(2)
        M = np.zeros((2,2))
        fstat = 0
        
        for psr, Nmat, TNT, phiinv, T in zip(self.psrs, self.Nmats,
                                             TNTs, phiinvs, Ts):
            
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            
            ntoa = len(psr.toas)
            
            A = np.zeros((2, ntoa))
            A[0, :] = 1 / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * psr.toas)
            A[1, :] = 1 / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * psr.toas)
            
            ip1 = innerProduct_rr(A[0, :], psr.residuals, Nmat, T, Sigma)
            ip2 = innerProduct_rr(A[1, :], psr.residuals, Nmat, T, Sigma)
            N = np.array([ip1, ip2])
                                  
            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    M[jj, kk] = innerProduct_rr(A[jj, :], A[kk, :], Nmat, T, Sigma)

            # take inverse of M
            Minv = np.linalg.pinv(M)
            fstat += 0.5 * np.dot(N, np.dot(Minv, N))
    
        return fstat


def innerProduct_rr(x, y, Nmat, Tmat, Sigma, TNx=None, TNy=None):
    """
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y
        
        :param x: vector timeseries 1
        :param y: vector timeseries 2
        :param Nmat: white noise matrix
        :param Tmat: Modified design matrix including red noise/jitter
        :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
        :param TNx: T^T N^{-1} x precomputed
        :param TNy: T^T N^{-1} y precomputed
        :return: inner product (x|y)
        """
    
    # white noise term
    Ni = Nmat
    xNy = np.dot(np.dot(x, Ni), y)
    Nx, Ny = np.dot(Ni, x), np.dot(Ni, y)
    
    if TNx == None and TNy == None:
        TNx = np.dot(Tmat.T, Nx)
        TNy = np.dot(Tmat.T, Ny)
    
    cf = sl.cho_factor(Sigma)
    SigmaTNy = sl.cho_solve(cf, TNy)

    ret = xNy - np.dot(TNx, SigmaTNy)

    return ret


def make_Nmat(phiinv, TNT, Nvec, T):
    
    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    Nshape = np.shape(T)[0]
    
    TtN = Nvec.solve(other = np.eye(Nshape),left_array = T)
    
    #Put pulsar's autoerrors in a diagonal matrix
    Ndiag = Nvec.solve(other = np.eye(Nshape),left_array = np.eye(Nshape))
    
    expval2 = sl.cho_solve(cf,TtN)
    #TtNt = np.transpose(TtN)
    
    #An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T,expval2)
