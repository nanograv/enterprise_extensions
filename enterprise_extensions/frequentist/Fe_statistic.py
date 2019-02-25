from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl
import json

from enterprise_extensions import models
from enterprise.signals import utils

class FeStat(object):
    """
    Class for the Fe-statistic.
    :param psrs: List of `enterprise` Pulsar instances.
    """
    
    def __init__(self, psrs, params=None):
        
        # initialize standard model with fixed white noise and powerlaw red noise
        print('Initializing the model...')
        self.pta = models.model_cw(psrs, noisedict=params, rn_psd='powerlaw',
                                   ecc=False, psrTerm=False,
                                   bayesephem=False, wideband=False)        
            
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

    def compute_Fe(self, f0, gw_theta, gw_phi, brave=False):
        """
        Computes the Fe-statistic.
        :param f0: GW frequency
        :param theta: first sky cordinate (=pi/2-DEC)
        :param phi: second sky coordinate (=RA)
        :returns:
        fstat: value of the Fe-statistic
        """
        
        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        
        if self.Nmats == None:
            
            self.Nmats = self.get_Nmats()
        
        N = np.zeros(4)
        M = np.zeros((4,4))
        
        for psr, Nmat, TNT, phiinv, T in zip(self.psrs, self.Nmats,
                                             TNTs, phiinvs, Ts):
            
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            
            ntoa = len(psr.toas)
            
            F_p, F_c, _ = utils.create_gw_antenna_pattern(psr.pos, gw_theta, gw_phi)
            

            A = np.zeros((4, ntoa))
            A[0, :] = F_p / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * psr.toas)
            A[1, :] = F_p / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * psr.toas)
            A[2, :] = F_c / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * psr.toas)
            A[3, :] = F_c / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * psr.toas)

            ip1 = innerProduct_rr(A[0, :], psr.residuals, Nmat, T, Sigma, brave=brave)
            ip2 = innerProduct_rr(A[1, :], psr.residuals, Nmat, T, Sigma, brave=brave)
            ip3 = innerProduct_rr(A[2, :], psr.residuals, Nmat, T, Sigma, brave=brave)
            ip4 = innerProduct_rr(A[2, :], psr.residuals, Nmat, T, Sigma, brave=brave)

            #print(psr.name)            
            #print(np.dot(psr.residuals, np.dot(Nmat, psr.residuals)))
            
            N += np.array([ip1, ip2, ip3, ip4])
                                  
            # define M matrix M_ij=(A_i|A_j)
            for jj in range(4):
                for kk in range(4):
                    M[jj, kk] += innerProduct_rr(A[jj, :], A[kk, :], Nmat, T, Sigma, brave=brave)

        # take inverse of M
        Minv = np.linalg.pinv(M)
        fstat = 0.5 * np.dot(N, np.dot(Minv, N))

        return fstat


def innerProduct_rr(x, y, Nmat, Tmat, Sigma, TNx=None, TNy=None, brave=False):
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
    
    if brave:
        cf = sl.cho_factor(Sigma, check_finite=False)
        SigmaTNy = sl.cho_solve(cf, TNy, check_finite=False)
    else:
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
