cimport numpy as np
import numpy as np
cimport scipy.linalg as sl
import scipy.linalg as sl

cdef extern from "cython_dL_update_omp.c":
    void dL_update_hmc2(double *pdL, double *pdLi, double *pdp, double *pdM, double *pdtj, int N) 

cdef void cython_dL_update_hmc2(
        np.ndarray[np.double_t,ndim=2] L,
        np.ndarray[np.double_t,ndim=2] Li,
        np.ndarray[np.double_t,ndim=1] p,
        np.ndarray[np.double_t,ndim=2] M,
        np.ndarray[np.double_t,ndim=1] tj):
    
    #cdef np.ndarray[np.double_t,ndim=2] Li = \
    #    np.ascontiguousarray(sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True))
    cdef int N = len(p)

    assert L.shape[0] == L.shape[1]
    
    assert L.shape[0] == len(p)

    dL_update_hmc2(&L[0,0], &Li[0,0], &p[0], &M[0,0], &tj[0], N)

def cython_dL_update_omp(L, Li, p):
    M = np.zeros_like(L, order='C')
    tj = np.zeros(len(L))
    cython_dL_update_hmc2(np.ascontiguousarray(L), np.ascontiguousarray(Li), p, M, tj)
    return M, tj

