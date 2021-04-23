import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt


def cython_block_shermor_0D( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                Nx[ii] -= beta * nir * ni[ii]

    return Nx

def cython_block_shermor_0D_ld( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                Nx[ii] -= beta * nir * ni[ii]

    return Jldet, Nx


def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            rblock = r[Uinds[cc,0]:Uinds[cc,1]]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            xNx -= beta * np.dot(rblock, niblock)**2
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx

def cython_block_shermor_1D( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir
    
    return Jldet, xNx



# Proposals for calculating the Z.T * N^-1 * Z combinations
def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    @param Z:       The design matrix, array (n x m)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    zNz = np.dot(Z.T*ni, Z)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, zNz

def cython_block_shermor_2D( \
        np.ndarray[np.double_t,ndim=2] Z, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    @param Z:       The design matrix, array (n x m)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(len(Nvec), 'd')
    cdef np.ndarray[np.double_t,ndim=2] zNz

    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    for cc in range(rows):
        Jldet += log(Nvec[cc])

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            nisum = 0.0
            for ii in range(len(niblock)):
                nisum += niblock[ii]

            beta = 1.0 / (nisum+1.0/Jvec[cc])
            Jldet += log(Jvec[cc]) - log(beta)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)

    return Jldet, zNz

# Proposals for calculating the Z.T * N^-1 * Z2 combinations
def python_block_shermor_2D_asymm(Z, Z2, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    @param Z:       The design matrix, array (n x m)
    @param Z2:      The second design matrix, array (n x m2)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    zNz = np.dot(Z.T*ni, Z2)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            Zblock2 = Z2[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            zn = np.dot(niblock, Zblock)
            zn2 = np.dot(niblock, Zblock2)
            zNz -= beta * np.outer(zn.T, zn2)
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, zNz

def python_draw_ecor(r, Nvec, Jvec, Uinds):
    """
    Given Jvec, draw new epoch-averaged residuals

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    
    rv = np.random.randn(len(Jvec)) * np.sqrt(Jvec)
    ni = 1.0 / Nvec

    for cc in range(len(Jvec)):
        rblock = r[Uinds[cc,0]:Uinds[cc,1]]
        niblock = ni[Uinds[cc,0]:Uinds[cc,1]]
        beta = 1.0 / np.einsum('i->', niblock)

        rv[cc] += beta * np.dot(rblock, niblock)

    return rv

def cython_draw_ecor( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Given Jvec, draw new epoch-averaged residuals

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double ji, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] rv = np.random.randn(cols)

    for cc in range(cols):
        rv[cc] *= sqrt(Jvec[cc])

    ni = 1.0 / Nvec

    for cc in range(cols):
        ji = 1.0 / Jvec[cc]

        nir = 0.0
        nisum = 0.0
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            nisum += ni[ii]
            nir += r[ii]*ni[ii]

        rv[cc] += nir / nisum

    return rv


def cython_shermor_draw_ecor( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Do both the Sherman-Morrison block-inversion for Jitter,
    and the draw of the ecor parameters together (Cythonized)

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: r.T * N^-1 * r, log(det(N)), Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] rv = np.random.randn(cols)

    ni = 1.0 / Nvec

    for cc in range(cols):
        rv[cc] *= sqrt(Jvec[cc])

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        nir = 0.0
        nisum = 0.0
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            nisum += ni[ii]
            nir += r[ii]*ni[ii]

        rv[cc] += nir / nisum

        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir

    return Jldet, xNx, rv


def cython_update_ea_residuals( \
        np.ndarray[np.double_t,ndim=1] gibbsresiduals, \
        np.ndarray[np.double_t,ndim=1] gibbssubresiduals, \
        np.ndarray[np.double_t,ndim=1] eat, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Given epoch-averaged residuals, update the residuals, and the subtracted
    residuals, so that these can be further processed by the other conditional
    probability density functions.

    @param gibbsresiduals:      The timing residuals, array (n)
    @param gibbssubresiduals:   The white noise amplitude, array (n)
    @param eat:                 epoch averaged residuals (k)
    @param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc

    for cc in range(Uinds.shape[0]):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            gibbssubresiduals[ii] += eat[cc]
            gibbsresiduals[ii] -= eat[cc]

    return gibbsresiduals, gibbssubresiduals


def cython_Uj(np.ndarray[np.double_t,ndim=1] j, \
        np.ndarray[np.int_t,ndim=2] Uinds, nobs):
    """
    Given epoch-averaged residuals (j), get the residuals.
    Used in 'updateDetSources'

    @param j:                   epoch averaged residuals (k)
    @param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)
    @param nobs:                Number of observations (length return vector)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc
    cdef np.ndarray[np.double_t,ndim=1] Uj = np.zeros(nobs, 'd')

    for cc in range(k):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            Uj[ii] += j[cc]

    return Uj

def cython_UTx(np.ndarray[np.double_t,ndim=1] x, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Given residuals (x), get np.dot(U.T, x)
    Used in 'updateDetSources'

    @param j:                   epoch averaged residuals (k)
    @param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc
    cdef np.ndarray[np.double_t,ndim=1] UTx = np.zeros(k, 'd')

    for cc in range(k):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            UTx[cc] += x[ii]

    return UTx

def cython_logdet_dN( \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=1] dNvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNp is the diagonal derivate of N wrt Np

    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param dNvec:   The white noise derivative, array (n)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nni = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni = dNvec / Nvec**2

    for cc in range(rows):
        tr += dNvec[cc] * ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum += Nni[ii]

            tr -= Nnisum / (nisum + ji)
    
    return tr

def cython_logdet_dJ( \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=1] dJvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dJp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dJp = U dJ/dJp U^{T}, with dJ/dJp the diagnal derivative of J wrt
          Jp

    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param dJvec:   The jitter derivative, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double dJldet=0.0, ji, beta, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]

            beta = 1.0 / (nisum + ji)

            dJldet += dJvec[cc]*(nisum - beta*nisum**2)
    
    return dJldet

def cython_logdet_dN_dN( \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=1] dNvec1, \
        np.ndarray[np.double_t,ndim=1] dNvec2, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp1 N^{-1} dN/dNp2), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNpx is the diagonal derivate of N wrt Npx

    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param dNvec1:  The white noise derivative, array (n)
    @param dNvec2:  The white noise derivative, array (n)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum1, Nnisum2, NniNnisum, beta
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nni1 = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nni2 = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni1 = dNvec1 / Nvec**2
    Nni2 = dNvec2 / Nvec**2

    for cc in range(rows):
        tr += dNvec1[cc] * dNvec2[cc] * ni[cc]**2

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum1 = 0.0
            Nnisum2 = 0.0
            NniNnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum1 += Nni1[ii]
                Nnisum2 += Nni2[ii]
                NniNnisum += Nni1[ii]*Nni2[ii]*Nvec[ii]

            beta = 1.0 / (nisum + ji)

            tr += Nnisum1 * Nnisum2 * beta**2
            tr -= 2 * NniNnisum * beta
    
    return tr

def cython_logdet_dN_dJ( \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=1] dNvec, \
        np.ndarray[np.double_t,ndim=1] dJvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp N^{-1} dN/dJp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNp is the diagonal derivate of N wrt Np
        - dN/dJp = U dJ/dJp U^{T}, with dJ/dJp the diagnal derivative of J wrt
          Jp

    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param dNvec:   The white noise derivative, array (n)
    @param dJvec:   The white noise ecor derivative, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum, beta
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nni = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni = dNvec / Nvec**2

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum += Nni[ii]

            beta = 1.0 / (nisum + ji)

            tr += Nnisum * dJvec[cc]
            tr -= 2 * nisum * dJvec[cc] * Nnisum * beta
            tr += Nnisum * nisum**2 * dJvec[cc] *beta**2
    
    return tr

def cython_logdet_dJ_dJ( \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=1] dJvec1, \
        np.ndarray[np.double_t,ndim=1] dJvec2, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dJp1 N^{-1} dN/dJp2), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dJpx = U dJ/dJpx U^{T}, with dJ/dJpx the diagnal derivative of J wrt
          Jpx

    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param dJvec1:  The white noise derivative, array (k)
    @param dJvec2:  The white noise derivative, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, beta
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]

            beta = 1.0 / (nisum + ji)

            tr += dJvec1[cc] * dJvec2[cc] * nisum**2
            tr -= 2 * dJvec1[cc] * dJvec2[cc] * beta * nisum**3
            tr += dJvec1[cc] * dJvec2[cc] * beta**2 * nisum**4

    return tr
