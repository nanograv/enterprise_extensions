/* cython_dL_update_hmc.c
 *
 * Rutger van Haasteren, December 12 2015, Pasadena
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


/* The aggregated algorithm for use in the Hamiltonian Sampler */
void dL_update_hmc2(const double *pdL, const double *pdLi, const double *pdp,
                          double *pdM, double *pdtj, const int N) {
    /*
    Formal derivative of rank-one update of Cholesky decomposition,
    adjusted to perform all rank-one updates at once for the derivative
    
    L'L'^{T} = LL^{T} + diag(B)
    dL' = L Phi(L^{-1} dB L^{-T})  With Phi the utril function

    B = B(x)

    We need: dot(d_L_d_x, p), and trace(L^{-1} d_L_d_x)

    Assuming we know dB/dx, we can get d_L_d_x from the chain-rule, using
    d_L_d_B. The output of this function lets us do that:
    dot(d_L_d_x, p) = dot(M, d_B_d_x)
    trace(L^{-1} d_L_d_x) = dot(tj, d_B_d_x)
    
    Re-parameterized: also works in the limit where a->0

    :param pdL:     Current updated Cholesky decomposition (L-prime)
    :param pdLi:    Inverse of Cholesky decomposition (L^{-1})
    :param pdp:     Vector we'll need to multiply dL with
    :param pdM:     The return matrix M   (output)
    :param pdtj:    The return vector tj  (output)
    :param N:       Size of all the objects
    */
    double *pdLdot, *pdU, *pdLtrans;
    double r, drdot, dcdot, ds, temp;
    int i, j, k, index;

    //const int maxthreads = omp_get_max_threads();

    /* Allocate memory for dL transpose */
    pdLtrans = malloc(N*N*sizeof(double));

    /* Set the input matrices to zero (is quick), and transpose L */
    for(i=0; i<N; ++i) {
        for(j=0; j<N; ++j) {
            pdM[j+N*i] = 0.0;
            pdLtrans[j+N*i] = pdL[i+N*j];
        } /* for j */
        pdtj[i] = 0.0;
    } /* for i */

#pragma omp parallel private(i, j, k, index, pdLdot, pdU, r, drdot, dcdot, ds, temp) shared(pdL, pdLtrans, pdLi, pdp, pdM, pdtj) default(none)
    {
        //const int nthreads = omp_get_num_threads();
        const int ithread = omp_get_thread_num();
        double *pdMlocal, dtjlocal;
        pdMlocal = calloc(N, sizeof(double));

        //printf("In thread %i of %i\n", ithread, nthreads);

        /* The index i represents the basis vector we are working with */
#pragma omp for nowait // schedule(dynamic)
        for(i=0; i<N; ++i) {
            /* Allocate memory inside the parallel region */
            pdLdot = calloc(N, sizeof(double));  /* columns of Ldot are stored only */
            pdU = calloc(N, sizeof(double));     /* basis vector we are updating */

            /* Initialize all our quantities */
            pdU[i] = 1.0;
            temp = 0.0;
            dtjlocal = 0.0;

            /* The index k represents the row of Ldot we are working with */
            for(k=0; k<N; ++k) {
                r = pdL[k+N*k];
                
                /* Initialize the vector quantities */
                drdot = 0.5*pdU[k]*pdU[k] / r;
                dcdot = drdot/pdL[k+N*k];
                ds = pdU[k] / pdL[k+N*k];

                /* Clear Ldot data */
                if(k > 0) {
                    pdLdot[k-1] = 0.0;
                } /* if k */
                pdLdot[k] = drdot;

                /* Update Ldot */
                /* The index j represents the column of Ldot we are working with */
                for(j=k+1; j<N; ++j) {
                    /* Using the transpose of pdL is faster */
                    //pdLdot[j] = ds*pdU[j] - dcdot * pdL[k+N*j];
                    pdLdot[j] = ds*pdU[j] - dcdot * pdLtrans[j+N*k];
                } /* for j */

                /* Update U */
                for(j=k+1; j<N; ++j) {
                    /* Using the transpose of pdL is faster */
                    //pdU[j] = pdU[j] - ds*pdL[k+N*j];
                    pdU[j] = pdU[j] - ds*pdLtrans[j+N*k];
                } /* for j */

                /* Update M */
                temp = 0;
                for(j=k; j<N; ++j) {
                    temp += pdLdot[j]*pdp[j];
                } /* for j */
                //pdM[i+N*k] += temp;
                pdMlocal[k] = temp;

                /* Update tj */
                temp = 0;
                for(j=0; j<N; ++j) {
                    temp += pdLi[j+N*k]*pdLdot[j];
                } /* for j */
                //pdtj[i] += temp;
                dtjlocal += temp;

            } /* for k */

            /* How do I update pdM and pdtj FAST????? */
            /* Depends on the compiler flags!! */
#pragma omp critical
            {
                for(k=0; k<N; ++k) {
                    index = i+N*k;

                    /* Doing this is FAST */
                    /* pdM[index] = 1.337; */

                    /* But instead this, is SLOW */
                    pdM[index] = pdMlocal[k];
                    //pdM[index] = 1.337;
                } /* for k */

                /* Doing this is FAST */
                /* pdtj[i] += 1.445; */

                /* But instead this, is SLOW */
                pdtj[i] += dtjlocal;
                //pdtj[i] += 1.445;
            }

            /* Free memory of parallel regions */
            free(pdLdot);
            free(pdU);
        } /* for i */

        free(pdMlocal);

    } /* pragma omp parallel */

    free(pdLtrans);
    return;
} /* dL_update_hmc */

