"""Module for running full outlier analysis for single pulsar using
HMC. The function `run_outlier` will perform the full analysis
and save plots to a desired output directory. This code follows the example
from the accompanying Jupyter notebook `outlier-notebook.ipynb`"""

# Generic imports
import os, sys, glob, tempfile, pickle
import numpy as np
import scipy.linalg as sl, scipy.optimize as so
import matplotlib.pyplot as plt
import numdifftools as nd
import corner

# The actual outlier code
import .interval as itvl
from .nutstrajectory import nuts6


def poutlier(p,likob):
    """Invoked on a sample parameter set and the appropriate likelihood,
    returns the outlier probability (a vector over the TOAs) and
    the individual sqrt(chisq) values"""
    
    # invoke the likelihood
    _, _ = likob.base_loglikelihood_grad(p)

    # get the piccard pulsar object
    # psr = likob.psr

    r = likob.detresiduals
    N = likob.Nvec

    Pb = likob.outlier_prob # a priori outlier probability for this sample
    P0 = likob.P0           # width of outlier range
    
    PA = 1.0 - Pb
    PB = Pb
    
    PtA = np.exp(-0.5*r**2/N) / np.sqrt(2*np.pi*N)
    PtB = 1.0/P0
    
    num = PtB * PB
    den = PtB * PB + PtA * PA
    
    return num/den, r/np.sqrt(N)


def run_outlier(pintpsr, outdir='', Nsamples=20000, Nburnin=1000):
    """Run full outlier analysis for given pulsar
    
    :param pintpsr: enterprise PintPulsar object
    :param outdir: Desired output directory for chains and
        plots, default is current working directory
    :param Nsamples: Number of samples to generate with HMC, default is 20000
    :param Nburnin: Number of samples for HMC burn-in phase, default is 1000
    """
    
    # Extract pulsar name and load Interval Likelihood object
    psr = pintpsr.name
    likob = itvl.Interval(pintpsr)
    
    # Now get an approximate maximum of the posterior
    def func(x):
        ll, _ = likob.full_loglikelihood_grad(x)
        
        return -np.inf if np.isnan(ll) else ll

    def jac(x):
        _, j = likob.full_loglikelihood_grad(x)
        return j
    
    # Compute the approximate max and save to a pickle file
    endpfile = outdir + psr + '-endp.pickle'
    if not os.path.isfile(endpfile):
        endp = likob.pstart
        for iter in range(3):
            res = so.minimize(lambda x: -func(x),
                              endp,
                              jac=lambda x: -jac(x),
                              hess=None,
                              method='L-BFGS-B', options={'disp': True})

            endp = res['x']
        pickle.dump(endp,open(endpfile,'wb'))
    else:
        endp = pickle.load(open(endpfile,'rb'))
    
    # Next to whiten the likelihood, compute the Hessian of the posterior
    nhyperpars = likob.ptadict[likob.pname + '_outlierprob'] + 1
    hessfile = outdir + psr + '-fullhessian.pickle'
    if not os.path.isfile(hessfile):
        reslice = np.arange(0,nhyperpars)

        def partfunc(x):
            p = np.copy(endp)
            p[reslice] = x
            return likob.full_loglikelihood_grad(p)[0]

        ndhessdiag = nd.Hessdiag(func)
        ndparthess = nd.Hessian(partfunc)

        # Create a good-enough approximation for the Hessian
        nhdiag = ndhessdiag(endp)
        nhpart = ndparthess(endp[reslice])
        fullhessian = np.diag(nhdiag)
        fullhessian[:nhyperpars,:nhyperpars] = nhpart
        pickle.dump(fullhessian,open(hessfile,'wb'))
    else:
        fullhessian = pickle.load(open(hessfile,'rb'))
    
    # Whiten the likelihood and starting parameter vector
    wl = itvl.whitenedLikelihood(likob, endp, -fullhessian)
    likob.pstart = endp
    wlps = wl.forward(endp)
    
    # Set up and run HMC sampler
    chaindir = outdir + 'chains_' + psr
    if not os.path.exists(chaindir):
        os.makedirs(chaindir)
    chainfile = chaindir + '/samples.txt'
    if not os.path.isfile(chainfile) or len(open(chainfile,'r').readlines()) < 19999:
        # Run NUTS for 20000 samples, with a burn-in of
        # 1000 samples (target acceptance = 0.6)
        samples, lnprob, epsilon = nuts6(wl.loglikelihood_grad, Nsamples, Nburnin,
                                         wlps, 0.6,
                                         verbose=True,
                                         outFile=chainfile,
                                         pickleFile=chaindir + '/save')
    
    #------------POST PROCESSING------------
    
    # Undo all the coordinate transformations and save samples to file
    parsfile = outdir + psr + '-pars.npy'
    if not os.path.isfile(parsfile):
        samples = np.loadtxt(chaindir + '/samples.txt')
        fullsamp = wl.backward(samples[:,:-2])
        funnelsamp = likob.backward(fullsamp)
        pars = likob.multi_full_backward(funnelsamp)
        np.save(parsfile,pars)
    else:
        pars = np.load(parsfile)
    
    # Corner plot of the hyperparameter posteriors
    parnames = list(likob.ptadict.keys())
    if not os.path.isfile(outdir + psr + '-corner.pdf'):
        corner.corner(pars[:,:nhyperpars], labels=parnames[:nhyperpars], show_titles=True);
        plt.savefig(outdir + psr + '-corner.pdf')
    
    # Array of outlier probabilities
    pobsfile = outdir + psr + '-pobs.npy'
    if not os.path.isfile(pobsfile):
        nsamples = len(pars)
        nobs = len(likob.Nvec)

        # basic likelihood
        lo = likob

        outps = np.zeros((nsamples,nobs),'d')
        sigma = np.zeros((nsamples,nobs),'d')

        for i,p in enumerate(pars):
            outps[i,:], sigma[i,:] = poutlier(p,lo)

        out = np.zeros((nsamples,nobs,2),'d')
        out[:,:,0], out[:,:,1] = outps, sigma
        np.save(pobsfile,out)
    else:
        out = np.load(pobsfile)
        outps, sigma = out[:,:,0], out[:,:,1]
        
    avgps = np.mean(outps,axis=0)
    medps = np.median(outps,axis=0)
    
    # Residual plot with outliers highlighted
    spd = 86400.0   # seconds per day
    T0 = 53000.0        # reference MJD
    
    residualplot = psr + '-residuals.pdf'

    if not os.path.isfile(residualplot):
        outliers = medps > 0.1
        nout = np.sum(outliers)
        nbig = nout
        
        print("Big: {}".format(nbig))
        
        if nout == 0:
            outliers = medps > 5e-4
            nout = np.sum(outliers)
        
        print("Plotted: {}".format(nout))

        plt.figure(figsize=(15,6))

        psrobj = likob.psr

        # convert toas to mjds
        toas = psrobj.toas/spd + T0

        # red noise at the starting fit point
        _, _ = likob.full_loglikelihood_grad(endp)
        rednoise = psrobj.residuals - likob.detresiduals

        # plot tim-file residuals (I think)
        plt.errorbar(toas,psrobj.residuals,yerr=psrobj.toaerrs,fmt='.',alpha=0.3)

        # red noise
        # plt.plot(toas,rednoise,'r-')

        # possible outliers
        plt.errorbar(toas[outliers],psrobj.residuals[outliers],yerr=psrobj.toaerrs[outliers],fmt='rx')

        plt.savefig(residualplot)
    
    # Text file with exact indices of outlying TOAs and their
    # outlier probabilities
    outlier_indices = 'outliers.txt'
    with open(outlier_indices, 'w') as f:
        for ii, elem in enumerate(outliers):
            if elem:
                f.write('TOA Index {}: Outlier Probability {}\n'.format(likob.isort_dict[ii], medps[ii]))
    
    return
