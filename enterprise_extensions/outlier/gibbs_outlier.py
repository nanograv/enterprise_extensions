import glob, os, time, sys
import numpy as np
import scipy.linalg as sl, scipy.stats, scipy.special

class OutlierGibbs(object):
    
    """Gibbs-based pulsar-timing outlier analysis.
    
    Based on:
    
        Article by Tak, Ellis, Ghosh:
        "Robust and Accurate Inference via a Mixture 
        of Gaussian and Student's t Errors",
        https://doi.org/10.1080/10618600.2018.1537925
        arXiv:1707.03057.

        Article by Wang, Taylor:
        "Controlling Outlier Contamination In Multimessenger 
        Time-domain Searches For Supermasssive Binary Black Holes",
        (In prep. 2021)

        Code from https://github.com/jellis18/gibbs_student_t
        
    Authors: 
    
        J. A. Ellis, S. R. Taylor, J. Wang
    
    Example usage:
    
        > gibbs = OutlierGibbs(pta, model='mixture', vary_df=True, 
                        theta_prior='beta', vary_alpha=True)
        > params = np.array([p.sample() for p in gibbs.params]).flatten()
        > gibbs.sample(params, outdir='./outlier/', 
                       niter=10000, resume=False)
        > poutlier = np.mean(gibbs.poutchain, axis = 0)
        # Gives marginalized outlier probability of each TOA
        
     
    """
    
    def __init__(self, pta, model='mixture', 
                 m=0.01, tdf=4, vary_df=True, 
                 theta_prior='beta', 
                 alpha=1e10, vary_alpha=True, 
                 pspin=None):
        """
        Parameters
        -----------
        pta : object
            instance of a pta object for a single pulsar
        model : str
            type of outlier model 
            [default = mixture of Gaussian and Student's t]
        tdf : int
            degrees of freedom for Student's t outlier distribution
            [default = 4]
        m : float
            a-priori proportion of observations that are outliers
            [default = 0.01]
        vary_df : boolean
            vary the Student's t degrees of freedom
            [default = True]
        theta_prior : str
            prior outlier probability
            [default = beta distribution]
        alpha : float
            relative width of outlier to inlier distribution
            [default = 1e10]
        vary_alpha : boolean
            vary the relative outlier to inlier width
            [default = True]
        pspin : float
            pulsar spin period for vvh17 model, arXiv:1609.02144
            [default = None]
        """

        self.pta = pta
        if np.any(['basis_ecorr' in key for 
                   key in self.pta._signal_dict.keys()]):
            pass
        else:
            print('ERROR: Gibbs outlier analysis must use basis_ecorr, not kernel ecorr')

        # a-priori proportion of observations that are outliers
        self.mp = m
        # a-priori outlier probability distribution
        self.theta_prior = theta_prior

        # spin period
        self.pspin = pspin

        # vary t-distribution d.o.f
        self.vary_df = vary_df

        # vary alpha
        self.vary_alpha = vary_alpha

        # For now assume one pulsar
        self._residuals = self.pta.get_residuals()[0]

        # which likelihood model
        self._lmodel = model

        # auxiliary variable stuff
        xs = [p.sample() for p in pta.params]
        self._b = np.zeros(self.pta.get_basis(xs)[0].shape[1])

        # for caching
        self.TNT = None
        self.d = None

        # outlier detection variables
        self._pout = np.zeros_like(self._residuals)
        self._z = np.zeros_like(self._residuals)
        if not vary_alpha:
            self._alpha = np.ones_like(self._residuals) * alpha
        else:
            self._alpha = np.ones_like(self._residuals)
        self._theta = self.mp
        self.tdf = tdf
        if model in ['t', 'mixture', 'vvh17']:
            self._z = np.ones_like(self._residuals)

            
    @property
    def params(self):
        ret = []
        for param in self.pta.params:
            ret.append(param)
        return ret

    
    def map_params(self, xs):
        return {par.name: x for par, 
                x in zip(self.params, xs)}


    def get_hyper_param_indices(self):
        ind = []
        for ct, par in enumerate(self.params):
            if 'ecorr' in par.name or 'log10_A' in par.name or 'gamma' in par.name:
                ind.append(ct)
        return np.array(ind)


    def get_white_noise_indices(self):
        ind = []
        for ct, par in enumerate(self.params):
            if 'efac' in par.name or 'equad' in par.name:
                ind.append(ct)
        return np.array(ind)


    def update_hyper_params(self, xs):

        # get hyper parameter indices
        hind = self.get_hyper_param_indices()

        # get initial log-likelihood and log-prior
        lnlike0, lnprior0 = self.get_lnlikelihood(xs), self.get_lnprior(xs)
        xnew = xs.copy()
        for ii in range(10):

            # standard gaussian jump (this allows for different step sizes)
            q = xnew.copy()
            sigmas = 0.05 * len(hind)
            probs = [0.1, 0.15, 0.5, 0.15, 0.1]
            sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            par = np.random.choice(hind, size=1) # only one hyper param at a time
            q[par] += np.random.randn(len(q[par])) * sigmas * scale

            # get log-like and log prior at new position
            lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

            # metropolis step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
            if diff > np.log(np.random.rand()):
                xnew = q
                lnlike0 = lnlike1
                lnprior0 = lnprior1
            else:
                xnew = xnew

        return xnew


    def update_white_params(self, xs):

        # get white noise parameter indices
        wind = self.get_white_noise_indices()

        xnew = xs.copy()
        lnlike0, lnprior0 = self.get_lnlikelihood_white(xnew), self.get_lnprior(xnew)
        for ii in range(20):

            # standard gaussian jump (this allows for different step sizes)
            q = xnew.copy()
            sigmas = 0.05 * len(wind)
            probs = [0.1, 0.15, 0.5, 0.15, 0.1]
            sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            par = np.random.choice(wind, size=1)
            q[par] += np.random.randn(len(q[par])) * sigmas * scale

            # get log-like and log prior at new position
            lnlike1, lnprior1 = self.get_lnlikelihood_white(q), self.get_lnprior(q)

            # metropolis step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
            if diff > np.log(np.random.rand()):
                xnew = q
                lnlike0 = lnlike1
                lnprior0 = lnprior1
            else:
                xnew = xnew
        return xnew

    
    def update_b(self, xs): 

        # map parameter vector
        params = self.map_params(xs)

        # start likelihood calculations
        loglike = 0

        # get auxiliaries
        Nvec = self._alpha**self._z * self.pta.get_ndiag(params)[0]
        phiinv = self.pta.get_phiinv(params, logdet=False)[0]
        residuals = self._residuals

        T = self.pta.get_basis(params)[0]
        if self.TNT is None and self.d is None:
            self.TNT = np.dot(T.T, T / Nvec[:,None])
            self.d = np.dot(T.T, residuals/Nvec)
        #d = self.pta.get_TNr(params)[0]
        #TNT = self.pta.get_TNT(params)[0]

        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)

        try:
            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, self.d)/s)
            Li = u * np.sqrt(1/s)
        except np.linalg.LinAlgError:

            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, self.d)
            u, s, _ = sl.svd(Sigi)
            Li = u * np.sqrt(1/s)

        b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

        return b


    def update_theta(self, xs):

        if self._lmodel in ['t', 'gaussian']:
            return self._theta
        elif self._lmodel in ['mixture', 'vvh17']:
            n = len(self._residuals)
            if self.theta_prior == 'beta':
                mk = n * self.mp
                k1mm = n * (1-self.mp)
            else:
                mk, k1mm = 1.0, 1.0
            # from Tak, Ellis, Ghosh (2018): k = sample size, m = 0.01
            ret = scipy.stats.beta.rvs(np.sum(self._z) + mk,
                                       n - np.sum(self._z) + k1mm)
            return ret


    def update_z(self, xs):

        # map parameters
        params = self.map_params(xs)

        if self._lmodel in ['t', 'gaussian']:
            return self._z
        elif self._lmodel in ['mixture', 'vvh17']:
            Nvec0 = self.pta.get_ndiag(params)[0]
            Tmat = self.pta.get_basis(params)[0]

            Nvec = self._alpha * Nvec0
            theta_mean = np.dot(Tmat, self._b)
            top = self._theta * scipy.stats.norm.pdf(self._residuals,
                                                     loc=theta_mean,
                                                     scale=np.sqrt(Nvec))
            if self._lmodel == 'vvh17':
                top = self._theta / self.pspin

            bot = top + (1-self._theta) * scipy.stats.norm.pdf(self._residuals,
                                                               loc=theta_mean,
                                                               scale=np.sqrt(Nvec0))
            q = top / bot
            q[np.isnan(q)] = 1
            self._pout = q
    
            return scipy.stats.binom.rvs(1, list(map(lambda x: min(x, 1), q)))


    def update_alpha(self, xs): 

        # map parameters
        params = self.map_params(xs)

        # equation 12 of Tak, Ellis, Ghosh
        if np.sum(self._z) >= 1 and self.vary_alpha:
            Nvec0 = self.pta.get_ndiag(params)[0]
            Tmat = self.pta.get_basis(params)[0]
            theta_mean = np.dot(Tmat, self._b)
            top = ((self._residuals - theta_mean)**2 * 
                   self._z / Nvec0 + self.tdf) / 2
            bot = scipy.stats.gamma.rvs((self._z + 
                                         self.tdf) / 2)
            return top / bot
        else:
            return self._alpha

        
    def update_df(self, xs):

        if self.vary_df:
            # 1. evaluate the log conditional posterior of df for 1, 2, ..., 30.
            log_den_df = np.array(list(map(self.get_lnlikelihood_df, 
                                           np.arange(1,31))))

            # 2. normalize the probabilities
            den_df = np.exp(log_den_df - log_den_df.max())
            den_df /= den_df.sum()

            # 3. sample one of values (1, 2, ..., 30) according to the probabilities
            df = np.random.choice(np.arange(1, 31), p=den_df)

            return df
        else:
            return self.tdf


    def get_lnlikelihood_white(self, xs):

        # map parameters
        params = self.map_params(xs)
        matrix = self.pta.get_ndiag(params)[0]
        
        # Nvec and Tmat
        Nvec = self._alpha**self._z * matrix
        Tmat = self.pta.get_basis(params)[0]

        # whitened residuals
        mn = np.dot(Tmat, self._b)
        yred = self._residuals - mn

        # log determinant of N
        logdet_N = np.sum(np.log(Nvec))

        # triple product in likelihood function
        rNr = np.sum(yred**2/Nvec)

        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike


    def get_lnlikelihood(self, xs):

        # map parameter vector
        params = self.map_params(xs)

        # start likelihood calculations
        loglike = 0

        # get auxiliaries
        Nvec = self._alpha**self._z * self.pta.get_ndiag(params)[0]
        phiinv, logdet_phi = self.pta.get_phiinv(params, 
                                                 logdet=True)[0]
        residuals = self._residuals

        T = self.pta.get_basis(params)[0]
        if self.TNT is None and self.d is None:
            self.TNT = np.dot(T.T, T / Nvec[:,None])
            self.d = np.dot(T.T, residuals/Nvec)

        # log determinant of N
        logdet_N = np.sum(np.log(Nvec))

        # triple product in likelihood function
        rNr = np.sum(residuals**2/Nvec)

        # first component of likelihood function
        loglike += -0.5 * (logdet_N + rNr)

        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)

        try:
            cf = sl.cho_factor(Sigma)
            expval = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
        loglike += 0.5 * (np.dot(self.d, expval) - 
                          logdet_sigma - logdet_phi)

        return loglike

    
    def get_lnlikelihood_df(self, df):
        n = len(self._residuals)
        ll = -(df/2) * np.sum(np.log(self._alpha)+1/self._alpha) + \
            n * (df/2) * np.log(df/2) - n*scipy.special.gammaln(df/2)
        return ll

    
    def get_lnprior(self, xs):

        return sum(p.get_logpdf(x) for p, x 
                   in zip(self.params, xs))


    def sample(self, xs, outdir='./', niter=10000, resume=False):

        print(f'Creating chain directory: {outdir}')
        os.system(f'mkdir -p {outdir}')
        
        self.chain = np.zeros((niter, len(xs)))
        self.bchain = np.zeros((niter, len(self._b)))
        self.thetachain = np.zeros(niter)
        self.zchain = np.zeros((niter, len(self._residuals)))
        self.alphachain = np.zeros((niter, len(self._residuals)))
        self.poutchain = np.zeros((niter, len(self._residuals)))
        self.dfchain = np.zeros(niter)
        
        self.iter = 0
        startLength = 0
        xnew = xs
        if resume:
            print('Resuming from previous run...')
            # read in previous chains
            tmp_chains = []
            tmp_chains.append(np.loadtxt(f'{outdir}/chain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/bchain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/thetachain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/zchain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/alphachain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/poutchain.txt'))
            tmp_chains.append(np.loadtxt(f'{outdir}/dfchain.txt'))
            
            # find minimum length
            minLength = np.min([tmp.shape[0] for tmp in tmp_chains])
            
            # take only the minimum length entries of each chain
            tmp_chains = [tmp[:minLength] for tmp in tmp_chains]
            
            # pad with zeros if shorter than niter
            self.chain[:tmp_chains[0].shape[0]] = tmp_chains[0]
            self.bchain[:tmp_chains[1].shape[0]] = tmp_chains[1]
            self.thetachain[:tmp_chains[2].shape[0]] = tmp_chains[2]
            self.zchain[:tmp_chains[3].shape[0]] = tmp_chains[3]
            self.alphachain[:tmp_chains[4].shape[0]] = tmp_chains[4]
            self.poutchain[:tmp_chains[5].shape[0]] = tmp_chains[5]
            self.dfchain[:tmp_chains[6].shape[0]] = tmp_chains[6]
            
            # set new starting point for sampling
            startLength = minLength
            xnew = self.chain[startLength-1]
            
        tstart = time.time()
        for ii in range(startLength, niter):
            self.iter = ii
            self.chain[ii, :] = xnew
            self.bchain[ii,:] = self._b
            self.zchain[ii,:] = self._z
            self.thetachain[ii] = self._theta
            self.alphachain[ii,:] = self._alpha
            self.dfchain[ii] = self.tdf
            self.poutchain[ii, :] = self._pout

            self.TNT = None
            self.d = None

            # update white parameters
            xnew = self.update_white_params(xnew)

            # update hyper-parameters
            xnew = self.update_hyper_params(xnew)

            # if accepted update quadratic params
            if np.all(xnew != self.chain[ii,-1]):
                self._b = self.update_b(xnew)

            # update outlier model params
            self._theta = self.update_theta(xnew)
            self._z = self.update_z(xnew)
            self._alpha = self.update_alpha(xnew)
            self.tdf = self.update_df(xnew)

            if ii % 100 == 0 and ii > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent in %g seconds.'%(ii / niter * 100, 
                                                                       time.time()-tstart))
                sys.stdout.flush()
                np.savetxt(f'{outdir}/chain.txt', self.chain[:ii+1, :])
                np.savetxt(f'{outdir}/bchain.txt', self.bchain[:ii+1, :])
                np.savetxt(f'{outdir}/thetachain.txt', self.thetachain[:ii+1])
                np.savetxt(f'{outdir}/zchain.txt', self.zchain[:ii+1, :])
                np.savetxt(f'{outdir}/alphachain.txt', self.alphachain[:ii+1, :])
                np.savetxt(f'{outdir}/poutchain.txt', self.poutchain[:ii+1, :])
                np.savetxt(f'{outdir}/dfchain.txt', self.dfchain[:ii+1])
                
                
    def marg_outlierprob(self, burn=None):
        
        if burn is None:
            burn = int(0.25*self.poutchain.shape[0])
            
        return np.mean(self.poutchain[burn:,:self.iter], axis = 0)
