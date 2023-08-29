import numpy as np
from tqdm import tqdm
import scipy.linalg as sl
from functools import cached_property
import os, time, glob, warnings, arviz
from enterprise_extensions import model_utils, blocks
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import signal_base, gp_signals
from la_forge.core import Core

class BayesPower(object):

    """
    The Gibbs Method class used for single-pulsar noise analyses.

    Based on:

        Article by van Haasteren & Vallisneri (2014),
        "New advances in the Gaussian-process approach
        to pulsar-timing data analysis",
        Physical Review D, Volume 90, Issue 10, id.104012
        arXiv:1407.1838

        Initial structure of the code based on https://github.com/jellis18/gibbs_student_t

    Authors:

        S. R. Taylor
        N. Laal
    """

    def __init__(self,
                psr = None,
                Tspan = None,
                select = 'backend',
                white_vary = False,
                inc_ecorr = False,
                ecorr_type = 'kernel',
                noise_dict = None,
                tm_marg = False,
                freq_bins = 10,
                rhomin = -9., rhomax = -4.):
        
        """
        Parameters
        -----------

        psr : object
            instance of an ENTERPRISE psr object for a single pulsar

        Tspan: float (optional)
            if given, the baseline of the pulsar is fixed to the input value. If not,
            baseline is determined inetrnally

        select: str
            the selection of backend ('backend' or 'none') for the white-noise parameters

        white_vary: bool
            whether to vary the white noise

        inc_ecorr: bool
            whether to include ecorr

        ecorr_type: str
            the type of ecorr to use. Choose between 'basis' or 'kernel'

        noise_dict: dict
            white noise dictionary in case 'white_vary' is set to False

        tm_marg: bool
            whether to marginalize over timing model parameters (do not use this if you are varying the white noise!)

        freq_bins: int
            number of frequency bins for the red noise process

        rhomin: float
            lower bound for the log10 of the rho parameter.

        rhomax: float
            upper bound for the log10 of the rho parameter
        """

        self.psr = [psr]
        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)
        self.name = self.psr[0].name
        self.inc_ecorr = inc_ecorr
        self.ecorr_type = ecorr_type
        self.white_vary = white_vary
        self.tm_marg = tm_marg
        self.start_wn_iter = 0
        self.start_ec_iter = 0
        self.wn_names = ['efac', 'equad', 'ecorr']
        self.rhomin = rhomin
        self.rhomax = rhomax
        self.freq_bins = freq_bins
        
        ##Making the pta object
        if self.tm_marg:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            warnings.warn('***FYI: the timing model is marginalized for. This will slow down the WN sampling!!***')
        else:
            tm = gp_signals.TimingModel(use_svd=True)
        if self.ecorr_type == 'basis':
            wn = blocks.white_noise_block(vary=self.white_vary, inc_ecorr=self.inc_ecorr, gp_ecorr=True, select = select)
        else:
            wn = blocks.white_noise_block(vary=self.white_vary, inc_ecorr=self.inc_ecorr, gp_ecorr=False, select = select, kernel_ecorr_method = 'fast-sherman-morrison')
        rn = blocks.common_red_noise_block(psd='spectrum', prior='log-uniform', Tspan=self.Tspan,logmin=self.rhomin, logmax=self.rhomax,
                                        components=freq_bins, gamma_val=None, name = 'gw')
        s = tm + wn + rn
        self.pta = signal_base.PTA([s(p) for p in self.psr], lnlikelihood = signal_base.LogLikelihoodDenseCholesky)
        if not white_vary:
              self.pta.set_default_params(noise_dict)

        if self.inc_ecorr and 'basis' in self.ecorr_type:
            # grabbing priors on ECORR params
            for ct, par in enumerate(self.pta.params):
                if 'ecorr' in str(par): ind = ct
            ecorr_priors = str(self.pta.params[ind].params[0])
            ecorr_priors = ecorr_priors.split('(')[1].split(')')[0].split(', ')
            self.ecorrmin, self.ecorrmax = (10**(2*float(ecorr_priors[0].split('=')[1])),
                                            10**(2*float(ecorr_priors[1].split('=')[1])))

        # Getting residuals
        self._residuals = self.pta.get_residuals()[0]
        # Intial guess for the model params
        self._xs = np.array([p.sample() for p in self.pta.params], dtype = object)
        # Initializign the b-coefficients. The shape is 2*freq_bins if tm_marg = True.
        self._b = np.zeros(self.pta.get_basis(self._xs)[0].shape[1])
        self.Tmat = self.pta.get_basis(params = {})[0]
        self.Tmat_T = self.Tmat.T
        if not self.white_vary:
            self.Nmat = self.pta.get_ndiag(params = {})[0]
        else:
            self.Nmat = None
        self.phiinv = None

        # find basis indices of GW process
        self.gwid = []
        ct = 0
        psigs = [sig for sig in self.pta.signals.keys() if self.name in sig]
        for sig in psigs:
            Fmat = self.pta.signals[sig].get_basis()
            if 'gw' in self.pta.signals[sig].name:
                self.gwid.append(ct + np.arange(0,Fmat.shape[1]))
            # Avoid None-basis processes.
            # Also assume red + GW signals share basis.
            if Fmat is not None and 'red' not in sig:
                ct += Fmat.shape[1]

    @cached_property
    def params(self):
        return self.pta.params

    @cached_property
    def param_names(self):
        return self.pta.param_names

    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct : ct + n][0] if n > 1 else float(xs[ct])
            ct += n
        return ret

    @cached_property
    def get_red_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'log10_A' in par or 'gamma' in par or 'rho' in par:
                ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_efacequad_indices(self):
        ind = []
        if 'basis' in self.ecorr_type:
            for ct, par in enumerate(self.param_names):
                if 'efac' in par or 'equad' in par:
                    ind.append(ct)
        else:
            for ct, par in enumerate(self.param_names):
                if 'ecorr' in par or 'efac' in par or 'equad' in par:
                    ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_basis_ecorr_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'ecorr' in par:
                ind.append(ct)
        return np.array(ind)

    def update_red_params(self, xs):
        '''
        Function to perform log10_rho updates given the Fourier coefficients.
        '''

        low = 10**(2 * self.rhomin)
        high = 10**(2 * self.rhomax)

        tau = self._b[tuple(self.gwid)]**2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1/(np.exp(-tau/high) - np.exp(-tau/low))
        x = np.random.default_rng().uniform(0, 1, size = tau.shape)
        rhonew = -tau/np.log(x/Norm + np.exp(-tau/low))
        xs[-1] = 0.5*np.log10(rhonew)
        return xs

    def update_b(self, xs):
        '''
        Function to update the coefficients given the other model parameters.
        '''
        params = self.map_params(xs)
        phiinv = self.pta.get_phiinv(params, logdet=False)[0]

        T = self.Tmat
        try:
            TNT = self.TNT
        except:
            TNT = self.Nmat.solve(T, left_array = T)

        Sigma = TNT + np.diag(phiinv)
        var = np.linalg.inv(Sigma)
        mean = self.Nmat.solve(self._residuals, left_array= (var @ self.Tmat_T).T)
        try:
            b = np.random.default_rng().multivariate_normal(mean = mean, cov = var, check_valid = 'raise', method = 'cholesky')
            self._b = b
        except np.linalg.LinAlgError:
            if self.bchain.any():
                self._b = self.bchain[np.random.default_rng().integers(0, len(self.bchain))]
            else:
                bchain = np.memmap(self._savepath + '/chain_1', dtype='float32', mode='r', shape = (self.niter, self.len_x + self.len_b))[:, -len(self._b):]
                self._b = bchain[np.random.default_rng().integers(0, len(bchain))]
                
    def update_white_params(self, xs, iters=10):
        '''
        Function to perform WN updates given other model parameters.
        If kernel ecorr is chosen, WN includes ecorr as well.
        '''
        # get white noise parameter indices
        wind = self.get_efacequad_indices
        xnew = xs
        x0 = xnew[wind].copy()
        lnlike0, lnprior0  = self.get_lnlikelihood_white(x0), self.get_wn_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(self.start_wn_iter + 1, self.start_wn_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_wn.PTMCMCOneStep(x0, lnlike0, lnprob0, ii)
        xnew[wind] = x0
        self.start_wn_iter = ii

        ##Do some caching of "later needed" parameters for improved performance
        self.Nmat = self.pta.get_ndiag(self.map_params(xnew))[0]
        Tmat = self.Tmat
        if not 'basis' in self.ecorr_type:
            self.TNT = self.Nmat.solve(Tmat, left_array = Tmat)
        else:
            TN = Tmat/self.Nmat[:,None]
            self.TNT = Tmat.T @ TN
            residuals = self._residuals
            self.rNr = np.sum(residuals**2/self.Nmat)
            self.logdet_N = np.sum(np.log(self.Nmat))
            self.d = TN.T @ residuals

        return xnew

    def update_basis_ecorr_params(self, xs, iters=10):
        '''
        Function to perform basis ecorr updates.
        '''
        # get white noise parameter indices
        eind = self.get_basis_ecorr_indices
        xnew = xs
        x0 = xnew[eind].copy()
        lnlike0, lnprior0  = self.get_basis_ecorr_lnlikelihood(x0), self.get_basis_ecorr_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(self.start_ec_iter + 1, self.start_ec_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_ec.PTMCMCOneStep(x0, lnlike0, lnprob0, ii)
        xnew[eind] = x0
        self.start_ec_iter = ii

        return xnew

    def get_lnlikelihood_white(self, xs):
        '''
        Function to calculate WN log-liklihood.
        '''
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        params = self.map_params(x0)
        Nmat = self.pta.get_ndiag(params)[0]
        # whitened residuals
        yred = self._residuals - self.Tmat @ self._b
        try:
            if not 'basis' in self.ecorr_type:
                rNr, logdet_N = Nmat.solve(yred, left_array=yred, logdet = True)
            else:
                rNr = np.sum(yred**2/Nmat)
                logdet_N = np.sum(np.log(Nmat))
        except:
            return -np.inf
        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike


    def get_basis_ecorr_lnlikelihood(self, xs):
        '''
        Function to calculate basis ecorr log-liklihood.
        '''
        x0 = self._xs.copy()
        x0[self.get_basis_ecorr_indices] = xs

        params = self.map_params(x0)
        # start likelihood calculations
        loglike = 0
        # get auxiliaries
        phiinv, logdet_phi = self.pta.get_phiinv(params,logdet=True)[0]
        # first component of likelihood function
        loglike += -0.5 * (self.logdet_N + self.rNr)
        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)
        try:
            cf = sl.cho_factor(Sigma)
            expval = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
        loglike += 0.5 * (self.d @ expval -
                          logdet_sigma - logdet_phi)

        return loglike

    def get_wn_lnprior(self, xs):
        '''
        Function to calculate WN log-prior.
        '''
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        return np.sum([p.get_logpdf(value = x0[ct]) for ct, p in enumerate(self.params)])

    def get_basis_ecorr_lnprior(self, xs):
        '''
        Function to calculate basis ecorr log-prior.
        '''
        x0 = self._xs.copy()
        x0[self.get_basis_ecorr_indices] = xs

        return np.sum([p.get_logpdf(value = x0[ct]) for ct, p in enumerate(self.params)])

    def sample(self, niter=int(1e4), wniters = 30, eciters = 10, run_type = 'fast', save_type = 'npz',
               savepath = None, ESS_freq = int(5e3), ESS_limit = 1000,
               SCAMweight=30, AMweight=15, DEweight=50,
               covUpdate = 1000, burn = 10000, **kwargs):
        '''
        Gibbs Sampling

        Parameters
        -----------
        niter: integer
            total number of Gibbs sampling iterations

        wniters:
            number of white noise MCMC sampling iterations within each Gibbs step

        eciters:
            number of basis ecorr MCMC sampling iterations within each Gibbs step

        save_type: str
            the type of saving. Choose either 'txt', 'npz', 'core', or 'mem'. 
                'txt' gives you a single chain_1.txt file, 
                'npz' gives you a single compressed numpy format of the chain divided into 'white', 'rho', and 'b'. 
                You can read the 'npz' file using 'np.loadtxt(path, allow_pickle = True)[x]'
                where x could be 'b', 'rho', 'white', 'params' (parameter names), or 'priors'.
                'core' gives you a la_forge core object
                'mem' leaves the chain readable for a numpy.memmap function (can only be used when `run_type` is set to `efficient`).

        run_type: str
            the type of run. Choose between `fast` and `efficient`. 
                `fast` option does not write to disk until the sampler is done.
                This is ideal for simulations and stable computers without mnemory restrictions.
                `efficient` option uses numpy.memmap function to update a copy of the chain on to disk.
                `efficient` is slower than `fast`!

        ESS_freq: int
            the frequency at which to check for the effective-sample-size (ESS)

        ESS_limit: int
            the threshold for ESS in order to terminate the sampling.

        savepath: str
            the path to save the chains

        covUpdate: integer
            Number of iterations between AM covariance updates

        SCAMweight: integer
            Weight of SCAM jumps in overall jump cycle

        AMweight: integer
            Weight of AM jumps in overall jump cycle

        DEweight: integer
            Weight of DE jumps in overall jump cycle

        kwargs: dict
            PTMCMC initialization settings not mentioned above
        '''

        os.makedirs(savepath, exist_ok = True)

        if not save_type in ['txt', 'npz', 'core', 'mem']:
            raise TypeError("Only `npz`, `core`, `txt`, and `mem` options are valid") 
        
        if self.white_vary:
            isave = int(4e6)
            thin = 1
            Niter = int(niter * wniters + 1)

            x0 = self._xs[self.get_efacequad_indices]
            ndim = len(x0)
            cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution
            self.sampler_wn = ptmcmc(ndim, self.get_lnlikelihood_white, self.get_wn_lnprior, cov,
                                 outDir = savepath,
                                 resume=False)
            self.sampler_wn.initialize(Niter = Niter, isave = isave, thin = thin, SCAMweight=SCAMweight,
                                       AMweight=AMweight, DEweight=DEweight, covUpdate = covUpdate,
                                       burn = burn, **kwargs)

            if 'basis' in self.ecorr_type and self.white_vary:
                x0 = self._xs[self.get_basis_ecorr_indices]
                ndim = len(x0)
                cov = np.diag(np.ones(ndim) * 0.01**2)
                self.sampler_ec = ptmcmc(ndim, self.get_basis_ecorr_lnlikelihood, self.get_basis_ecorr_lnprior, cov,
                                     outDir = savepath,
                                     resume=False)
                self.sampler_ec.initialize(Niter = Niter, isave = isave, thin = thin, SCAMweight=SCAMweight,
                                           AMweight=AMweight, DEweight=DEweight, covUpdate = covUpdate,
                                           burn = burn, **kwargs)

        np.savetxt(savepath + '/pars.txt',list(map(str, self.pta.param_names)), fmt='%s')
        np.savetxt(savepath +'/priors.txt',list(map(lambda x: str(x.__repr__()), self.pta.params)), fmt='%s')
        freqs = np.arange(1/self.Tspan, (self.freq_bins + .001)/self.Tspan, 1/self.Tspan)
        np.save(savepath +'/freqs.npy',freqs)
        [os.remove(dpa) for dpa in glob.glob(savepath + '/*jump.txt')]

        xnew = self._xs.copy()
        pbar = tqdm(range(niter), colour="GREEN")
        pbar.set_description("Sampling %s" % self.name)
        startt = time.time()
        endt = None
        
        if run_type == 'fast':

            self.rhochain = np.zeros((niter, len(self._xs[-1])))
            if self.white_vary:
                self.whitechain = np.zeros((niter, len(self._xs[:-1])))
            self.bchain = np.zeros((niter, len(self._b)))

            for ii in pbar:
                if self.white_vary:
                    xnew = self.update_white_params(xnew, iters=wniters)

                if self.inc_ecorr and 'basis' in self.ecorr_type:
                    xnew = self.update_basis_ecorr_params(xnew, iters=eciters)

                self.rhochain[ii, :] = xnew[-1]
                if self.white_vary:
                    self.whitechain[ii,:] = xnew[:-1]

                self.update_b(xs = xnew)
                self.bchain[ii, :] = self._b

                xnew = self.update_red_params(xs = xnew)

                if ii%ESS_freq == 0 and ii > 1000:
                    if self.white_vary:
                        _ = np.concatenate((self.rhochain[:ii+1, :], self.bchain[:ii+1, :], self.whitechain[:ii+1, :]), axis = 1)
                        ess = arviz.ess(arviz.convert_to_dataset(_[None, :, :])).to_array().values
                    else:
                        _ = np.concatenate((self.rhochain[:ii+1, :], self.bchain[:ii+1, :]), axis = 1)
                        ess = arviz.ess(arviz.convert_to_dataset(_[None, :, :])).to_array().values

                    if ess.min() >= ESS_limit:
                        self.rhochain = self.rhochain[:ii, :]
                        if self.white_vary:
                            self.whitechain = self.whitechain[:ii, :]
                        self.bchain = self.bchain[:ii, :]
                        endt = time.time()
                        break
                    else:
                        print(ess.min())

                if not endt:
                    endt = time.time()

            
            if self.white_vary:
                if save_type =='npz':
                    np.savez_compressed(savepath + '/{}.npz'.format(self.name), b = self.bchain, rho = self.rhochain,
                                        white = self.whitechain, time =  [endt - startt],
                                        params = self.pta.param_names,
                                        priors = list(map(lambda x: str(x.__repr__()), self.pta.params)))
                    os.remove(savepath + '/chain_1.txt')

                elif save_type == 'core':
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.whitechain, self.rhochain, self.bchain), axis = 1))
                    c = Core(chaindir = savepath, label = '')
                    c.set_rn_freqs(freqs)
                    c.save(savepath + '/core.core')
                    os.remove(savepath + '/chain_1.txt')
                elif save_type == 'txt':
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.whitechain, self.rhochain, self.bchain), axis = 1))

                elif save_type == 'mem':
                    print('this option is not available when `run_type == fast`. Saving as a txt file instead!')
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.rhochain, self.bchain), axis = 1))    
            else:
                if save_type =='npz':
                    np.savez_compressed(savepath + '/{}.npz'.format(self.name), b = self.bchain, rho = self.rhochain,
                                        time =  [endt - startt],
                                        params = self.pta.param_names,
                                        priors = list(map(lambda x: str(x.__repr__()), self.pta.params)))
                    os.remove(savepath + '/chain_1.txt')
                
                elif save_type == 'core':
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.rhochain, self.bchain), axis = 1))
                    c = Core(chaindir = savepath, label = '')
                    c.set_rn_freqs(freqs)
                    c.save(savepath + '/core.core')
                    os.remove(savepath + '/chain_1.txt')
                elif save_type == 'txt':
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.rhochain, self.bchain), axis = 1))                
                elif save_type == 'mem':
                    print('this option is not available when `run_type == fast`. Saving as a txt file instead!')
                    np.savetxt(savepath + '/chain_1.txt', np.concatenate((self.rhochain, self.bchain), axis = 1)) 

        elif run_type == 'efficient':

            len_b = len(self._b)
            len_x = len(np.hstack(self._xs))
            self._savepath = savepath
            
            fp = np.memmap(savepath + '/chain_1', dtype='float32', mode='w+', shape = (niter, len_x + len_b))

            for ii in pbar:
                if self.white_vary:
                    xnew = self.update_white_params(xnew, iters=wniters)

                if self.inc_ecorr and 'basis' in self.ecorr_type:
                    xnew = self.update_basis_ecorr_params(xnew, iters=eciters)

                self.update_b(xs = xnew)
                xnew = self.update_red_params(xs = xnew)

                fp[ii, -len_b:] = self._b
                fp[ii, 0:len_x] = np.hstack(xnew)

                if ii%ESS_freq == 0 and ii > 1000:

                    lp = np.memmap(savepath + '/chain_1', dtype='float32', mode='r', shape = (niter, len_x + len_b))
                    ess = arviz.ess(arviz.convert_to_dataset(lp[None, :, :])).to_array().values
                    if ess.min() >= ESS_limit:
                        rhochain = rhochain[:ii, :]
                        if self.white_vary:
                            whitechain = whitechain[:ii, :]
                        bchain = bchain[:ii, :]
                        endt = time.time()
                        break
                    else:
                        print(ess.min())

                if not endt:
                    endt = time.time()

            if self.white_vary:
                if save_type =='npz':
                    np.savez_compressed(savepath + '/{}.npz'.format(self.name), b = fp[:, -len_b:], rho = fp[ii, -(len_b + self.freq_bins):-len_b],
                                        white = fp[ii, 0:-(len_b + self.freq_bins)], time =  [endt - startt],
                                        params = self.pta.param_names,
                                        priors = list(map(lambda x: str(x.__repr__()), self.pta.params)))
                    os.remove(savepath + '/chain_1.txt')
                    os.remove(savepath + '/chain_1')

                elif save_type == 'core':
                    np.savetxt(savepath + '/chain_1.txt', fp)
                    c = Core(chaindir = savepath, label = '')
                    c.set_rn_freqs(freqs)
                    c.save(savepath + '/core.core')
                    os.remove(savepath + '/chain_1.txt')
                    os.remove(savepath + '/chain_1')
                elif save_type == 'txt':
                    np.savetxt(savepath + '/chain_1.txt', fp)
                    os.remove(savepath + '/chain_1')
                elif save_type == 'mem':
                    print('The chain is kept as a numpy.memmap readable file.')  
                    os.remove(savepath + '/chain_1.txt')
            else:
                if save_type =='npz':
                    np.savez_compressed(savepath + '/{}.npz'.format(self.name), b = fp[:, -len_b:], rho = fp[ii, -(len_b + self.freq_bins):-len_b],
                                            time =  [endt - startt],
                                            params = self.pta.param_names,
                                            priors = list(map(lambda x: str(x.__repr__()), self.pta.params)))

                elif save_type == 'core':
                    np.savetxt(savepath + '/chain_1.txt', fp)
                    c = Core(chaindir = savepath, label = '')
                    c.set_rn_freqs(freqs)
                    c.save(savepath + '/core.core')
                    os.remove(savepath + '/chain_1.txt')

                elif save_type == 'txt':
                    np.savetxt(savepath + '/chain_1.txt', fp)

                elif save_type == 'mem':
                    print('The chain is kept as a numpy.memmap readable file.')
                   
        else:
            raise TypeError("Only `fast` and `efficient` options are valid")