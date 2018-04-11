from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt
from glob import glob
import json
import os
import hashlib
import acor

try:
    import cPickle as pickle
except:
    import pickle

from enterprise.pulsar import Pulsar
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

def get_tspan(psrs):
    """ Returns maximum time span for all pulsars.

    :param psrs: List of pulsar objects

    """

    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])

    return tmax - tmin

class JumpProposal(object):

    def __init__(self, pta, snames=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        # collecting signal parameters across pta
        if snames is None:
            self.snames = dict.fromkeys(np.unique([[qq.signal_name for qq in pp._signals]
                                                   for pp in pta._signalcollections]))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = np.unique(self.snames[key]).tolist()
        else:
            self.snames = snames

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        idx = np.random.randint(0, self.npar)

        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[idx] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'red noise'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    def draw_from_dm_gp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'dm_gp'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    def draw_from_dm1yr_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dm1yr_names = [dmname for dmname in self.pnames if 'dm_s1yr' in dmname]
        dmname = np.random.choice(dm1yr_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'phase' in dmname:
            q[idx] = np.random.uniform(0, 2*np.pi)

        return q, 0

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_gw')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dipole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_dipole')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_monopole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_monopole')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_altpol_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        polnames = [pol for pol in self.pnames if 'log10Apol' in pol]
        if 'kappa' in self.pnames:
            polnames.append('kappa')
        pol = np.random.choice(polnames)
        idx = self.pnames.index(pol)
        if pol == 'log10Apol_tt':
            q[idx] = np.random.uniform(-18, -12)
        elif pol == 'log10Apol_st':
            q[idx] = np.random.uniform(-18, -12)
        elif pol == 'log10Apol_vl':
            q[idx] = np.random.uniform(-18, -15)
        elif pol == 'log10Apol_sl':
            q[idx] = np.random.uniform(-18, -16)
        elif pol == 'kappa':
            q[idx] = np.random.uniform(0, 10)

        return q, 0

    def draw_from_ephem_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'phys_ephem'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    def draw_from_bwm_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'bwm'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_h')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    ipars = np.array([p for p in pars if p not in gpars])

    return gpars, ipars


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if any(gpars):
        groups.extend([[params.index(gp) for gp in gpars]])

    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if p not in gpars]
            if ind:
                groups.extend([ind])

    return groups


def setup_sampler(pta, outdir='chains', resume=False):
    """
    Sets up an instance of PTMCMC sampler.

    We initialize the sampler the likelihood and prior function
    from the PTA object. We set up an initial jump covariance matrix
    with fairly small jumps as this will be adapted as the MCMC runs.

    We will setup an output directory in `outdir` that will contain
    the chain (first n columns are the samples for the n parameters
    and last 4 are log-posterior, log-likelihood, acceptance rate, and
    an indicator variable for parallel tempering but it doesn't matter
    because we aren't using parallel tempering).

    We then add several custom jump proposals to the mix based on
    whether or not certain parameters are in the model. These are
    all either draws from the prior distribution of parameters or
    draws from uniform distributions.
    """

    # dimension of parameter space
    ndim = len(pta.param_names)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.1**2)

    # parameter groupings
    groups = get_parameter_groups(pta)

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                     outDir=outdir, resume=resume)
    np.savetxt(outdir+'/pars.txt',
               list(map(str, pta.param_names)), fmt='%s')
    np.savetxt(outdir+'/priors.txt',
               list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

    # additional jump proposals
    jp = JumpProposal(pta)

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 5)

    # Red noise prior draw
    if 'red noise' in jp.snames:
        print('Adding red noise prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

    # DM GP noise prior draw
    if 'dm_gp' in jp.snames:
        print('Adding DM GP noise prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

    # DM annual prior draw
    if 'dm_s1yr' in jp.snames:
        print('Adding DM annual prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

    # Ephemeris prior draw
    if 'd_jupiter_mass' in pta.param_names:
        print('Adding ephemeris model prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if 'log10_A_gw' in pta.param_names:
        print('Adding GWB uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if 'log10_A_dipole' in pta.param_names:
        print('Adding dipole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if 'log10_A_monopole' in pta.param_names:
        print('Adding monopole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if 'log10Apol_tt' in pta.param_names:
        print('Adding alternative GW-polarization uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_altpol_log_uniform_distribution, 10)

    return sampler

class PostProcessing(object):

    def __init__(self, chain, pars, burn_percentage=0.25):
        burn = int(burn_percentage*chain.shape[0])
        self.chain = chain[burn:]
        self.pars = pars

    def plot_trace(self, plot_kwargs={}):
        ndim = len(self.pars)
        if ndim > 1:
            ncols = 4
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1

        fig = plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.plot(self.chain[:, ii], **plot_kwargs)
            plt.title(self.pars[ii], fontsize=8)
        plt.tight_layout()

    def plot_hist(self, hist_kwargs={'bins':50, 'normed':True}):
        ndim = len(self.pars)
        if ndim > 1:
            ncols = 4
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1

        fig = plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.hist(self.chain[:, ii], **hist_kwargs)
            plt.title(self.pars[ii], fontsize=8)
        plt.tight_layout()


def ul(chain, q=95.0):
    """
    Computes upper limit and associated uncertainty.

    :param chain: MCMC samples of GWB (or common red noise) amplitude
    :param q: desired percentile of upper-limit value [out of 100, default=95]

    :returns: (upper limit, uncertainty on upper limit)
    """

    hist = np.histogram(10.0**chain, bins=100)
    hist_dist = scistats.rv_histogram(hist)

    A_ul = 10**np.percentile(chain, q=q)
    p_ul = hist_dist.pdf(A_ul)

    Aul_error = np.sqrt( (q/100.) * (1.0 - (q/100.0)) /
                        (chain.shape[0]/acor.acor(chain)[0]) ) / p_ul

    return A_ul, Aul_error


def bayes_fac(samples, ntol = 200, logAmin = -18, logAmax = -14):
    """
    Computes the Savage Dickey Bayes Factor and uncertainty.

    :param samples: MCMC samples of GWB (or common red noise) amplitude
    :param ntol: Tolerance on number of samples in bin

    :returns: (bayes factor, 1-sigma bayes factor uncertainty)
    """

    prior = 1 / (logAmax - logAmin)
    dA = np.linspace(0.01, 0.1, 100)
    bf = []
    bf_err = []
    mask = [] # selecting bins with more than 200 samples

    for ii,delta in enumerate(dA):
        n = np.sum(samples <= (logAmin + delta))
        N = len(samples)

        post = n / N / delta

        bf.append(prior/post)
        bf_err.append(bf[ii]/np.sqrt(n))

        if n > ntol:
            mask.append(ii)

    return np.mean(np.array(bf)[mask]), np.std(np.array(bf)[mask])


def odds_ratio(chain, models=[0,1], uncertainty=True, thin=False):

    if thin:
        indep_samples = np.rint( chain.shape[0] / acor.acor(chain)[0] )
        samples = np.random.choice(chain.copy(), int(indep_samples))
    else:
        samples = chain.copy()

    mask_top = np.rint(samples) == max(models)
    mask_bot = np.rint(samples) == min(models)

    top = float(np.sum(mask_top))
    bot = float(np.sum(mask_bot))

    if top == 0.0 and bot != 0.0:
        bf = 1.0 / bot
    elif bot == 0.0 and top != 0.0:
        bf = top
    else:
        bf = top / bot

    if uncertainty:

        if bot == 0. or top == 0.:
            sigma = 0.0
        else:
            # Counting transitions from model 1 model 2
            ct_tb = 0
            for ii in range(len(mask_top)-1):
                if mask_top[ii]:
                    if not mask_top[ii+1]:
                        ct_tb += 1

            # Counting transitions from model 2 to model 1
            ct_bt = 0
            for ii in range(len(mask_bot)-1):
                if mask_bot[ii]:
                    if not mask_bot[ii+1]:
                        ct_bt += 1

            try:
                sigma = bf * np.sqrt( (float(top) - float(ct_tb))/(float(top)*float(ct_tb)) +
                                     (float(bot) - float(ct_bt))/(float(bot)*float(ct_bt)) )
            except ZeroDivisionError:
                sigma = 0.0

        return bf, sigma

    elif not uncertainty:

        return bf


class HyperModel(object):
    """
    Class to define hyper-model that is the concatenation of all models.
    """

    def __init__(self, models):
        self.models = models
        self.num_models = len(self.models)

        #########
        self.param_names, ind = np.unique(np.concatenate([p.param_names
                                                     for p in self.models.values()]),
                                     return_index=True)
        self.param_names = self.param_names[np.argsort(ind)]
        self.param_names = np.append(self.param_names, 'nmodel').tolist()
        #########

        #########
        self.params = [p for p in self.models[0].params] # start of param list
        uniq_params = [str(p) for p in self.models[0].params] # which params are unique
        for model in self.models.values():
            # find differences between next model and concatenation of previous
            param_diffs = np.setdiff1d([str(p) for p in model.params], uniq_params)
            mask = np.array([str(p) in param_diffs for p in model.params])
            # concatenate for next loop iteration
            uniq_params = np.union1d([str(p) for p in model.params], uniq_params)
            # extend list of unique parameters
            self.params.extend([pp for pp in np.array(model.params)[mask]])
        #########

        #########
        # get signal collections
        self.snames = dict.fromkeys(np.unique(sum(sum([[[qq.signal_name for qq in pp._signals]
                                                        for pp in self.models[mm]._signalcollections]
                                                       for mm in self.models], []), [])))
        for key in self.snames: self.snames[key] = []

        for mm in self.models:
            for sc in self.models[mm]._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
        for key in self.snames: self.snames[key] = np.unique(self.snames[key]).tolist()

        for key in self.snames:
            uniq_params, ind = np.unique([p.name for p in self.snames[key]],
                                         return_index=True)
            uniq_params = uniq_params[np.argsort(ind)].tolist()
            all_params = [p.name for p in self.snames[key]]

            self.snames[key] = np.array(self.snames[key])[[all_params.index(q)
                                                           for q in uniq_params]].tolist()
        #########

    def get_lnlikelihood(self, x):

        # find model index variable
        idx = list(self.param_names).index('nmodel')
        nmodel = np.rint(x[idx])

        # find parameters of active model
        q = []
        for par in self.models[nmodel].param_names:
            idx = self.param_names.index(par)
            q.append(x[idx])

        # only active parameters enter likelihood
        return self.models[nmodel].get_lnlikelihood(q)

    def get_lnprior(self, x):

        # find model index variable
        idx = list(self.param_names).index('nmodel')
        nmodel = np.rint(x[idx])

        if nmodel not in self.models.keys():
            return -np.inf
        else:
            lnP = 0
            for p in self.models.values():
                q = []
                for par in p.param_names:
                    idx = self.param_names.index(par)
                    q.append(x[idx])
                lnP += p.get_lnprior(np.array(q))

            return lnP

    def get_parameter_groups(self):

        groups = []
        for p in self.models.values():
            groups.extend(get_parameter_groups(p))
        list(np.unique(groups))

        groups.extend([[len(self.param_names)-1]]) # nmodel

        return groups

    def initial_sample(self):
        """
        Draw an initial sample from within the hyper-model prior space.
        """

        x0 = [np.array(p.sample()).ravel().tolist() for p in self.models[0].params]
        uniq_params = [str(p) for p in self.models[0].params]

        for model in self.models.values():
            param_diffs = np.setdiff1d([str(p) for p  in model.params], uniq_params)
            mask = np.array([str(p) in param_diffs for p in model.params])
            x0.extend([np.array(pp.sample()).ravel().tolist() for pp in np.array(model.params)[mask]])

            uniq_params = np.union1d([str(p) for p in model.params], uniq_params)

        x0.extend([[0.1]])

        return np.array([p for sublist in x0 for p in sublist])

    def draw_from_nmodel_prior(self, x, iter, beta):
        """
        Model-index uniform distribution prior draw.
        """

        q = x.copy()

        idx = list(self.param_names).index('nmodel')
        q[idx] = np.random.uniform(-0.5,self.num_models-0.5)

        lqxy = 0

        return q, float(lqxy)

    def setup_sampler(self, outdir='chains', resume=False):
        """
        Sets up an instance of PTMCMC sampler.

        We initialize the sampler the likelihood and prior function
        from the PTA object. We set up an initial jump covariance matrix
        with fairly small jumps as this will be adapted as the MCMC runs.

        We will setup an output directory in `outdir` that will contain
        the chain (first n columns are the samples for the n parameters
        and last 4 are log-posterior, log-likelihood, acceptance rate, and
        an indicator variable for parallel tempering but it doesn't matter
        because we aren't using parallel tempering).

        We then add several custom jump proposals to the mix based on
        whether or not certain parameters are in the model. These are
        all either draws from the prior distribution of parameters or
        draws from uniform distributions.
        """

        # dimension of parameter space
        ndim = len(self.param_names)

        # initial jump covariance matrix
        cov = np.diag(np.ones(ndim) * 0.1**2)

        # parameter groupings
        groups = self.get_parameter_groups()

        sampler = ptmcmc(ndim, self.get_lnlikelihood, self.get_lnprior, cov,
                         groups=groups, outDir=outdir, resume=resume)
        np.savetxt(outdir+'/pars.txt', self.param_names, fmt='%s')
        np.savetxt(outdir+'/priors.txt', self.params, fmt='%s')

        # additional jump proposals
        jp = JumpProposal(self, self.snames)

        # always add draw from prior
        sampler.addProposalToCycle(jp.draw_from_prior, 5)

        # Red noise prior draw
        if 'red noise' in self.snames:
            print('Adding red noise prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

        # DM GP noise prior draw
        if 'dm_gp' in self.snames:
            print('Adding DM GP noise prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

        # DM annual prior draw
        if 'dm_s1yr' in jp.snames:
            print('Adding DM annual prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

        # Ephemeris prior draw
        if 'd_jupiter_mass' in self.param_names:
            print('Adding ephemeris model prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

        # GWB uniform distribution draw
        if 'log10_A_gw' in self.param_names:
            print('Adding GWB uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

        # Dipole uniform distribution draw
        if 'log10_A_dipole' in self.param_names:
            print('Adding dipole uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

        # Monopole uniform distribution draw
        if 'log10_A_monopole' in self.param_names:
            print('Adding monopole uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

        # BWM prior draw
        if 'log10_A_bwm' in self.param_names:
            print('Adding BWM prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

        # CW prior draw
        if 'log10_h' in self.param_names:
            print('Adding CW prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)

        # Model index distribution draw
        if 'nmodel' in self.param_names:
            print('Adding nmodel uniform distribution draws...\n')
            sampler.addProposalToCycle(self.draw_from_nmodel_prior, 20)

        return sampler
