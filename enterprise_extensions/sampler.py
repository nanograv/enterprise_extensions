# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
import os
from enterprise import constants as const
import pickle

from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

class JumpProposal(object):

    def __init__(self, pta, snames=None, empirical_distr=None):
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
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames

        # empirical distributions
        if empirical_distr is not None and os.path.isfile(empirical_distr):
            try:
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f, encoding='latin1')
            except:
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                except:
                    print('I can\'t open the empirical distribution pickle file!')
                    pickled_distr = None

            if pickled_distr is None:
                self.empirical_distr = None

        elif isinstance(empirical_distr,list):
            pass
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx,d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) > 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
            else:
                self.empirical_distr = None

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
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

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
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_empirical_distr(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        if self.empirical_distr is not None:

            # randomly choose one of the empirical distributions
            distr_idx = np.random.randint(0, len(self.empirical_distr))

            if self.empirical_distr[distr_idx].ndim == 1:

                idx = self.pnames.index(self.empirical_distr[distr_idx].param_name)
                q[idx] = self.empirical_distr[distr_idx].draw()

                lqxy = (self.empirical_distr[distr_idx].logprob(x[idx]) -
                        self.empirical_distr[distr_idx].logprob(q[idx]))

            else:

                oldsample = [x[self.pnames.index(p)]
                             for p in self.empirical_distr[distr_idx].param_names]
                newsample = self.empirical_distr[distr_idx].draw()

                for p,n in zip(self.empirical_distr[distr_idx].param_names, newsample):
                    q[self.pnames.index(p)] = n

                lqxy = (self.empirical_distr[distr_idx].logprob(oldsample) -
                        self.empirical_distr[distr_idx].logprob(newsample))

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
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

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

    def draw_from_dmexpdip_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dmexp_names = [dmname for dmname in self.pnames if 'dmexp' in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'log10_tau' in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        elif 'sign_param' in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmexpcusp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dmexp_names = [dmname for dmname in self.pnames if 'dm_cusp' in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'log10_tau' in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        #elif 't0' in dmname:
        #    q[idx] = np.random.uniform(53393.0, 57388.0)
        elif 'sign_param' in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmx_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'dmx_signal'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('gw_log10_A')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dipole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('dipole_log10_A')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_monopole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('monopole_log10_A')
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
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

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
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_cw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'cw'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_h')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dm_sw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'gp_sw'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_signal_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0
        std = ['linear timing model',
               'red noise',
               'phys_ephem',
               'gw',
               'cw',
               'bwm',
               'gp_sw',
               'ecorr_sherman-morrison',
               'ecorr',
               'efac',
               'equad',
               ]
        non_std = [nm for nm in self.snames.keys() if nm not in std]
        # draw parameter from signal model
        signal_name = np.random.choice(non_std)
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = list(set(par for par in pars if pars.count(par) > 1))
    ipars = [par for par in pars if par not in gpars]

    # gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    # ipars = np.array([p for p in pars if p not in gpars])

    return np.array(gpars), np.array(ipars)


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    params = pta.param_names
    ndim = len(params)
    groups = [list(np.arange(0, ndim))]

    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if gpars.size:
        # add a group of all global parameters
        groups.append([params.index(gp) for gp in gpars])

    # make a group for each signal, with all non-global parameters
    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if not gpars.size or p not in gpars]
            if ind:
                groups.append(ind)

    return groups


def get_cw_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    ang_pars = ['costheta', 'phi', 'cosinc', 'phase0', 'psi']
    mfdh_pars = ['log10_Mc', 'log10_fgw', 'log10_dL', 'log10_h']
    freq_pars = ['log10_Mc', 'log10_fgw', 'pdist', 'pphase']

    groups = []
    for pars in [ang_pars, mfdh_pars, freq_pars]:
        groups.append(group_from_params(pta, pars))

    return groups


def group_from_params(pta, params):
    gr = []
    for p in params:
        for q in pta.param_names:
            if p in q:
                gr.append(pta.param_names.index(q))
    return gr


def setup_sampler(pta, outdir='chains', resume=False, empirical_distr=None):
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
    params = pta.param_names
    ndim = len(params)

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
    jp = JumpProposal(pta, empirical_distr=empirical_distr)

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 5)

    # try adding empirical proposals
    if empirical_distr is not None:
        print('Adding empirical proposals...\n')
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 10)

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

    # DM dip prior draw
    if 'dmexp' in jp.snames:
        print('Adding DM exponential dip prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)

    # DM cusp prior draw
    if 'dm_cusp' in jp.snames:
        print('Adding DM exponential cusp prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)

    # DMX prior draw
    if 'dmx_signal' in jp.snames:
        print('Adding DMX prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

    # Ephemeris prior draw
    if 'd_jupiter_mass' in pta.param_names:
        print('Adding ephemeris model prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if 'gw_log10_A' in pta.param_names:
        print('Adding GWB uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if 'dipole_log10_A' in pta.param_names:
        print('Adding dipole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if 'monopole_log10_A' in pta.param_names:
        print('Adding monopole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if 'log10Apol_tt' in pta.param_names:
        print('Adding alternative GW-polarization uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_altpol_log_uniform_distribution, 10)

    # BWM prior draw
    if 'bwm_log10_A' in pta.param_names:
        print('Adding BWM prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

    # CW prior draw
    if 'cw_log10_h' in pta.param_names:
        print('Adding CW strain prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)
    if 'cw_log10_Mc' in pta.param_names:
        print('Adding CW prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_cw_distribution, 10)

    return sampler
