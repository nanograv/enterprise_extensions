# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
import scipy.stats as scistats
import acor
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

<<<<<<< HEAD
from enterprise.pulsar import Pulsar
from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
=======
# Log-spaced frequncies
def linBinning(T, logmode, f_min, nlin, nlog):
    """
    Get the frequency binning for the low-rank approximations, including
    log-spaced low-frequency coverage.
    Credit: van Haasteren & Vallisneri, MNRAS, Vol. 446, Iss. 2 (2015)
    :param T:       Duration experiment
    :param logmode: From which linear mode to switch to log
    :param f_min:   Down to which frequency we'll sample
    :param nlin:    How many linear frequencies we'll use
    :param nlog:    How many log frequencies we'll use
    """
    if logmode < 0:
        raise ValueError("Cannot do log-spacing when all frequencies are"
                         "linearly sampled")

    # First the linear spacing and weights
    df_lin = 1.0 / T
    f_min_lin = (1.0 + logmode) / T
    f_lin = np.linspace(f_min_lin, f_min_lin + (nlin-1)*df_lin, nlin)
    w_lin = np.sqrt(df_lin * np.ones(nlin))

    if nlog > 0:
        # Now the log-spacing, and weights
        f_min_log = np.log(f_min)
        f_max_log = np.log( (logmode+0.5)/T )
        df_log = (f_max_log - f_min_log) / (nlog)
        f_log = np.exp(np.linspace(f_min_log+0.5*df_log,
                                   f_max_log-0.5*df_log, nlog))
        w_log = np.sqrt(df_log * f_log)
        return np.append(f_log, f_lin), np.append(w_log, w_lin)
    else:
        return f_lin, w_lin
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

# New filter for different cadences
def cadence_filter(psr, start_time=None, end_time=None, cadence=None):
    """ Filter data for coarser cadences. """

    if start_time is None and end_time is None and cadence is None:
        mask = np.ones(psr._toas.shape, dtype=bool)
    else:
        # find start and end indices of cadence filtering
        start_idx = (np.abs((psr._toas / 86400) - start_time)).argmin()
        end_idx = (np.abs((psr._toas / 86400) - end_time)).argmin()
        # make a safe copy of sliced toas
        tmp_toas = psr._toas[start_idx:end_idx+1].copy()
        # cumulative sum of time differences
        cumsum = np.cumsum(np.diff(tmp_toas / 86400))
        tspan = (tmp_toas.max() - tmp_toas.min()) / 86400
        # find closest indices of sliced toas to desired cadence
        mask = []
        for ii in np.arange(1.0, tspan, cadence):
            idx = (np.abs(cumsum - ii)).argmin()
            mask.append(idx)
        # append start and end segements with cadence-sliced toas
        mask = np.append(np.arange(start_idx),
                         np.array(mask) + start_idx)
        mask = np.append(mask, np.arange(end_idx, len(psr._toas)))

    psr._toas = psr._toas[mask]
    psr._toaerrs = psr._toaerrs[mask]
    psr._residuals = psr._residuals[mask]
    psr._ssbfreqs = psr._ssbfreqs[mask]

    psr._designmatrix = psr._designmatrix[mask, :]
    dmx_mask = np.sum(psr._designmatrix, axis=0) != 0.0
    psr._designmatrix = psr._designmatrix[:, dmx_mask]

    for key in psr._flags:
        psr._flags[key] = psr._flags[key][mask]

    if psr._planetssb is not None:
        psr._planetssb = psr.planetssb[mask, :, :]

    psr.sort_data()

def get_tspan(psrs):
    """ Returns maximum time span for all pulsars.

    :param psrs: List of pulsar objects

    """

    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])

    return tmax - tmin

<<<<<<< HEAD
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
            else:
                # only save the empirical distributions for parameters that are in the model
                mask = []
                for idx,d in enumerate(pickled_distr):
                    if d.ndim == 1:
                        if d.param_name in pta.param_names:
                            mask.append(idx)
                    else:
                        if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                            mask.append(idx)
                if len(mask) > 1:
                    self.empirical_distr = [pickled_distr[m] for m in mask]
                else:
                    self.empirical_distr = None
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
            q[idx] = np.random.uniform(np.log10(5), np.log10(100))
        elif 't0' in dmname:
            q[idx] = np.random.uniform(53393.0, 57388.0)
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
            q[idx] = np.random.uniform(np.log10(5), np.log10(100))
        elif 't0' in dmname:
            q[idx] = np.random.uniform(53393.0, 57388.0)
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

        signal_name = 'dm_sw'

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

    def draw_from_par_prior(self, par_names):
        # Preparing and comparing par_names with PTA parameters
        par_names = np.atleast_1d(par_names)
        par_list = []
        name_list = []
        for par_name in par_names:
            pn_list = [n for n in self.plist if par_name in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter prior match found between {} and PTA.object."
                              .format(par_names))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """Prior draw function generator for custom par_names.
            par_names: list of strings

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # randomly choose parameter
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)

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

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_prior'.format(name_string)
        return draw

    def draw_from_par_log_uniform(self, par_dict):
        # Preparing and comparing par_dict.keys() with PTA parameters
        par_list = []
        name_list = []
        for par_name in par_dict.keys():
            pn_list = [n for n in self.plist if par_name in n and 'log' in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter dictionary match found between {} and PTA.object."
                              .format(par_dict.keys()))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """log uniform prior draw function generator for custom par_names.
            par_dict: dictionary with {"par_names":(lower bound,upper bound)}
                                      { "string":(float,float)}

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # draw parameter from signal model
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)
            q[idx] = np.random.uniform(par_dict[par_name][0],par_dict[par_name][1])

            return q, 0

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_log_uniform'.format(name_string)
        return draw

    def draw_from_signal(self, signal_names):
        # Preparing and comparing signal_names with PTA signals
        signal_names = np.atleast_1d(signal_names)
        signal_list = []
        name_list = []
        for signal_name in signal_names:
            try:
                param_list = self.snames[signal_name]
                signal_list.append(param_list)
                name_list.append(signal_name)
            except:
                pass
        if not signal_list:
            raise UserWarning("No signal match found between {} and PTA.object!"
                              .format(signal_names))
        signal_list = np.concatenate(signal_list,axis=None)

        def draw(x, iter, beta):
            """Signal draw function generator for custom signal_names.
            signal_names: list of strings

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # draw parameter from signal model
            param = np.random.choice(signal_list)
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

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_signal'.format(name_string)
        return draw


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
    groups  = [list(np.arange(0, ndim))]

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
=======
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3

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

        plt.figure(figsize=(15, 2*nrows))
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

        plt.figure(figsize=(15, 2*nrows))
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

def bic(chain, nobs, log_evidence=False):
    """
    Computes the Bayesian Information Criterion.

    :param chain: MCMC samples of all parameters, plus meta-data
    :param nobs: Number of observations in data
    :param evidence: return evidence estimate too?

    :returns: (bic, evidence)
    """
    nparams = chain.shape[1] - 4 # removing 4 aux columns
    maxlnlike = chain[:,-4].max()

    bic = np.log(nobs)*nparams - 2.0*maxlnlike
    if log_evidence:
        return (bic, -0.5*bic)
    else:
        return bic

<<<<<<< HEAD
class HyperModel(object):
    """
    Class to define hyper-model that is the concatenation of all models.
    """

    def __init__(self, models, log_weights=None):
        self.models = models
        self.num_models = len(self.models)
        self.log_weights = log_weights

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
        for key in self.snames: self.snames[key] = list(set(self.snames[key]))

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
        nmodel = int(np.rint(x[idx]))

        # find parameters of active model
        q = []
        for par in self.models[nmodel].param_names:
            idx = self.param_names.index(par)
            q.append(x[idx])

        # only active parameters enter likelihood
        active_lnlike = self.models[nmodel].get_lnlikelihood(q)

        if self.log_weights is not None:
            active_lnlike += self.log_weights[nmodel]

        return active_lnlike

    def get_lnprior(self, x):

        # find model index variable
        idx = list(self.param_names).index('nmodel')
        nmodel = int(np.rint(x[idx]))

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

        # DM dip prior draw
        if 'dmexp' in '\t'.join(jp.snames):
            print('Adding DM exponential dip prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)
    
        # DM cusp prior draw
        if 'dm_cusp' in jp.snames:
            print('Adding DM exponential cusp prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)
            
        # DM annual prior draw
        if 'dmx_signal' in jp.snames:
            print('Adding DMX prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

        # Ephemeris prior draw
        if 'd_jupiter_mass' in self.param_names:
            print('Adding ephemeris model prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

        # GWB uniform distribution draw
        if 'gw_log10_A' in self.param_names:
            print('Adding GWB uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

        # Dipole uniform distribution draw
        if 'dipole_log10_A' in self.param_names:
            print('Adding dipole uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

        # Monopole uniform distribution draw
        if 'monopole_log10_A' in self.param_names:
            print('Adding monopole uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

        # BWM prior draw
        if 'bwm_log10_A' in self.param_names:
            print('Adding BWM prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

        # CW prior draw
        if 'cw_log10_h' in self.param_names:
            print('Adding CW prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)

        # Model index distribution draw
        if 'nmodel' in self.param_names:
            print('Adding nmodel uniform distribution draws...\n')
            sampler.addProposalToCycle(self.draw_from_nmodel_prior, 20)

        return sampler


    def get_process_timeseries(self, psr, chain, burn, comp='DM',
                               mle=False, model=0):
        """
        Construct a time series realization of various constrained processes.
        :param psr: etnerprise pulsar object
        :param chain: MCMC chain from sampling all models
        :param burn: desired number of initial samples to discard
        :param comp: which process to reconstruct? (red noise or DM) [default=DM]
        :param mle: create time series from ML of GP hyper-parameters? [default=False]
        :param model: which sub-model within the super-model to reconstruct from? [default=0]

        :return ret: time-series of the reconstructed process
        """

        wave = 0
        pta = self.models[model]
        model_chain = chain[np.rint(chain[:,-5])==model,:]

        # get parameter dictionary
        if mle:
            ind = np.argmax(model_chain[:, -4])
        else:
            ind = np.random.randint(burn, model_chain.shape[0])
        params = {par: model_chain[ind, ct]
                  for ct, par in enumerate(self.param_names)
                  if par in pta.param_names}

        # deterministic signal part
        wave += pta.get_delay(params=params)[0]

        # get linear parameters
        Nvec = pta.get_ndiag(params)[0]
        phiinv = pta.get_phiinv(params, logdet=False)[0]
        T = pta.get_basis(params)[0]

        d = pta.get_TNr(params)[0]
        TNT = pta.get_TNT(params)[0]

        # Red noise piece
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        try:
            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, d)/s)
            Li = u * np.sqrt(1/s)
        except np.linalg.LinAlgError:

            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, d)
            u, s, _ = sl.svd(Sigi)
            Li = u * np.sqrt(1/s)

        b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

        # find basis indices
        pardict = {}
        for sc in pta._signalcollections:
            ntot = 0
            for sig in sc._signals:
                if sig.signal_type == 'basis':
                    basis = sig.get_basis(params=params)
                    nb = basis.shape[1]
                    pardict[sig.signal_name] = np.arange(ntot, nb+ntot)
                    ntot += nb

        # DM quadratic + GP
        if comp == 'DM':
            idx = pardict['dm_gp']
            wave += np.dot(T[:,idx], b[idx])
            ret = wave * (psr.freqs**2 * const.DM_K * 1e12)
        elif comp == 'red':
            idx = pardict['red noise']
            wave += np.dot(T[:,idx], b[idx])
            ret = wave
        elif comp == 'FD':
            idx = pardict['FD']
            wave += np.dot(T[:,idx], b[idx])
            ret = wave
        elif comp == 'all':
            wave += np.dot(T, b)
            ret = wave
        else:
            ret = wave

        return ret

#########Solar Wind Modeling########

=======
>>>>>>> 321e941210a10d08d4a276d74b3f6481dc895fc3
def mask_filter(psr, mask):
    """filter given pulsar data by user defined mask"""
    psr._toas = psr._toas[mask]
    psr._toaerrs = psr._toaerrs[mask]
    psr._residuals = psr._residuals[mask]
    psr._ssbfreqs = psr._ssbfreqs[mask]

    psr._designmatrix = psr._designmatrix[mask, :]
    dmx_mask = np.sum(psr._designmatrix, axis=0) != 0.0
    psr._designmatrix = psr._designmatrix[:, dmx_mask]

    for key in psr._flags:
        psr._flags[key] = psr._flags[key][mask]

    if psr._planetssb is not None:
        psr._planetssb = psr.planetssb[mask, :, :]

    psr.sort_data()


AU_light_sec = const.AU / const.c #1 AU in light seconds
AU_pc = const.AU / const.pc #1 AU in parsecs (for DM normalization)

def _dm_solar_close(n_earth,r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)

def _dm_solar(n_earth,theta_impact,r_earth):
    return ( (np.pi - theta_impact) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta_impact))) )


def dm_solar(n_earth,theta_impact,r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta_impact: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 20007 for more details.
    """
    return np.where(np.pi - theta_impact >= 1e-5,
                    _dm_solar(n_earth, theta_impact, r_earth),
                    _dm_solar_close(n_earth, r_earth))

def solar_wind(psr, n_earth=8.7):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    dispersion measure due to the solar wind and solar impact angle.

    param:: psr enterprise Pulsar objects
    param:: n_earth, proton density at 1 Au

    returns: DM due to solar wind (pc/cm^3) and solar impact angle (rad)
    """
    earth = psr.planetssb[:, 2, :3]
    R_earth = np.sqrt(np.einsum('ij,ij->i',earth, earth))
    Re_cos_theta_impact = np.einsum('ij,ij->i',earth, psr.pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)

    dm_sol_wind = dm_solar(n_earth, theta_impact, R_earth)

    return dm_sol_wind, theta_impact


def solar_wind_mask(psrs,angle_cutoff=None,std_cutoff=None):
    """
    Convenience function for masking TOAs lower than a certain solar impact.
    Alternatively one can set a standard deviation limit, so that all TOAs above
        a certain st dev away from the solar wind DM average for a given pulsar
        can be excised.
    param:: psrs list of enterprise Pulsar objects
    param:: angle_cutoff (degrees) Mask TOAs within this angle
    param:: std_cutoff (float number) Number of St. Devs above average to excise

    returns:: dictionary of maskes for each pulsar
    """
    solar_wind_mask = {}
    if std_cutoff and angle_cutoff:
        raise ValueError('Can not make mask using St Dev and Angular Cutoff!!')
    if std_cutoff:
        for ii,p in enumerate(psrs):
            dm_sw, _ = solar_wind(p)
            std = np.std(dm_sw)
            mean = np.mean(dm_sw)
            solar_wind_mask[p.name] = np.where(dm_sw < (mean + std_cutoff * std),
                                               True, False)
    elif angle_cutoff:
        angle_cutoff = np.deg2rad(angle_cutoff)
        for ii,p in enumerate(psrs):
            _, impact_ang = solar_wind(p)
            solar_wind_mask[p.name] = np.where(impact_ang > angle_cutoff,
                                               True, False)

    return solar_wind_mask

#########Empirical Distributions########

# class used to define a 1D empirical distribution
# based on posterior from another MCMC
class EmpiricalDistribution1D(object):

    def __init__(self, param_name, samples, bins):
        """
            :param samples: samples for hist
            :param bins: edges to use for hist (left and right)
            make sure bins cover whole prior!
            """
        self.ndim = 1
        self.param_name = param_name
        self._Nbins = len(bins)-1
        hist, x_bins = np.histogram(samples, bins=bins)

        self._edges = x_bins[:-1]
        self._wids = np.diff(x_bins)

        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / float(counts) / self._wids
        self._cdf = np.cumsum((self._pdf*self._wids).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)

        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = self._edges[idx] + self._wids[idx]*np.random.rand()
        return np.array(samp)

    def prob(self, params):
        ix = min(np.searchsorted(self._edges, params),
                 self._Nbins-1)

        return self._pdf[ix]

    def logprob(self, params):
        ix = min(np.searchsorted(self._edges, params),
                 self._Nbins-1)

        return self._logpdf[ix]


# class used to define a 2D empirical distribution
# based on posteriors from another MCMC
class EmpiricalDistribution2D(object):
    def __init__(self, param_names, samples, bins):
        """
            :param samples: samples for hist
            :param bins: edges to use for hist (left and right)
            make sure bins cover whole prior!
            """
        self.ndim = 2
        self.param_names = param_names
        self._Nbins = [len(b)-1 for b in bins]
        hist, x_bins, y_bins = np.histogram2d(*samples, bins=bins)

        self._edges = np.array([x_bins[:-1], y_bins[:-1]])
        self._wids = np.diff([x_bins, y_bins])

        area = np.outer(*self._wids)
        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / counts / area
        self._cdf = np.cumsum((self._pdf*area).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)

        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = [self._edges[ii, idx[ii]] + self._wids[ii, idx[ii]]*np.random.rand()
                for ii in range(2)]
        return np.array(samp)

    def prob(self, params):
        ix, iy = [min(np.searchsorted(self._edges[ii], params[ii]),
                      self._Nbins[ii]-1) for ii in range(2)]

        return self._pdf[ix, iy]

    def logprob(self, params):
        ix, iy = [min(np.searchsorted(self._edges[ii], params[ii]),
                      self._Nbins[ii]-1) for ii in range(2)]

        return self._logpdf[ix, iy]


def make_empirical_distributions(paramlist, params, chain,
                                 burn=0, nbins=41, filename='distr.pkl'):
    """
        Utility function to construct empirical distributions.
        :param paramlist: a list of parameter names,
                          either single parameters or pairs of parameters
        :param params: list of all parameter names for the MCMC chain
        :param chain: MCMC chain from a previous run
        :param burn: desired number of initial samples to discard
        :param nbins: number of bins to use for the empirical distributions

        :return distr: list of empirical distributions
        """

    distr = []

    for pl in paramlist:

        if type(pl) is not list:

            pl = [pl]

        if len(pl) == 1:

            # get the parameter index
            idx = params.index(pl[0])

            # get the bins for the histogram
            bins = np.linspace(min(chain[burn:, idx]), max(chain[burn:, idx]), nbins)

            new_distr = EmpiricalDistribution1D(pl[0], chain[burn:, idx], bins)

            distr.append(new_distr)

        elif len(pl) == 2:

            # get the parameter indices
            idx = [params.index(pl1) for pl1 in pl]

            # get the bins for the histogram
            bins = [np.linspace(min(chain[burn:, i]), max(chain[burn:, i]), nbins) for i in idx]

            new_distr = EmpiricalDistribution2D(pl, chain[burn:, idx].T, bins)

            distr.append(new_distr)

        else:
            print('Warning: only 1D and 2D empirical distributions are currently allowed.')

    # save the list of empirical distributions as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(distr, f, protocol=2)
        
    print('The empirical distributions have been pickled to {0}.'.format(filename))
