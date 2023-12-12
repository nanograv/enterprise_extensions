# -*- coding: utf-8 -*-

import glob
import os
import pickle
import platform

import healpy as hp
import numpy as np
from PTMCMCSampler import __version__ as __vPTMCMC__
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import __version__
from enterprise_extensions.empirical_distr import (EmpiricalDistribution1D,
                                                   EmpiricalDistribution1DKDE,
                                                   EmpiricalDistribution2D,
                                                   EmpiricalDistribution2DKDE)


def extend_emp_dists(pta, emp_dists, npoints=100_000,
                     save_ext_dists=False, outdir='./chains'):

    new_emp_dists = []
    modified = False  # check if anything was changed
    for emp_dist in emp_dists:
        if (isinstance(emp_dist, EmpiricalDistribution2D) or
           isinstance(emp_dist, EmpiricalDistribution2DKDE)):
            # check if we need to extend the distribution
            prior_ok=True
            for ii, (param, nbins) in enumerate(zip(emp_dist.param_names,
                                                    emp_dist._Nbins)):
                param_names = [par.name for par in pta.params]
                if param not in param_names:  # skip if one of the parameters isn't in our PTA object
                    short_par = '_'.join(param.split('_')[:-1])  # make sure we aren't skipping priors with size!=None
                    if short_par in param_names:
                        param = short_par
                    else:
                        continue
                # check 2 conditions on both params to make sure that they cover their priors
                # skip if emp dist already covers the prior
                param_idx = param_names.index(param)
                if pta.params[param_idx].type not in ['uniform', 'normal']:
                    msg = '{} cannot be covered automatically by the empirical distribution\n'.format(pta.params[param_idx].prior)
                    msg += 'Please check that your prior is covered by the empirical distribution.\n'
                    print(msg)
                    continue
                elif pta.params[param_idx].type == 'uniform':
                    prior_min = pta.params[param_idx].prior._defaults['pmin']
                    prior_max = pta.params[param_idx].prior._defaults['pmax']
                elif pta.params[param_idx].type == 'normal':
                    prior_min = pta.params[param_idx].prior._defaults['mu'] - 10 * pta.params[param_idx].prior._defaults['sigma']
                    prior_max = pta.params[param_idx].prior._defaults['mu'] + 10 * pta.params[param_idx].prior._defaults['sigma']

                # no need to extend if histogram edges are already prior min/max
                if isinstance(emp_dist, EmpiricalDistribution2D):
                    if not (emp_dist._edges[ii][0] == prior_min and
                            emp_dist._edges[ii][-1] == prior_max):
                        prior_ok = False
                        continue
                elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                    if not (emp_dist.minvals[ii] == prior_min and
                            emp_dist.maxvals[ii] == prior_max):
                        prior_ok=False
                        continue
            if prior_ok:
                new_emp_dists.append(emp_dist)
                continue
            modified = True
            samples = np.zeros((npoints, emp_dist.draw().shape[0]))
            for ii in range(npoints):  # generate samples from old emp dist
                samples[ii] = emp_dist.draw()
            new_bins = []
            minvals = []
            maxvals = []
            idxs_to_remove = []
            for ii, (param, nbins) in enumerate(zip(emp_dist.param_names,
                                                    emp_dist._Nbins)):
                param_idx = param_names.index(param)
                if pta.params[param_idx].type == 'uniform':
                    prior_min = pta.params[param_idx].prior._defaults['pmin']
                    prior_max = pta.params[param_idx].prior._defaults['pmax']
                elif pta.params[param_idx].type == 'normal':
                    prior_min = pta.params[param_idx].prior._defaults['mu'] - 10 * pta.params[param_idx].prior._defaults['sigma']
                    prior_max = pta.params[param_idx].prior._defaults['mu'] + 10 * pta.params[param_idx].prior._defaults['sigma']
                # drop samples that are outside the prior range (in case prior is smaller than samples)
                if isinstance(emp_dist, EmpiricalDistribution2D):
                    samples[(samples[:, ii] < prior_min) | (samples[:, ii] > prior_max), ii] = -np.inf
                elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                    idxs_to_remove.extend(np.arange(npoints)[(samples[:, ii] < prior_min) | (samples[:, ii] > prior_max)])
                    minvals.append(prior_min)
                    maxvals.append(prior_max)
                # new distribution with more bins this time to extend it all the way out in same style as above.
                new_bins.append(np.linspace(prior_min, prior_max, nbins + 40))
            samples = np.delete(samples, idxs_to_remove, axis=0)
            if isinstance(emp_dist, EmpiricalDistribution2D):
                new_emp = EmpiricalDistribution2D(emp_dist.param_names,
                                                  samples.T, new_bins)
            elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                # new distribution with more bins this time to extend it all the way out in same style as above.
                new_emp = EmpiricalDistribution2DKDE(emp_dist.param_names,
                                                     samples.T, minvals=minvals,
                                                     maxvals=maxvals,
                                                     nbins=nbins+40,
                                                     bandwidth=emp_dist.bandwidth)
            new_emp_dists.append(new_emp)

        elif (isinstance(emp_dist, EmpiricalDistribution1D) or
              isinstance(emp_dist, EmpiricalDistribution1DKDE)):
            param_names = [par.name for par in pta.params]
            if emp_dist.param_name not in param_names:  # skip if one of the parameters isn't in our PTA object
                short_par = '_'.join(emp_dist.param_name.split('_')[:-1])  # make sure we aren't skipping priors with size!=None
                if short_par in param_names:
                    param = short_par
                else:
                    continue
            else:
                param = emp_dist.param_name
            param_idx = param_names.index(param)
            if pta.params[param_idx].type not in ['uniform', 'normal']:
                msg = 'This prior cannot be covered automatically by the empirical distribution\n'
                msg += 'Please check that your prior is covered by the empirical distribution.\n'
                print(msg)
                continue
            if pta.params[param_idx].type == 'uniform':
                prior_min = pta.params[param_idx].prior._defaults['pmin']
                prior_max = pta.params[param_idx].prior._defaults['pmax']
            elif pta.params[param_idx].type == 'uniform':
                prior_min = pta.params[param_idx].prior._defaults['mu'] - 10 * pta.params[param_idx].prior._defaults['sigma']
                prior_max = pta.params[param_idx].prior._defaults['mu'] + 10 * pta.params[param_idx].prior._defaults['sigma']
            # check 2 conditions on param to make sure that it covers the prior
            # skip if emp dist already covers the prior
            if isinstance(emp_dist, EmpiricalDistribution1D):
                if emp_dist._edges[0] == prior_min and emp_dist._edges[-1] == prior_max:
                    new_emp_dists.append(emp_dist)
                    continue
            elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                if emp_dist.minval == prior_min and emp_dist.maxval == prior_max:
                    new_emp_dists.append(emp_dist)
                    continue
            modified = True
            samples = np.zeros((npoints, 1))
            for ii in range(npoints):  # generate samples from old emp dist
                samples[ii] = emp_dist.draw()
            new_bins = []
            idxs_to_remove = []
            # drop samples that are outside the prior range (in case prior is smaller than samples)
            if isinstance(emp_dist, EmpiricalDistribution1D):
                samples[(samples < prior_min) | (samples > prior_max)] = -np.inf
            elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                idxs_to_remove.extend(np.arange(npoints)[(samples.squeeze() < prior_min) | (samples.squeeze() > prior_max)])

            samples = np.delete(samples, idxs_to_remove, axis=0)
            new_bins = np.linspace(prior_min, prior_max, emp_dist._Nbins + 40)
            if isinstance(emp_dist, EmpiricalDistribution1D):
                new_emp = EmpiricalDistribution1D(emp_dist.param_name, samples, new_bins)
            elif isinstance(emp_dist, EmpiricalDistribution1DKDE):
                new_emp = EmpiricalDistribution1DKDE(emp_dist.param_name, samples,
                                                     minval=prior_min, maxval=prior_max,
                                                     bandwidth=emp_dist.bandwidth)
            new_emp_dists.append(new_emp)
        else:
            print('Unable to extend class of unknown type to the edges of the priors.')
            new_emp_dists.append(emp_dist)
            continue

    if save_ext_dists and modified:  # if user wants to save them, and they have been modified...
        with open(outdir + '/new_emp_dists.pkl', 'wb') as f:
            pickle.dump(new_emp_dists, f)
    return new_emp_dists


class UserDraw(object):
    """object for user specified proposal distributions
    """
    def __init__(self, idxs, samplers, log_qs=None, name=None):
        """
        :param idxs: list of parameter indices to use for this jump
        :param samplers: dict of callable samplers
            keys should include all idxs
        :param lqxys: dict of callable log proposal distributions
            keys should include all idxs
            for symmetric proposals set `log_qs=None`, then `log_qxy=0`
        :param name: name for PTMCMC bookkeeping
        """
        self.idxs = idxs
        self.samplers = samplers
        self.log_qs = log_qs

        if name is None:
            namestr = 'draw'
            for ii in samplers.keys():
                namestr += '_{}'.format(ii)
            self.__name__ = namestr
        else:
            self.__name__ = name

    def __call__(self, x, iter, beta):
        """proposal from parameter prior distribution
        """
        y = x.copy()

        # draw parameter from idxs
        ii = np.random.choice(self.idxs)

        try:  # vector parameter
            y[ii] = self.samplers[ii]()[0]
        except (IndexError, TypeError):
            y[ii] = self.samplers[ii]()

        if self.log_qs is None:
            lqxy = 0
        else:
            lqxy = self.log_qs[ii](x[ii]) - self.log_qs[ii](y[ii])

        return y, lqxy


def build_prior_draw(pta_params, parlist, name=None):
    """create a callable object to perfom a prior draw
    :param pta_param_names:
        parameters from a PTA object (pta.params)
    :param parlist:
        single string or list of strings of parameter name(s) to
        use for this jump.
    :param name:
        display name for PTMCMCSampler bookkeeping
    """
    if not isinstance(parlist, list):
        parlist = [parlist]

    pta_param_names = []
    for p in pta_params:
        if p.size:
            for ii in range(p.size):
                pta_param_names.append(p.name + '_' + str(ii))
        else:
            pta_param_names.append(p.name)
    idxs = [pta_param_names.index(par) for par in parlist]

    # parameter map
    pmap = []
    ct = 0
    for ii, pp in enumerate(pta_params):
        size = pp.size or 1
        for nn in range(size):
            pmap.append(ii)
        ct += size

    sampler = {ii: pta_params[pmap[ii]].sample for ii in idxs}
    log_q = {ii: pta_params[pmap[ii]].get_logpdf for ii in idxs}

    return UserDraw(idxs, sampler, log_q, name=name)


class EmpDistrDraw(object):
    """object for empirical proposal distributions
    """
    def __init__(self, distr, parlist, Nmax=None, name=None):
        """
        :param distr: list of EmpiricalDistribution2D or EmpiricalDistribution1D objects
        :param parlist: list of all model params (pta.param_names)
            to figure out which indices to use
        :param Nmax: maximum number of distributions to propose
            simultaneously
        :param name: name for PTMCMC bookkeeping
        """
        self._distr = distr
        self.Nmax = Nmax if Nmax else len(distr)
        self.__name__ = name if name else 'draw_empirical'

        # which model indices go with which distr?
        for dd in self._distr:
            dd._idx = []
            if "2D" in str(type(dd)):
                for pp in parlist:
                    if pp in dd.param_names:
                        dd._idx.append(parlist.index(pp))
            else:
                for pp in parlist:
                    if pp == dd.param_name:
                        dd._idx.append(parlist.index(pp))

    def __call__(self, x, iter, beta):
        """propose a move from empirical distribution
        """
        y = x.copy()
        lqxy = 0

        # which distrs to propose moves
        if self.Nmax == 1:
            N = 1
        else:
            N = np.random.randint(1, self.Nmax)
        which = np.random.choice(self._distr, size=N, replace=False)

        for distr in which:
            old = x[distr._idx]
            new = distr.draw()
            y[distr._idx] = new

            lqxy += (distr.logprob(old) -
                     distr.logprob(new))

        return y, lqxy


class JumpProposal(object):

    def __init__(self, pta, snames=None, empirical_distr=None, f_stat_file=None, save_ext_dists=False, outdir='./chains'):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.psrnames = pta.pulsars
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.plist = [p.name for p in pta.params]

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
            for key in self.snames:
                self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames:
                self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames

        # empirical distributions
        if isinstance(empirical_distr, list):
            # check if a list of emp dists is provided
            self.empirical_distr = empirical_distr

        # check if a directory of empirical dist pkl files are provided
        elif empirical_distr is not None and os.path.isdir(empirical_distr):

            dir_files = glob.glob(empirical_distr+'*.pkl')  # search for pkls

            pickled_distr = np.array([])
            for idx, emp_file in enumerate(dir_files):
                try:
                    with open(emp_file, 'rb') as f:
                        pickled_distr = np.append(pickled_distr, pickle.load(f))
                except:
                    try:
                        with open(emp_file, 'rb') as f:
                            pickled_distr = np.append(pickled_distr, pickle.load(f))
                    except:
                        print(f'\nI can\'t open the empirical distribution pickle file at location {idx} in list!')
                        print("Empirical distributions set to 'None'")
                        pickled_distr = None
                        break

            self.empirical_distr = pickled_distr

        # check if single pkl file provided
        elif empirical_distr is not None and os.path.isfile(empirical_distr):  # checking for single file
            try:
                # try opening the file
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                # second attempt at opening the file
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                # if the second attempt fails...
                except:
                    print('\nI can\'t open the empirical distribution pickle file!')
                    pickled_distr = None

            self.empirical_distr = pickled_distr

        # all other cases - emp dists set to None
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx, d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) >= 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
                # extend empirical_distr here:
                print('Extending empirical distributions to priors...\n')
                self.empirical_distr = extend_emp_dists(pta, self.empirical_distr, npoints=100_000,
                                                        save_ext_dists=save_ext_dists, outdir=outdir)
            else:
                self.empirical_distr = None

        if empirical_distr is not None and self.empirical_distr is None:
            # if an emp dist path is provided, but fails the code, this helpful msg is provided
            print("Adding empirical distributions failed!! Empirical distributions set to 'None'\n")

        # F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile['freqs']
            self.fe = npzfile['fe']

    def fe_jump(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        fe_limit = np.max(self.fe)

        # draw skylocation and frequency from f-stat map
        accepted = False
        while accepted is False:
            log_f_new = self.params[self.pimap['log10_fgw']].sample()
            f_idx = (np.abs(np.log10(self.fe_freqs) - log_f_new)).argmin()

            gw_theta = np.arccos(self.params[self.pimap['cos_gwtheta']].sample())
            gw_phi = self.params[self.pimap['gwphi']].sample()
            hp_idx = hp.ang2pix(hp.get_nside(self.fe), gw_theta, gw_phi)

            fe_new_point = self.fe[f_idx, hp_idx]
            if np.random.uniform()<(fe_new_point/fe_limit):
                accepted = True

        # draw other parameters from prior
        cos_inc = self.params[self.pimap['cos_inc']].sample()
        psi = self.params[self.pimap['psi']].sample()
        phase0 = self.params[self.pimap['phase0']].sample()
        log10_h = self.params[self.pimap['log10_h']].sample()

        # put new parameters into q
        for param_name, new_param in zip(['log10_fgw', 'gwphi', 'cos_gwtheta', 'cos_inc', 'psi', 'phase0', 'log10_h'],
                                         [log_f_new, gw_phi, np.cos(gw_theta), cos_inc, psi, phase0, log10_h]):
            q[self.pimap[param_name]] = new_param

        # calculate Hastings ratio
        log_f_old = x[self.pimap['log10_fgw']]
        f_idx_old = (np.abs(np.log10(self.fe_freqs) - log_f_old)).argmin()

        gw_theta_old = np.arccos(x[self.pimap['cos_gwtheta']])
        gw_phi_old = x[self.pimap['gwphi']]
        hp_idx_old = hp.ang2pix(hp.get_nside(self.fe), gw_theta_old, gw_phi_old)

        fe_old_point = self.fe[f_idx_old, hp_idx_old]
        if fe_old_point>fe_limit:
            fe_old_point = fe_limit

        log10_h_old = x[self.pimap['log10_h']]
        phase0_old = x[self.pimap['phase0']]
        psi_old = x[self.pimap['psi']]
        cos_inc_old = x[self.pimap['cos_inc']]

        hastings_extra_factor = self.params[self.pimap['log10_h']].get_pdf(log10_h_old)
        hastings_extra_factor *= 1/self.params[self.pimap['log10_h']].get_pdf(log10_h)
        hastings_extra_factor = self.params[self.pimap['phase0']].get_pdf(phase0_old)
        hastings_extra_factor *= 1/self.params[self.pimap['phase0']].get_pdf(phase0)
        hastings_extra_factor = self.params[self.pimap['psi']].get_pdf(psi_old)
        hastings_extra_factor *= 1/self.params[self.pimap['psi']].get_pdf(psi)
        hastings_extra_factor = self.params[self.pimap['cos_inc']].get_pdf(cos_inc_old)
        hastings_extra_factor *= 1/self.params[self.pimap['cos_inc']].get_pdf(cos_inc)

        lqxy = np.log(fe_old_point/fe_new_point * hastings_extra_factor)

        return q, float(lqxy)


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = list(set(par for par in pars if pars.count(par) > 1))
    ipars = [par for par in pars if par not in gpars]

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


def get_psr_groups(pta):
    groups = []
    for psr in pta.pulsars:
        grp = [pta.param_names.index(par)
               for par in pta.param_names if psr in par]
        groups.append(grp)
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


def save_runtime_info(pta, outdir='chains', human=None):
    """save system info, enterprise PTA.summary, and other metadata to file
    """
    # save system info and enterprise PTA.summary to single file
    sysinfo = {}
    if human is not None:
        sysinfo.update({"human": human})
    sysinfo.update(platform.uname()._asdict())

    with open(os.path.join(outdir, "runtime_info.txt"), "w") as fout:
        for field, data in sysinfo.items():
            fout.write(field + " : " + data + "\n")
        fout.write("\n")
        fout.write("enterprise_extensions v" + __version__ +"\n")
        fout.write("PTMCMCSampler v" + __vPTMCMC__ +"\n")
        fout.write(pta.summary())

    # save paramter list
    with open(os.path.join(outdir, "pars.txt"), "w") as fout:
        for pname in pta.param_names:
            fout.write(pname + "\n")

    # save list of priors
    with open(os.path.join(outdir, "priors.txt"), "w") as fout:
        for pp in pta.params:
            fout.write(pp.__repr__() + "\n")


def setup_sampler(pta, outdir='chains', resume=False,
                  empirical_distr=None, groups=None, human=None,
                  save_ext_dists=False, loglkwargs={}, logpkwargs={}):
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

    save_ext_dists: saves distributions that have been extended to
    cover priors as a pickle to the outdir folder. These can then
    be loaded later as distributions to save a minute at the start
    of the run.
    """

    # dimension of parameter space
    params = pta.param_names
    ndim = len(params)

    # initial jump covariance matrix
    if os.path.exists(outdir+'/cov.npy') and resume:
        cov = np.load(outdir+'/cov.npy')

        # check that the one we load is the same shape as our data
        cov_new = np.diag(np.ones(ndim) * 0.1**2)
        if cov.shape != cov_new.shape:
            msg = 'The covariance matrix (cov.npy) in the output folder is '
            msg += 'the wrong shape for the parameters given. '
            msg += 'Start with a different output directory or '
            msg += 'change resume to False to overwrite the run that exists.'

            raise ValueError(msg)
    else:
        cov = np.diag(np.ones(ndim) * 0.1**2)

    # parameter groupings
    if groups is None:
        groups = get_parameter_groups(pta)
        groups.extend(get_psr_groups(pta))

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                     outDir=outdir, resume=resume, loglkwargs=loglkwargs,
                     logpkwargs=logpkwargs)

    save_runtime_info(pta, sampler.outDir, human)

    # additional jump proposals
    jp = JumpProposal(pta, empirical_distr=empirical_distr, save_ext_dists=save_ext_dists, outdir=outdir)
    sampler.jp = jp

    # always add draw from prior
    sampler.addProposalToCycle(build_prior_draw(pta.params, pta.param_names,
                                                name='draw_from_prior'), 5)

    # try adding empirical proposals
    if empirical_distr is not None:
        print('Attempting to add empirical proposals...\n')
        sampler.addProposalToCycle(EmpDistrDraw(jp.empirical_distr,
                                                pta.param_names,
                                                name='draw_from_empirical_distr'), 10)

    # list of typical signal names
    snames = ['red noise', 'dm_gp', 'chrom_gp',
              'dmx_signal', 'phys_ephem', 'bwm', 'fdm', 'cw', 'gp_sw',
              'linear timing model', 'ecorr_sherman-morrison',
              'measurement_noise', 'tnequad']

    for sname in snames:
        # adding prior draws
        if (sname in jp.snames) and (len(jp.snames[sname]) >= 1):
            print(f'Adding {sname} prior draws...\n')

            param_names = []
            for p in jp.snames[sname]:
                if p.size:
                    for ii in range(p.size):
                        param_names.append(p.name + '_' + str(ii))
                else:
                    param_names.append(p.name)

            sampler.addProposalToCycle(build_prior_draw(pta.params,
                                                        param_names,
                                                        name='draw_from_'+sname), 10)

    # adding other signal draws
    param_names = ['dipole', 'monopole', 'hd', 'log10_rho',
                   'dmexp', 'dm_cusp', 'dm_s1yr']

    for p in param_names:
        params = [par for par in pta.param_names if p in par]
        if len(params) >= 1:
            print(f'Adding {p} prior draws...\n')
            sampler.addProposalToCycle(build_prior_draw(pta.params, params,
                                                        name='draw_from_'+p), 10)

    return sampler
