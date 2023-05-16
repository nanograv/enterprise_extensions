# -*- coding: utf-8 -*-

import glob
import os
import pickle
import platform
import warnings

import healpy as hp
import numpy as np
from scipy.stats import multivariate_normal as mv_norm

from PTMCMCSampler import __version__ as __vPTMCMC__
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import __version__
from enterprise_extensions.empirical_distr import (EmpiricalDistribution1D,
                                                   EmpiricalDistribution1DKDE,
                                                   EmpiricalDistribution2D,
                                                   EmpiricalDistribution2DKDE)


def extend_emp_dists(pta, emp_dists, npoints=100_000, save_ext_dists=False, outdir='./chains'):
    new_emp_dists = []
    modified = False  # check if anything was changed
    for emp_dist in emp_dists:
        if isinstance(emp_dist, EmpiricalDistribution2D) or isinstance(emp_dist, EmpiricalDistribution2DKDE):
            # check if we need to extend the distribution
            prior_ok=True
            for ii, (param, nbins) in enumerate(zip(emp_dist.param_names, emp_dist._Nbins)):
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
                    if not (emp_dist._edges[ii][0] == prior_min and emp_dist._edges[ii][-1] == prior_max):
                        prior_ok = False
                        continue
                elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                    if not (emp_dist.minvals[ii] == prior_min and emp_dist.maxvals[ii] == prior_max):
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
            for ii, (param, nbins) in enumerate(zip(emp_dist.param_names, emp_dist._Nbins)):
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
                new_emp = EmpiricalDistribution2D(emp_dist.param_names, samples.T, new_bins)
            elif isinstance(emp_dist, EmpiricalDistribution2DKDE):
                # new distribution with more bins this time to extend it all the way out in same style as above.
                new_emp = EmpiricalDistribution2DKDE(emp_dist.param_names, samples.T, minvals=minvals, maxvals=maxvals, nbins=nbins+40, bandwidth=emp_dist.bandwidth)
            new_emp_dists.append(new_emp)

        elif isinstance(emp_dist, EmpiricalDistribution1D) or isinstance(emp_dist, EmpiricalDistribution1DKDE):
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


class JumpProposal(object):

    def __init__(self, pta, snames=None, empirical_distr=None, f_stat_file=None, save_ext_dists=False, outdir='./chains', timing=False, psr=None, sampler=None, restrict_mass=True):
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
            self.pmap[str(p)] = slice(ct, ct + size)
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
                    with open(emp_file, "rb") as f:
                        pickled_distr = np.append(pickled_distr, pickle.load(f))
                except:
                    try:
                        with open(emp_file, 'rb') as f:
                            pickled_distr = np.append(pickled_distr, pickle.load(f))
                    except:
                        print(
                            f"\nI can't open the empirical distribution pickle file at location {idx} in list!"
                        )
                        print("Empirical distributions set to 'None'")
                        pickled_distr = None
                        break

            self.empirical_distr = pickled_distr

        # check if single pkl file provided
        elif empirical_distr is not None and os.path.isfile(
            empirical_distr
        ):  # checking for single file
            try:
                # try opening the file
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                # second attempt at opening the file
                try:
                    with open(empirical_distr, "rb") as f:
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
                    if (
                        d.param_names[0] in pta.param_names
                        and d.param_names[1] in pta.param_names
                    ):
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
            print(
                "Adding empirical distributions failed!! Empirical distributions set to 'None'\n"
            )

        # F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile["freqs"]
            self.fe = npzfile["fe"]

        if timing:
            self.restrict_mass = restrict_mass
            self.max_emergency_iter = 1000  # Prevents infinite loop and from sample taking too long
            self.sampler = sampler

            if not psr:
                raise ValueError("Must include a pulsar object in JumpProposal.")
            if not sampler:
                raise ValueError("Must include a sampler object in JumpProposal.")

            tm_groups = get_timing_groups(pta)
            tm_idx = np.unique([inner for outer in tm_groups for inner in outer])
            tm_groups.extend(tm_idx)
            self.tm_groups = np.array(tm_groups, dtype=object)

            self.mass_pars = ["A1", "M2", "PB", "SINI", "COSI"]
            self.inclination_flag = "SINI"
            mass_dict = {}
            for par, ii in self.pimap.items():
                if par.split('_')[-1] in self.mass_pars:
                    mass_dict[par] = ii
                if "COSI" in par:
                    self.inclination_flag = "COSI"
            self.mass_idxs = []
            self.unscaled_mass_values = []
            for m_p in self.mass_pars:
                if m_p in psr.tm_params_orig.keys():
                    self.unscaled_mass_values.append((psr.tm_params_orig[m_p][0], psr.tm_params_orig[m_p][1]))
                    for mass_key, mass_idx in mass_dict.items():
                        if m_p in mass_key:
                            self.mass_idxs.append(mass_idx)

            if len(mass_dict.keys()) > len(self.mass_pars):
                raise ValueError("There are the wrong amount of mass parameters being used. Check the dictionary: ", mass_dict)
            # special_pars = ["PX", "SINI", "COSI", "ECC"]
            # Any parameter not centered around zero is considered a "special parameter" that does not draw from a zero-centered Gaussian
            special_pars = []
            for x in [str(y) for y in pta.params if "timing_model" in str(y)]:
                if "Uniform" in x:
                    pmin = float(x.split("Uniform")[-1].split("pmin=")[1].split(",")[0])
                    pmax = float(x.split("Uniform")[-1].split("pmax=")[-1].split(")")[0])
                    if pmin + pmax != 0.0:
                        special_pars.append(x.split(":")[0])
                elif "BoundedNormal" in x:
                    pmin = float(x.split("BoundedNormal")[-1].split("[")[-1].split(",")[0])
                    pmax = float(x.split("BoundedNormal")[-1].split("[")[-1].split(",")[1].split(']')[0])
                    if pmin + pmax != 0.0:
                        special_pars.append(x.split(":")[0])
                else:
                    special_pars.append(x.split(":")[0])

            self.special_idxs = [
                ii
                for par, ii in self.pimap.items()
                if np.any([sp in par for sp in special_pars])
            ]
        else:
            self.restrict_mass = False

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param = np.random.choice(self.params)

        if self.restrict_mass:
            # Used to check parameter will change pulsar mass
            if "timing_model" in str(param).split(":")[0]:
                accepted = False
                emergency_iter = 0  # If the initial sample is bad, the sampler cannot change the mass values
                while not accepted and emergency_iter < self.max_emergency_iter:
                    if emergency_iter > 0:
                        # draw different parameter from mass groups model
                        param = np.random.choice([x for x in self.snames["timing_model"] if str(x).split(":")[0].split('_')[-1] in self.mass_pars])

                    if param.size:
                        idx2 = np.random.randint(0, param.size)
                        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

                    # scalar parameter
                    else:
                        q[self.pmap[str(param)]] = param.sample()
                    accepted = self.check_pulsar_mass(q)
                    emergency_iter += 1

                if emergency_iter > 500:
                    print("draw_from_prior")
                    print("Emergency iter:", emergency_iter)
                    _ = self.check_pulsar_mass(q, print_mp=True)
            else:
                # if vector parameter jump in random component
                if param.size:
                    idx2 = np.random.randint(0, param.size)
                    q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

                # scalar parameter
                else:
                    q[self.pmap[str(param)]] = param.sample()
        else:
            # if vector parameter jump in random component
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "red noise"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

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

                lqxy = self.empirical_distr[distr_idx].logprob(
                    x[idx]
                ) - self.empirical_distr[distr_idx].logprob(q[idx])

                dist = self.empirical_distr[distr_idx]
                # if we fall outside the emp distr support, pull from prior instead
                if x[idx] < dist._edges[0] or x[idx] > dist._edges[-1]:
                    q, lqxy = self.draw_from_prior(x, iter, beta)

                dist = self.empirical_distr[distr_idx]
                # if we fall outside the emp distr support, pull from prior instead
                if x[idx] < dist._edges[0] or x[idx] > dist._edges[-1]:
                    q, lqxy = self.draw_from_prior(x, iter, beta)

            else:
                dist = self.empirical_distr[distr_idx]
                oldsample = [x[self.pnames.index(p)] for p in dist.param_names]
                newsample = dist.draw()

                lqxy = (dist.logprob(oldsample) - dist.logprob(newsample))

                for p, n in zip(dist.param_names, newsample):
                    q[self.pnames.index(p)] = n

                # if we fall outside the emp distr support, pull from prior instead
                for ii in range(len(oldsample)):
                    if oldsample[ii] < dist._edges[ii][0] or oldsample[ii] > dist._edges[ii][-1]:
                        q, lqxy = self.draw_from_prior(x, iter, beta)

        return q, float(lqxy)

    def draw_from_psr_empirical_distr(self, x, iter, beta):
        q = x.copy()
        lqxy = 0

        if self.empirical_distr is not None:

            # make list of empirical distributions with psr name
            psr = np.random.choice(self.psrnames)
            pnames = [ed.param_name if ed.ndim==1 else ed.param_names
                      for ed in self.empirical_distr]

            # Retrieve indices of emp dists with pulsar pars.
            idxs = []
            for par in pnames:
                if isinstance(par, str):
                    if psr in par:
                        idxs.append(pnames.index(par))
                elif isinstance(par, list):
                    if any([psr in p for p in par]):
                        idxs.append(pnames.index(par))

            for idx in idxs:
                if self.empirical_distr[idx].ndim == 1:
                    pidx = self.pimap[self.empirical_distr[idx].param_name]
                    q[pidx] = self.empirical_distr[idx].draw()

                    lqxy += self.empirical_distr[idx].logprob(
                        x[pidx]
                    ) - self.empirical_distr[idx].logprob(q[pidx])

                else:
                    oldsample = [
                        x[self.pnames.index(p)]
                        for p in self.empirical_distr[idx].param_names
                    ]
                    newsample = self.empirical_distr[idx].draw()

                    for p, n in zip(self.empirical_distr[idx].param_names, newsample):
                        q[self.pnames.index(p)] = n

                    lqxy += self.empirical_distr[idx].logprob(
                        oldsample
                    ) - self.empirical_distr[idx].logprob(newsample)

        return q, float(lqxy)

    def draw_from_dm_gp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "dm_gp"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_dm1yr_prior(self, x, iter, beta):

        q = x.copy()

        dm1yr_names = [dmname for dmname in self.pnames if "dm_s1yr" in dmname]
        dmname = np.random.choice(dm1yr_names)
        idx = self.pnames.index(dmname)
        if "log10_Amp" in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif "phase" in dmname:
            q[idx] = np.random.uniform(0, 2 * np.pi)

        return q, 0

    def draw_from_dmexpdip_prior(self, x, iter, beta):

        q = x.copy()

        dmexp_names = [dmname for dmname in self.pnames if "dmexp" in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if "log10_Amp" in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif "log10_tau" in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        elif "sign_param" in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmexpcusp_prior(self, x, iter, beta):

        q = x.copy()

        dmexp_names = [dmname for dmname in self.pnames if "dm_cusp" in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if "log10_Amp" in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif "log10_tau" in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        # elif 't0' in dmname:
        #    q[idx] = np.random.uniform(53393.0, 57388.0)
        elif "sign_param" in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmx_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "dmx_signal"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_chrom_gp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "chrom_gp"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        signal_name = [par for par in self.pnames
                       if ('gw' in par and 'log10_A' in par)][0]
        idx = list(self.pnames).index(signal_name)
        param = self.params[idx]

        q[self.pmap[str(param)]] = np.random.uniform(param.prior._defaults['pmin'], param.prior._defaults['pmax'])

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_dipole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()

        # draw parameter from signal model
        idx = self.pnames.index("dipole_log10_A")
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_monopole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()

        # draw parameter from signal model
        idx = self.pnames.index("monopole_log10_A")
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_altpol_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()

        # draw parameter from signal model
        polnames = [pol for pol in self.pnames if "log10Apol" in pol]
        if "kappa" in self.pnames:
            polnames.append("kappa")
        pol = np.random.choice(polnames)
        idx = self.pnames.index(pol)
        if pol == "log10Apol_tt":
            q[idx] = np.random.uniform(-18, -12)
        elif pol == "log10Apol_st":
            q[idx] = np.random.uniform(-18, -12)
        elif pol == "log10Apol_vl":
            q[idx] = np.random.uniform(-18, -15)
        elif pol == "log10Apol_sl":
            q[idx] = np.random.uniform(-18, -16)
        elif pol == "kappa":
            q[idx] = np.random.uniform(0, 10)

        return q, 0

    def draw_from_ephem_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "phys_ephem"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_bwm_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "bwm"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_fdm_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "fdm"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_cw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "cw"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()

        # draw parameter from signal model
        idx = self.pnames.index("log10_h")
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dm_sw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = "gp_sw"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def draw_from_gw_rho_prior(self, x, iter, beta):
        """
        Jump proposals on free spec
        """

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        parnames = [par.name for par in self.params]
        pname = [pnm for pnm in parnames
                 if ('gw' in pnm and 'rho' in pnm)][0]

        idx = parnames.index(pname)
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

    def draw_from_signal_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0
        std = ['timing_model',
               'linear timing model',
               'red noise',
               'phys_ephem',
               'gw',
               'cw',
               'bwm',
               'fdm',
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
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

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
        par_list = np.concatenate(par_list, axis=None)

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
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
                q[self.pmap[str(param)]]
            )

            return q, float(lqxy)

        name_string = "_".join(name_list)
        draw.__name__ = "draw_from_{}_prior".format(name_string)
        return draw

    def draw_from_par_log_uniform(self, par_dict):
        # Preparing and comparing par_dict.keys() with PTA parameters
        par_list = []
        name_list = []
        for par_name in par_dict.keys():
            pn_list = [n for n in self.plist if par_name in n and "log" in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter dictionary match found between {} and PTA.object."
                              .format(par_dict.keys()))
        par_list = np.concatenate(par_list, axis=None)

        def draw(x, iter, beta):
            """log uniform prior draw function generator for custom par_names.
            par_dict: dictionary with {"par_names":(lower bound,upper bound)}
                                      { "string":(float,float)}

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()

            # draw parameter from signal model
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)
            q[idx] = np.random.uniform(par_dict[par_name][0], par_dict[par_name][1])

            return q, 0

        name_string = "_".join(name_list)
        draw.__name__ = "draw_from_{}_log_uniform".format(name_string)
        return draw

    def draw_from_psr_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from pulsar names
        psr = np.random.choice(self.psrnames)
        idxs = [self.pimap[par] for par in self.pnames if psr in par]
        for idx in idxs:
            q[idx] = self.params[idx].sample()

        # forward-backward jump probability
        first = np.sum([self.params[idx].get_logpdf(x[idx]) for idx in idxs])
        last = np.sum([self.params[idx].get_logpdf(q[idx]) for idx in idxs])

        lqxy = first - last

        return q, float(lqxy)

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
        signal_list = np.concatenate(signal_list, axis=None)

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
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
                q[self.pmap[str(param)]]
            )

            return q, float(lqxy)

        name_string = "_".join(name_list)
        draw.__name__ = "draw_from_{}_signal".format(name_string)
        return draw

    def fe_jump(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        fe_limit = np.max(self.fe)

        # draw skylocation and frequency from f-stat map
        accepted = False
        while accepted is False:
            log_f_new = self.params[self.pimap['log10_fgw']].sample()
            f_idx = (np.abs(np.log10(self.fe_freqs) - log_f_new)).argmin()

            gw_theta = np.arccos(self.params[self.pimap["cos_gwtheta"]].sample())
            gw_phi = self.params[self.pimap["gwphi"]].sample()
            hp_idx = hp.ang2pix(hp.get_nside(self.fe), gw_theta, gw_phi)

            fe_new_point = self.fe[f_idx, hp_idx]
            if np.random.uniform() < (fe_new_point / fe_limit):
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
        if fe_old_point > fe_limit:
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

    # SCAM jump
    def covarianceJumpProposalSCAM(self, x, iter, beta):
        """
        Single Component Adaptive Jump Proposal. This function will occasionally
        jump in more than 1 parameter. It will also occasionally use different
        jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain
        @param sampler: PTMCMCSampler sampler object

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = np.random.randint(0, len(self.sampler.groups))
        ndim = len(self.sampler.groups[jumpind])
        mass_check = any(x in self.mass_idxs for x in self.sampler.groups[jumpind])

        # adjust step size
        prob = np.random.rand()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:

        # standard medium jump
        else:
            scale = 1.0

        # scale = np.random.uniform(0.5, 10)

        # adjust scale based on temperature
        if self.sampler.temp <= 100:
            scale *= np.sqrt(self.sampler.temp)

        # get parmeters in new diagonalized basis
        # y = np.dot(self.U.T, x[self.covinds])

        # make correlated componentwise adaptive jump
        ind = np.unique(np.random.randint(0, ndim, 1))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        if self.restrict_mass:
            # If the initial sample is bad, the sampler cannot change the mass values (hence iter<10)
            # Or if the drawn parameter will change the pulsar mass
            if iter < 10 or mass_check:
                accepted = False
                emergency_iter = 0  # If the initial sample is bad, the sampler cannot change the mass values
                while not accepted and emergency_iter < self.max_emergency_iter:
                    if emergency_iter > 0:
                        q = x.copy()
                        # choose mass group
                        jumpind = [i for (i, x) in enumerate(self.sampler.groups) if set(x).issubset(self.mass_idxs)][0]
                        # draw different parameter from mass groups model
                        ndim = len(self.sampler.groups[jumpind])
                        # make correlated componentwise adaptive jump
                        ind = np.unique(np.random.randint(0, ndim, 1))
                        neff = len(ind)
                        cd = 2.4 / np.sqrt(2 * neff) * scale

                    q[self.sampler.groups[jumpind]] += (np.random.randn() * cd * np.sqrt(self.sampler.S[jumpind][ind]) * self.sampler.U[jumpind][:, ind].flatten())

                    accepted = self.check_pulsar_mass(q)
                    emergency_iter += 1

                if emergency_iter > 500:
                    print("covarianceJumpProposalSCAM")
                    print("Emergency iter:", emergency_iter)
                    _ = self.check_pulsar_mass(q, print_mp=True)
            else:
                # y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
                # q[self.covinds] = np.dot(self.U, y)
                q[self.sampler.groups[jumpind]] += (
                    np.random.randn() * cd * np.sqrt(self.sampler.S[jumpind][ind]) * self.sampler.U[jumpind][:, ind].flatten()
                )
        else:
            # y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
            # q[self.covinds] = np.dot(self.U, y)
            q[self.sampler.groups[jumpind]] += (
                np.random.randn() * cd * np.sqrt(self.sampler.S[jumpind][ind]) * self.sampler.U[jumpind][:, ind].flatten()
            )

        return q, qxy

    # AM jump
    def covarianceJumpProposalAM(self, x, iter, beta):
        """
        Adaptive Jump Proposal. This function will occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain
        @param sampler: PTMCMCSampler sampler object

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = np.random.randint(0, len(self.sampler.groups))
        mass_check = any(x in self.mass_idxs for x in self.sampler.groups[jumpind])

        # adjust step size
        prob = np.random.rand()

        # large jump
        if prob > 0.97:
            scale = 10

        # small jump
        elif prob > 0.9:
            scale = 0.2

        # small-medium jump
        # elif prob > 0.6:
        #    scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if self.sampler.temp <= 100:
            scale *= np.sqrt(self.sampler.temp)

        # get parmeters in new diagonalized basis
        y = np.dot(self.sampler.U[jumpind].T, x[self.sampler.groups[jumpind]])

        # make correlated componentwise adaptive jump
        ind = np.arange(len(self.sampler.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.sampler.S[jumpind][ind])

        if self.restrict_mass:
            # If the initial sample is bad, the sampler cannot change the mass values (hence iter<10)
            # Or if the drawn parameter will change the pulsar mass
            if iter < 10 or mass_check:
                accepted = False
                emergency_iter = 0  # If the initial sample is bad, the sampler cannot change the mass values
                while not accepted and emergency_iter < self.max_emergency_iter:
                    if emergency_iter > 0:
                        q = x.copy()
                        # choose mass group
                        jumpind = [i for (i, x) in enumerate(self.sampler.groups) if set(x).issubset(self.mass_idxs)][0]
                        # get mass parmeters in new diagonalized basis
                        y = np.dot(self.sampler.U[jumpind].T, x[self.sampler.groups[jumpind]])

                        # make correlated componentwise adaptive jump
                        ind = np.arange(len(self.sampler.groups[jumpind]))
                        neff = len(ind)
                        cd = 2.4 / np.sqrt(2 * neff) * scale

                        y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.sampler.S[jumpind][ind])

                    q[self.sampler.groups[jumpind]] = np.dot(self.sampler.U[jumpind], y)

                    accepted = self.check_pulsar_mass(q)
                    emergency_iter += 1

                if emergency_iter > 500:
                    print("covarianceJumpProposalAM")
                    print("Emergency iter:", emergency_iter)
                    _ = self.check_pulsar_mass(q, print_mp=True)
            else:
                q[self.sampler.groups[jumpind]] = np.dot(self.sampler.U[jumpind], y)
        else:
            q[self.sampler.groups[jumpind]] = np.dot(self.sampler.U[jumpind], y)

        return q, qxy

    # Differential evolution jump
    def DEJump(self, x, iter, beta):
        """
        Differential Evolution Jump. This function will  occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain
        @param sampler: PTMCMCSampler sampler object

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        # get old parameters
        q = x.copy()
        qxy = 0

        # after burn in, actually use DE jumps
        if (iter - 1) >= self.sampler.burn and self.sampler.MPIrank == 0:

            # choose group
            jumpind = np.random.randint(0, len(self.sampler.groups))
            ndim = len(self.sampler.groups[jumpind])
            mass_check = any(x in self.mass_idxs for x in self.sampler.groups[jumpind])

            bufsize = len(self.sampler._DEbuffer)

            # draw a random integer from 0 - iter
            mm = np.random.randint(0, bufsize)
            nn = np.random.randint(0, bufsize)

            # make sure mm and nn are not the same iteration
            while mm == nn:
                nn = np.random.randint(0, bufsize)

            # get jump scale size
            prob = np.random.rand()

            # mode jump
            if prob > 0.5:
                scale = 1.0

            else:
                scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / beta)

            if self.restrict_mass:
                # If the initial sample is bad, the sampler cannot change the mass values (hence iter<10)
                # Or if the drawn parameter will change the pulsar mass
                if iter < 10 or mass_check:
                    accepted = False
                    emergency_iter = 0  # If the initial sample is bad, the sampler cannot change the mass values
                    while not accepted and emergency_iter < self.max_emergency_iter:
                        if emergency_iter > 0:
                            q = x.copy()
                            # choose mass group
                            jumpind = [i for (i, x) in enumerate(self.sampler.groups) if set(x).issubset(self.mass_idxs)][0]
                            ndim = len(self.sampler.groups[jumpind])

                            # get jump scale size
                            prob = np.random.rand()
                            # mode jump
                            if prob > 0.5:
                                scale = 1.0
                            else:
                                scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / beta)

                        for ii in range(ndim):
                            # jump size
                            sigma = self.sampler._DEbuffer[mm, self.sampler.groups[jumpind][ii]] - self.sampler._DEbuffer[nn, self.sampler.groups[jumpind][ii]]
                            # jump
                            q[self.sampler.groups[jumpind][ii]] += scale * sigma

                        accepted = self.check_pulsar_mass(q)
                        emergency_iter += 1

                    if emergency_iter > 500:
                        print("DEJump")
                        print("Emergency iter:", emergency_iter)
                        _ = self.check_pulsar_mass(q, print_mp=True)
                else:
                    for ii in range(ndim):

                        # jump size
                        sigma = self.sampler._DEbuffer[mm, self.sampler.groups[jumpind][ii]] - self.sampler._DEbuffer[nn, self.sampler.groups[jumpind][ii]]

                        # jump
                        q[self.sampler.groups[jumpind][ii]] += scale * sigma
            else:
                for ii in range(ndim):

                    # jump size
                    sigma = self.sampler._DEbuffer[mm, self.sampler.groups[jumpind][ii]] - self.sampler._DEbuffer[nn, self.sampler.groups[jumpind][ii]]

                    # jump
                    q[self.sampler.groups[jumpind][ii]] += scale * sigma

        return q, qxy

    def draw_from_timing_model(self, x, iter, beta):
        """
        Pull from standard normal distributions (based on the fit timing
        parameters) as jump proposals. Pull from various timing parameters,
        based on the groups of parameters in tm_groups, which includes
        individual parameter proposals.
        """
        q = x.copy()
        lqxy = 0

        # signal_name = "timing_model"

        # draw parameter from signal model
        idxs = np.random.choice(self.tm_groups)
        try:
            L = len(idxs)
            pidxs = [idx for idx in idxs if idx in self.special_idxs]
            q[idxs] = np.random.randn(L)
            for pidx in pidxs:
                q[pidx] = self.params[pidx].sample()
            mass_check = any(idx in self.mass_idxs for idx in idxs)
        except TypeError:
            L = 1
            pidxs = []
            if idxs in self.special_idxs:
                pidxs = [idxs]
            q[idxs] = np.random.randn(L)
            mass_check = idxs in self.mass_idxs

        if self.restrict_mass:
            # If the initial sample is bad, the sampler cannot change the mass values (hence iter<10)
            # Or if the drawn parameter will change the pulsar mass
            if iter < 10 or mass_check:
                accepted = False
                emergency_iter = 0  # If the initial sample is bad, the sampler cannot change the mass values
                while not accepted and emergency_iter < self.max_emergency_iter:
                    if emergency_iter > 0:
                        # draw different parameter from mass groups model
                        idxs = self.mass_idxs
                        L = len(idxs)
                        pidxs = [idx for idx in idxs if idx in self.special_idxs]
                    q[idxs] = np.random.randn(L)
                    if len(pidxs) == 0:
                        pass
                    else:
                        for pidx in pidxs:
                            q[pidx] = self.params[pidx].sample()
                    accepted = self.check_pulsar_mass(q)
                    emergency_iter += 1

                if emergency_iter > 500:
                    print("draw_from_timing_model")
                    print("Emergency iter:", emergency_iter)
                    _ = self.check_pulsar_mass(q, print_mp=True)

        # forward-backward jump probability
        lqxy = mv_norm.logpdf(x[idxs], mean=np.zeros(L)) - mv_norm.logpdf(
            q[idxs], mean=np.zeros(L)
        )

        return q, float(lqxy)

    def draw_from_timing_model_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0

        signal_name = "timing_model"

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])

        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        if self.restrict_mass:
            # If the initial sample is bad, the sampler cannot change the mass values (hence iter<10)
            # Or if the drawn parameter will change the pulsar mass
            if iter < 10 or str(param).split(":")[0].split('_')[-1] in self.mass_pars:
                accepted = False
                emergency_iter = 0
                while not accepted and emergency_iter < self.max_emergency_iter:
                    if emergency_iter > 0:
                        # draw different parameter from mass groups model
                        param = np.random.choice([x for x in self.snames[signal_name] if str(x).split(":")[0].split('_')[-1] in self.mass_pars])

                        # scalar parameter
                        q[self.pmap[str(param)]] = param.sample()

                    accepted = self.check_pulsar_mass(q)
                    emergency_iter += 1

                if emergency_iter > 500:
                    print("draw_from_timing_model_prior")
                    print("Emergency iter:", emergency_iter)
                    _ = self.check_pulsar_mass(q, print_mp=True)

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(
            q[self.pmap[str(param)]]
        )

        return q, float(lqxy)

    def check_pulsar_mass(self, new_draw, print_mp=False):
        """
        Computes the companion mass from the Keplerian mass function,
        given projected size and orbital period. This function uses a
        Newton-Raphson method since the equation is transcendental.

        :param new_draw: array of newly drawn parameters
        """

        for i, mass_idx in enumerate(self.mass_idxs):
            if mass_idx in self.special_idxs:
                value = new_draw[mass_idx]
            else:
                value = self.unscaled_mass_values[i][0] + self.unscaled_mass_values[i][1] * new_draw[mass_idx]

            # print(self.pnames[mass_idx],"unscaled values:",self.unscaled_mass_values[i][0],"new values:",new_draw[mass_idx],value)

            if i == 0:
                A1 = value
            elif i == 1:
                M2 = value
            elif i == 2:
                PB = value
            elif i == 3:
                if self.inclination_flag == "SINI":
                    SINI = value
                elif self.inclination_flag == "COSI":
                    with warnings.catch_warnings(record=True) as w:
                        SINI = np.sqrt(1-value**2)
                        if len(w) > 0:
                            # print("Error!!! COSI Value above 1 or below 0:",value)
                            return False
                else:
                    raise ValueError("inclination_flag can only be SINI or COSI")

        T_sun = 4.925490947e-6  # conversion from solar masses to seconds
        nb = 2 * np.pi / PB / 86400
        mf = nb**2 * A1**3 / T_sun

        with warnings.catch_warnings(record=True) as w:
            mp = np.sqrt((M2 * SINI) ** 3 / mf) - M2
            if len(w) > 0:
                # print("Error!!! M2 Value is negative:",M2)
                # print("SINI Value",SINI)
                # print("M2 Value",M2)
                # print("mf Value",mf)
                return False

        if print_mp:
            print("Pulsar Mass:", mp)

        # If newly sampled pulsar mass is between 0 and 3 solar masses, accept it
        if mp < 0. or mp > 3.:
            return False
        else:
            return True


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
            ind = [
                params.index(p)
                for p in signal.param_names
                if not gpars.size or p not in gpars
            ]
            if ind:
                groups.append(ind)

    return groups


def get_psr_groups(pta):
    groups = []
    for psr in pta.pulsars:
        grp = [pta.param_names.index(par) for par in pta.param_names if psr in par]
        groups.append(grp)
    return groups


def get_cw_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    ang_pars = ["costheta", "phi", "cosinc", "phase0", "psi"]
    mfdh_pars = ["log10_Mc", "log10_fgw", "log10_dL", "log10_h"]
    freq_pars = ["log10_Mc", "log10_fgw", "pdist", "pphase"]

    groups = []
    for pars in [ang_pars, mfdh_pars, freq_pars]:
        groups.append(group_from_params(pta, pars))

    return groups


def get_timing_groups(pta):
    """Utility function to get parameter groups for timing sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    pos_pars = ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]
    spin_pars = ["F", "F0", "F1", "F2", "P", "P1", "Offset"]
    mass_pars = ["PB", "A1", "SINI", "COSI", "M2"]
    kep_pars = [
        "PB",
        "T0",
        "A1",
        "OM",
        "E",
        "ECC",
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "FB",
        "SINI",
        "COSI",
        "MTOT",
        "M2",
        "A1DOT",
        "XDOT",
        "X2DOT",
        "EDOT",
        "KOM",
        "KIN",
        "TASC",
    ]
    gr_pars = [
        "H3",
        "H4",
        "OMDOT",
        "OM2DOT",
        "XOMDOT",
        "PBDOT",
        "XPBDOT",
        "GAMMA",
        "PPNGAMMA",
        "DR",
        "DTHETA",
    ]
    pm_pars = [
        "PMDEC",
        "PMRA",
        "PMELONG",
        "PMELAT",
        "PMRV",
        "PMBETA",
        "PMLAMBDA",
    ]

    groups = []
    for pars in [pos_pars, spin_pars, kep_pars, gr_pars, pm_pars, mass_pars]:
        group = []
        for p in pars:
            for q in pta.param_names:
                if p == q.split("_")[-1]:
                    group.append(pta.param_names.index(q))
        if len(group):
            groups.append(group)

    dmx_group = group_from_partial_par_name(pta, part="DMX")
    if len(dmx_group):
        groups.append(dmx_group)
    jump_fd_group = group_from_partial_par_name(pta, part="FD")
    jump_fd_group.extend(group_from_partial_par_name(pta, part="JUMP"))
    jump_fd_group.extend(group_from_partial_par_name(pta, part="dm_model"))
    if len(jump_fd_group):
        groups.append(jump_fd_group)

    return groups


def group_from_partial_par_name(pta, part="DMX"):
    return [pta.param_names.index(q) for q in pta.param_names if part in q]


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
                  save_ext_dists=False, timing=False, psr=None,
                  restrict_mass=True,
                  loglkwargs={}, logpkwargs={}):
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

    if timing:
        groups.extend(get_timing_groups(pta))
        groups.append(group_from_params(pta,
                      [x for x in pta.param_names if any(y in x for y in ["timing_model", "ecorr"])]))

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                     outDir=outdir, resume=resume, loglkwargs=loglkwargs,
                     logpkwargs=logpkwargs)

    save_runtime_info(pta, sampler.outDir, human)

    # additional jump proposals
    jp = JumpProposal(pta, empirical_distr=empirical_distr, save_ext_dists=save_ext_dists, outdir=outdir, timing=timing, psr=psr, sampler=sampler, restrict_mass=restrict_mass)

    sampler.jp = jp

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 15)

    # try adding empirical proposals
    if empirical_distr is not None:
        print('Attempting to add empirical proposals...\n')
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 30)

    # Red noise prior draw
    if "red noise" in jp.snames:
        print("Adding red noise prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

    # DM GP noise prior draw
    if "dm_gp" in jp.snames:
        print("Adding DM GP noise prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

    # DM annual prior draw
    if "dm_s1yr" in jp.snames:
        print("Adding DM annual prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

    # DM dip prior draw
    if "dmexp" in jp.snames:
        print("Adding DM exponential dip prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)

    # DM cusp prior draw
    if "dm_cusp" in jp.snames:
        print("Adding DM exponential cusp prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)

    # DMX prior draw
    if "dmx_signal" in jp.snames:
        print("Adding DMX prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

    # Ephemeris prior draw
    if "d_jupiter_mass" in pta.param_names:
        print("Adding ephemeris model prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if np.any([('gw' in par and 'log10_A' in par) for par in pta.param_names]):
        print('Adding GWB uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if "dipole_log10_A" in pta.param_names:
        print("Adding dipole uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if "monopole_log10_A" in pta.param_names:
        print("Adding monopole uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if "log10Apol_tt" in pta.param_names:
        print("Adding alternative GW-polarization uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_altpol_log_uniform_distribution, 10)

    # BWM prior draw
    if "bwm_log10_A" in pta.param_names:
        print("Adding BWM prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

    # FDM prior draw
    if "fdm_log10_A" in pta.param_names:
        print("Adding FDM prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_fdm_prior, 10)

    # CW prior draw
    if "cw_log10_h" in pta.param_names:
        print("Adding CW strain prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)
    if "cw_log10_Mc" in pta.param_names:
        print("Adding CW prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_cw_distribution, 10)

    # Non Linear Timing Draws
    if "timing_model" in jp.snames:
        print("Adding timing model jump proposal...\n")
        sampler.addProposalToCycle(jp.draw_from_timing_model, 25)
    if "timing_model" in jp.snames:
        print("Adding timing model prior draw...\n")
        sampler.addProposalToCycle(jp.draw_from_timing_model_prior, 25)

    # DM Model Draws
    if "dm_model" in jp.snames:
        print("Adding dm model prior draw...\n")
        sampler.addProposalToCycle(jp.draw_from_signal("dm_model"), 10)

    if timing:
        if jp.restrict_mass:
            # SCAM and AM Draws
            # add SCAM
            print("Adding SCAM Jump Proposal...\n")
            sampler.addProposalToCycle(jp.covarianceJumpProposalSCAM, 20)

            # add AM
            print("Adding AM Jump Proposal...\n")
            sampler.addProposalToCycle(jp.covarianceJumpProposalAM, 20)

            # DE does not work well with restricting the pulsar mass
    # free spectrum prior draw
    if np.any(['log10_rho' in par for par in pta.param_names]):
        print('Adding free spectrum prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gw_rho_prior, 25)

    return sampler
