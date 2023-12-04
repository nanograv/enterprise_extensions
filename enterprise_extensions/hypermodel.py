# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.linalg as sl

from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from .sampler import (JumpProposal, get_parameter_groups,
                      save_runtime_info, build_prior_draw, EmpDistrDraw)


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

        self.pulsars = np.unique(np.concatenate([p.pulsars
                                                 for p in self.models.values()]))
        self.pulsars = np.sort(self.pulsars)

        #########
        self.params = [p for p in self.models[0].params]  # start of param list
        uniq_params = [str(p) for p in self.models[0].params]  # which params are unique
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
        for key in self.snames:
            self.snames[key] = []

        for mm in self.models:
            for sc in self.models[mm]._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
        for key in self.snames:
            self.snames[key] = list(set(self.snames[key]))

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

        unique_groups = []
        for p in self.models.values():
            groups = get_parameter_groups(p)
            # check for any duplicate groups
            # e.g. the GWB may have different indices in model 1 and model 2
            for group in groups:
                check_group = []
                for idx in group:
                    param_name = p.param_names[idx]
                    check_group.append(self.param_names.index(param_name))
                if check_group not in unique_groups:
                    unique_groups.append(check_group)
        unique_groups.extend([[len(self.param_names) - 1]])
        return unique_groups

    def initial_sample(self):
        """
        Draw an initial sample from within the hyper-model prior space.
        """

        x0 = [np.array(p.sample()).ravel().tolist() for p in self.models[0].params]
        uniq_params = [str(p) for p in self.models[0].params]

        for model in self.models.values():
            param_diffs = np.setdiff1d([str(p) for p in model.params], uniq_params)
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
        q[idx] = np.random.uniform(-0.5, self.num_models-0.5)

        lqxy = 0

        return q, float(lqxy)

    def setup_sampler(self, outdir='chains', resume=False, sample_nmodel=True,
                      empirical_distr=None, groups=None, human=None,
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
        """

        # dimension of parameter space
        ndim = len(self.param_names)

        # initial jump covariance matrix
        if os.path.exists(outdir+'/cov.npy') and resume:
            cov = np.load(outdir+'/cov.npy')

            # check that the one we load is the same shape as our data
            cov_new = np.diag(np.ones(ndim) * 1.0**2)
            if cov.shape != cov_new.shape:
                msg = 'The covariance matrix (cov.npy) in the output folder is '
                msg += 'the wrong shape for the parameters given. '
                msg += 'Start with a different output directory or '
                msg += 'change resume to False to overwrite the run that exists.'

                raise ValueError(msg)
        else:
            cov = np.diag(np.ones(ndim) * 1.0**2)  # used to be 0.1

        # parameter groupings
        if groups is None:
            groups = self.get_parameter_groups()

        sampler = ptmcmc(ndim, self.get_lnlikelihood, self.get_lnprior, cov,
                         groups=groups, outDir=outdir, resume=resume,
                         loglkwargs=loglkwargs, logpkwargs=logpkwargs)

        save_runtime_info(self, sampler.outDir, human)

        # additional jump proposals
        jp = JumpProposal(self, self.snames, empirical_distr=empirical_distr)
        sampler.jp = jp

        # always add draw from prior
        sampler.addProposalToCycle(build_prior_draw(self.params,
                                                    self.param_names[:-1],  # ignore nmodel
                                                    name='draw_from_prior'), 5)

        # try adding empirical proposals
        if empirical_distr is not None:
            print('Attempting to add empirical proposals...\n')
            sampler.addProposalToCycle(EmpDistrDraw(jp.empirical_distr,
                                                    self.param_names[:-1],  # ignore nmodel
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
                param_names = [p.name for p in jp.snames[sname]]
                sampler.addProposalToCycle(build_prior_draw(self.params,
                                                            param_names,
                                                            name='draw_from_'+sname), 10)

        # adding other signal draws
        param_names = ['dipole', 'monopole', 'hd', 'log10_rho',
                       'dmexp', 'dm_cusp', 'dm_s1yr']

        for p in param_names:
            par_names = [par for par in self.param_names if p in par]
            if len(par_names) >= 1:
                print(f'Adding {p} prior draws...\n')
                sampler.addProposalToCycle(build_prior_draw(self.params, par_names,
                                                            name='draw_from_'+p), 10)

        return sampler

    def get_process_timeseries(self, psr, chain, burn, comp='DM',
                               mle=False, model=0):
        """
        Construct a time series realization of various constrained processes.

        :param psr: enterprise pulsar object
        :param chain: MCMC chain from sampling all models
        :param burn: desired number of initial samples to discard
        :param comp: which process to reconstruct? (red noise or DM) [default=DM]
        :param mle: create time series from ML of GP hyper-parameters? [default=False]
        :param model: which sub-model within the super-model to reconstruct from? [default=0]

        :return ret: time-series of the reconstructed process
        """

        wave = 0
        pta = self.models[model]
        model_chain = chain[np.rint(chain[:, -5])==model, :]

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
        # Nvec = pta.get_ndiag(params)[0] # Not currently used in code
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
            wave += np.dot(T[:, idx], b[idx])
            ret = wave * (psr.freqs**2 * const.DM_K * 1e12)
        elif comp == 'scattering':
            idx = pardict['scattering_gp']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave * (psr.freqs**4)  # * const.DM_K * 1e12)
        elif comp == 'red':
            idx = pardict['red noise']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave
        elif comp == 'FD':
            idx = pardict['FD']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave
        elif comp == 'all':
            wave += np.dot(T, b)
            ret = wave
        else:
            ret = wave

        return ret

    def summary(self, to_stdout=False):
        """generate summary string for HyperModel, including all PTAs

        :param to_stdout: [bool]
            print summary to `stdout` instead of returning it
        :return: [string]

        """

        summary = ""
        for ii, pta in self.models.items():
            summary += "model " + str(ii) + "\n"
            summary += "=" * 9 + "\n\n"
            summary += pta.summary()
            summary += "=" * 90 + "\n\n"
        if to_stdout:
            print(summary)
        else:
            return summary
