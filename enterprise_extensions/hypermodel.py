# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.linalg as sl

from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from .sampler import JumpProposal, get_parameter_groups, save_runtime_info


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

        groups = []
        for p in self.models.values():
            groups.extend(get_parameter_groups(p))
        list(np.unique(groups))

        groups.extend([[len(self.param_names)-1]])  # nmodel

        return groups

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
                      empirical_distr=None, groups=None, human=None):
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
        if os.path.exists(outdir+'/cov.npy'):
            cov = np.load(outdir+'/cov.npy')
        else:
            cov = np.diag(np.ones(ndim) * 1.0**2)  # used to be 0.1

        # parameter groupings
        if groups is None:
            groups = self.get_parameter_groups()

        sampler = ptmcmc(ndim, self.get_lnlikelihood, self.get_lnprior, cov,
                         groups=groups, outDir=outdir, resume=resume)
        save_runtime_info(self, sampler.outDir, human)

        # additional jump proposals
        jp = JumpProposal(self, self.snames, empirical_distr=empirical_distr)
        sampler.jp = jp

        # always add draw from prior
        sampler.addProposalToCycle(jp.draw_from_prior, 5)

        # try adding empirical proposals
        if empirical_distr is not None:
            print('Adding empirical proposals...\n')
            sampler.addProposalToCycle(jp.draw_from_empirical_distr, 25)

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

        # DMX prior draw
        if 'dmx_signal' in jp.snames:
            print('Adding DMX prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

        # SW prior draw
        if 'gp_sw' in jp.snames:
            print('Adding Solar Wind DM GP prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_dm_sw_prior, 10)

        # Chromatic GP noise prior draw
        if 'chrom_gp' in self.snames:
            print('Adding Chromatic GP noise prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_chrom_gp_prior, 10)

        # Ephemeris prior draw
        if 'd_jupiter_mass' in self.param_names:
            print('Adding ephemeris model prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

        # GWB uniform distribution draw
        if np.any([('gw' in par and 'log10_A' in par) for par in self.param_names]):
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

        # FDM prior draw
        if 'fdm_log10_A' in self.param_names:
            print('Adding FDM prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_fdm_prior, 10)

        # CW prior draw
        if 'cw_log10_h' in self.param_names:
            print('Adding CW prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)

        # Prior distribution draw for parameters named GW
        if any([str(p).split(':')[0] for p in list(self.params) if 'gw' in str(p)]):
            print('Adding gw param prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_par_prior(
                par_names=[str(p).split(':')[0] for
                           p in list(self.params)
                           if 'gw' in str(p)]), 10)

        # Model index distribution draw
        if sample_nmodel:
            if 'nmodel' in self.param_names:
                print('Adding nmodel uniform distribution draws...\n')
                sampler.addProposalToCycle(self.draw_from_nmodel_prior, 25)

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
