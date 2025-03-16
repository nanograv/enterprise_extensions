# sampling utilities
#  based on https://github.com/ipta/IPTA_DR2_analysis/blob/master/exe/sample_utils.py

import numpy as np


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
        #TODO check all idxs in keys!
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

        try: # vector parameter
            y[ii] = self.samplers[ii]()[0]
        except (IndexError, TypeError) as e:
            y[ii] = self.samplers[ii]()

        if self.log_qs is None:
            lqxy = 0
        else:
            lqxy = self.log_qs[ii](x[ii]) - self.log_qs[ii](y[ii])

        return y, lqxy


def build_prior_draw(pta, parlist, name=None):
    """create a callable object to perfom a prior draw
    :param pta:
        instantiated PTA object
    :param parlist:
        single string or list of strings of parameter name(s) to
        use for this jump.
    :param name:
        display name for PTMCMCSampler bookkeeping
    """
    if not isinstance(parlist, list):
        parlist = [parlist]
    idxs = [pta.param_names.index(par) for par in parlist]

    # parameter map
    pmap = []
    ct = 0
    for ii, pp in enumerate(pta.params):
        size = pp.size or 1
        for nn in range(size):
            pmap.append(ii)
        ct += size

    sampler = {ii: pta.params[pmap[ii]].sample for ii in idxs}
    log_q = {ii: pta.params[pmap[ii]].get_logpdf for ii in idxs}

    return UserDraw(idxs, sampler, log_q, name=name)


def grubin(data, M=2, burn=0):
    """
    Gelman-Rubin split R hat statistic to verify convergence.

    See section 3.1 of https://arxiv.org/pdf/1903.08008.pdf.
    Values > 1.01 => recommend continuing sampling due to poor convergence.

    Input:
        data (ndarray): consists of entire chain file
        pars (list): list of parameters for each column
        M (integer): number of times to split the chain
        burn (int or float): number of samples or fraction of chain to cut for burn-in

    Output:
        Rhat (ndarray): array of values for each index
    """
    if isinstance(burn, float):
        burn = int(burn * data.shape[0])  # cut off burn-in
    try:
        data_split = np.split(data[burn:], M)
    except:
        # this section is to make everything divide evenly into M arrays
        P = int(np.floor((len(data[:, 0]) - burn) / M))  # nearest integer to division
        X = len(data[:, 0]) - burn - M * P # number of additional burn in points
        burn += X  # burn in to the nearest divisor
        burn = int(burn)

        data_split = np.split(data[burn:], M)

    N = len(data[burn:, 0])
    data = np.array(data_split)

    theta_bar_dotm = np.mean(data, axis=1)  # mean of each subchain
    theta_bar_dotdot = np.mean(theta_bar_dotm, axis=0)  # mean of between chains
    B = N / (M - 1) * np.sum((theta_bar_dotm - theta_bar_dotdot)**2, axis=0)  # between chains

    # do some clever broadcasting:
    sm_sq = 1 / (N - 1) * np.sum((data - theta_bar_dotm[:, None, :])**2, axis=1)
    W = 1 / M * np.sum(sm_sq, axis=0)  # within chains
    
    var_post = (N - 1) / N * W + 1 / N * B
    Rhat = np.sqrt(var_post / W)

    return Rhat


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
