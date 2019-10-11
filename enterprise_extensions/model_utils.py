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

    :param samples: MCMCF samples of GWB (or common red noise) amplitude
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
