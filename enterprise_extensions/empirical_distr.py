# -*- coding: utf-8 -*-

import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class EmpiricalDistribution1D(object):
    """
    Class used to define a 1D empirical distribution
    based on posterior from another MCMC.
    """
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
            msg = 'WARNING: only 1D and 2D empirical distributions are currently allowed.'
            logger.warning(msg)

    # save the list of empirical distributions as a pickle file
    if len(distr) > 0:
        with open(filename, 'wb') as f:
            pickle.dump(distr, f)

        msg = 'The empirical distributions have been pickled to {0}.'.format(filename)
        logger.info(msg)
    else:
        msg = 'WARNING: No empirical distributions were made!'
        logger.warning(msg)
