# -*- coding: utf-8 -*-

import logging
import pickle

import numpy as np

try:
    from sklearn.neighbors import KernelDensity
    sklearn_available=True
except ModuleNotFoundError:
    sklearn_available=False
from scipy.interpolate import interp1d, interp2d

logger = logging.getLogger(__name__)


class EmpiricalDistribution1D(object):
    """
    Class used to define a 1D empirical distribution
    based on posterior from another MCMC.

    :param samples: samples for hist
    :param bins: edges to use for hist (left and right) make sure bins
        cover whole prior!

    """
    def __init__(self, param_name, samples, bins):
        self.ndim = 1
        self.param_name = param_name
        self._Nbins = len(bins)-1
        hist, x_bins = np.histogram(samples, bins=bins)

        self._edges = x_bins
        self._wids = np.diff(x_bins)

        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / float(counts) / self._wids
        self._cdf = np.cumsum((self._pdf*self._wids).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw, side='right')

        idx = np.unravel_index(draw_bin, self._Nbins)[0]
        samp = self._edges[idx] + self._wids[idx]*np.random.rand()
        return np.array(samp)

    def prob(self, params):
        ix = np.searchsorted(self._edges, params) - 1

        return self._pdf[ix]

    def logprob(self, params):
        ix = np.searchsorted(self._edges, params) - 1

        return self._logpdf[ix]


class EmpiricalDistribution1DKDE(object):
    def __init__(self, param_name, samples, minval=None, maxval=None, bandwidth=0.1, nbins=40):
        """
        Minvals and maxvals should specify priors for these. Should make these required.
        """
        self.ndim = 1
        self.param_name = param_name
        self.bandwidth = bandwidth
        # code below  relies on samples axes being swapped. but we
        # want to keep inputs the same
        # create a 2D KDE from which to evaluate
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples.reshape((samples.size, 1)))
        if minval is None:
            # msg = "minvals for KDE empirical distribution were not supplied. Resulting distribution may not have support over full prior"
            # logger.warning(msg)
            # widen these to add support
            minval = min(samples)
            maxval = max(samples)
        # significantly faster probability estimation using interpolation
        # instead of evaluating KDE every time
        self.minval = minval
        self.maxval = maxval
        xvals = np.linspace(minval, maxval, num=nbins)
        self._Nbins = nbins
        scores = np.array([self.kde.score(np.atleast_2d(xval)) for xval in xvals])
        # interpolate within prior
        self._logpdf = interp1d(xvals, scores, kind='linear', fill_value=-1000)

    def draw(self):
        params = self.kde.sample(1).T
        return params.squeeze()


# class used to define a 2D empirical distribution
# based on posteriors from another MCMC
class EmpiricalDistribution2D(object):
    """
    Class used to define a 1D empirical distribution
    based on posterior from another MCMC.

    :param samples: samples for hist
    :param bins: edges to use for hist (left and right)
        make sure bins cover whole prior!

    """
    def __init__(self, param_names, samples, bins):
        self.ndim = 2
        self.param_names = param_names
        self._Nbins = [len(b)-1 for b in bins]
        hist, x_bins, y_bins = np.histogram2d(*samples, bins=bins)

        self._edges = np.array([x_bins, y_bins])
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
        ix, iy = [np.searchsorted(self._edges[ii], params[ii]) - 1 for ii in range(2)]

        return self._pdf[ix, iy]

    def logprob(self, params):
        ix, iy = [np.searchsorted(self._edges[ii], params[ii]) - 1 for ii in range(2)]

        return self._logpdf[ix, iy]


class EmpiricalDistribution2DKDE(object):
    def __init__(self, param_names, samples, minvals=None, maxvals=None, bandwidth=0.1, nbins=40):
        """
        Minvals and maxvals should specify priors for these. Should make these required.

        :param param_names: 2-element list of parameter names
        :param samples: samples, with dimension (2 x Nsamples)


        :return distr: list of empirical distributions
        """
        self.ndim = 2
        self.param_names = param_names
        self.bandwidth = bandwidth
        # code below  relies on samples axes being swapped. but we
        # want to keep inputs the same
        # create a 2D KDE from which to evaluate

        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples.T)
        if minvals is None:
            msg = "minvals for KDE empirical distribution were not supplied. Resulting distribution may not have support over full prior"
            logger.warning(msg)
            # widen these to add support
            minvals = (min(samples[0, :]), min(samples[1, :]))
            maxvals = (max(samples[0, :]), max(samples[1, :]))
        # significantly faster probability estimation using interpolation
        # instead of evaluating KDE every time
        self.minvals = minvals
        self.maxvals = maxvals
        xvals = np.linspace(minvals[0], maxvals[0], num=nbins)
        yvals = np.linspace(minvals[1], maxvals[1], num=nbins)
        self._Nbins = [yvals.size for ii in range(xvals.size)]
        scores = np.array([self.kde.score(np.array([xvals[ii], yvals[jj]]).reshape((1, 2))) for ii in range(xvals.size) for jj in range(yvals.size)])
        # interpolate within prior
        self._logpdf = interp2d(xvals, yvals, scores, kind='linear', fill_value=-1000)

    def draw(self):
        params = self.kde.sample(1).T
        return params.squeeze()

    def prob(self, params):
        # just in case...make sure to make this zero outside of our prior ranges
        param1_out = params[0] < self.minvals[0] or params[0] > self.maxvals[0]
        param2_out = params[1] < self.minvals[1] or params[1] > self.maxvals[1]
        if param1_out or param2_out:
            # essentially zero
            return -1000
        else:
            return np.exp(self._logpdf(*params))[0]

    def logprob(self, params):
        return self._logpdf(*params)[0]


def make_empirical_distributions(pta, paramlist, params, chain,
                                 burn=0, nbins=81, filename='distr.pkl',
                                 return_distribution=True,
                                 save_dists=True):
    """
        Utility function to construct empirical distributions.

        :param pta: the pta object used to generate the posteriors
        :param paramlist: a list of parameter names,
                          either single parameters or pairs of parameters
        :param chain: MCMC chain from a previous run
        :param burn: desired number of initial samples to discard
        :param nbins: number of bins to use for the empirical distributions

        :return distr: list of empirical distributions

        """

    distr = []

    if not save_dists and not return_distribution:
        msg = "no distribution returned or saved, are you sure??"
        logger.info(msg)

    for pl in paramlist:

        if type(pl) is not list:

            pl = [pl]

        if len(pl) == 1:
            idx = pta.param_names.index(pl[0])
            prior_min = pta.params[idx].prior._defaults['pmin']
            prior_max = pta.params[idx].prior._defaults['pmax']

            # get the bins for the histogram
            bins = np.linspace(prior_min, prior_max, nbins)

            new_distr = EmpiricalDistribution1D(pl[0], chain[burn:, idx], bins)

            distr.append(new_distr)

        elif len(pl) == 2:

            # get the parameter indices
            idx = [pta.param_names.index(pl1) for pl1 in pl]

            # get the bins for the histogram
            bins = [np.linspace(pta.params[i].prior._defaults['pmin'],
                                pta.params[i].prior._defaults['pmax'], nbins) for i in idx]

            new_distr = EmpiricalDistribution2D(pl, chain[burn:, idx].T, bins)

            distr.append(new_distr)

        else:
            msg = 'WARNING: only 1D and 2D empirical distributions are currently allowed.'
            logger.warning(msg)

    # save the list of empirical distributions as a pickle file
    if save_dists:
        if len(distr) > 0:
            with open(filename, 'wb') as f:
                pickle.dump(distr, f)

            msg = 'The empirical distributions have been pickled to {0}.'.format(filename)
            logger.info(msg)
        else:
            msg = 'WARNING: No empirical distributions were made!'
            logger.warning(msg)

    if return_distribution:
        return distr


def make_empirical_distributions_KDE(pta, paramlist, params, chain,
                                     burn=0, nbins=41, filename='distr.pkl',
                                     bandwidth=0.1,
                                     return_distribution=True,
                                     save_dists=True):
    """
        Utility function to construct empirical distributions.

        :param paramlist: a list of parameter names,
                          either single parameters or pairs of parameters
        :param params: list of all parameter names for the MCMC chain
        :param chain: MCMC chain from a previous run, has dimensions Nsamples x Nparams
        :param burn: desired number of initial samples to discard
        :param nbins: number of bins to use for the empirical distributions

        :return distr: list of empirical distributions

        """

    distr = []
    if not save_dists and not return_distribution:
        msg = "no distribution returned or saved, are you sure??"
        logger.info(msg)

    for pl in paramlist:

        if type(pl) is not list:

            pl = [pl]

        if len(pl) == 1:

            # get the parameter index
            idx = pta.param_names.index(pl[0])
            prior_min = pta.params[idx].prior._defaults['pmin']
            prior_max = pta.params[idx].prior._defaults['pmax']

            # get the bins for the histogram

            new_distr = EmpiricalDistribution1DKDE(pl[0], chain[burn:, idx], bandwidth=bandwidth, minval=prior_min, maxval=prior_max)

            distr.append(new_distr)

        elif len(pl) == 2:

            # get the parameter indices
            idx = [pta.param_names.index(pl1) for pl1 in pl]

            # get the bins for the histogram
            bins = [np.linspace(pta.params[i].prior._defaults['pmin'],
                                pta.params[i].prior._defaults['pmax'], nbins) for i in idx]
            minvals = [pta.params[0].prior._defaults['pmin'], pta.params[1].prior._defaults['pmin']]
            maxvals = [pta.params[0].prior._defaults['pmax'], pta.params[1].prior._defaults['pmax']]

            # get the bins for the histogram
            if sklearn_available:
                new_distr = EmpiricalDistribution2DKDE(pl, chain[burn:, idx].T, bandwidth=bandwidth, minvals=minvals, maxvals=maxvals)
            else:
                logger.warn('`sklearn` package not available. Fall back to using histgrams for empirical distribution')
                new_distr = EmpiricalDistribution2D(pl, chain[burn:, idx].T, bins)

            distr.append(new_distr)

        else:
            msg = 'WARNING: only 1D and 2D empirical distributions are currently allowed.'
            logger.warning(msg)

    # save the list of empirical distributions as a pickle file
    if save_dists:
        if len(distr) > 0:
            with open(filename, 'wb') as f:
                pickle.dump(distr, f)

            msg = 'The empirical distributions have been pickled to {0}.'.format(filename)
            logger.info(msg)
        else:
            msg = 'WARNING: No empirical distributions were made!'
            logger.warning(msg)
    if return_distribution:
        return distr
