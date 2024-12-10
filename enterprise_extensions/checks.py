import time
import numpy as np

from enterprise_extensions import models


class CompareTimingModels():
    """
    Compare difference between the usual and marginalized timing models.

    After instantiating, the __call__() method can be used for sampling for any number of points.
    To see the results, use the results() method.

    :param psrs: Pulsar object containing pulsars from model
    :param model_name: String name of model to test. Model must be defined in enterprise_extensions.models.
    :param abs_tol: absolute tolerance for error between timing models (default 1e-3), set to None to bypass errors
    :param rel_tol: relative tolerance for error between timing models (default 1e-6), set to None to bypass errors
    :param dense: use the dense cholesky algorithm over sparse
    """

    def __init__(self, psrs, model_name='model_1', abs_tol=1e-3, rel_tol=1e-6, dense=True, **kwargs):
        model = getattr(models, model_name)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        if dense:
            self.pta_marg = model(psrs, tm_marg=True, dense_like=True, **kwargs)  # marginalized model
        else:
            self.pta_marg = model(psrs, tm_marg=True, **kwargs)  # marginalized model
        self.pta_norm = model(psrs, **kwargs)  # normal model
        self.tm_correction = 0
        for psr in psrs:
            self.tm_correction -= 0.5 * np.log(1e40) * psr.Mmat.shape[1]
        self.abs_err = []
        self.rel_err = []
        self.count = 0

    def check_timing(self, number=10_000):
        print('Timing sample creation...')
        start = time.time()
        for __ in range(number):
            x0 = np.hstack([p.sample() for p in self.pta_marg.params])
        end = time.time()
        sample_time = end - start
        print('Sampling {0} points took {1} seconds.'.format(number, sample_time))

        print('Timing MarginalizedTimingModel...')
        start = time.time()
        for __ in range(number):
            x0 = np.hstack([p.sample() for p in self.pta_marg.params])
            self.pta_marg.get_lnlikelihood(x0)
        end = time.time()
        time_marg = end - start - sample_time  # remove sampling time from total time taken
        print('Sampling {0} points took {1} seconds.'.format(number, time_marg))

        print('Timing TimingModel...')
        start = time.time()
        for __ in range(number):
            x0 = np.hstack([p.sample() for p in self.pta_marg.params])
            self.pta_norm.get_lnlikelihood(x0)
        end = time.time()
        time_norm = end - start - sample_time  # remove sampling time from total time taken
        print('Sampling {0} points took {1} seconds.'.format(number, time_norm))

        res = time_norm / time_marg
        print('MarginalizedTimingModel is {0} times faster than TimingModel after {1} points.'.format(res, number))
        return res

    def get_sample_point(self):
        x0 = np.hstack([p.sample() for p in self.pta_marg.params])
        return x0

    def __call__(self, x0):
        res_norm = self.pta_norm.get_lnlikelihood(x0)
        res_marg = self.pta_marg.get_lnlikelihood(x0)
        abs_err = np.abs(res_marg - res_norm)
        rel_err = abs_err / res_norm
        self.abs_err.append(abs_err)
        self.rel_err.append(rel_err)
        self.count += 1
        if self.abs_tol is not None and abs_err > self.abs_tol:
            abs_raise = 'Absolute error is {0} at {1} which is larger than abs_tol of {2}.'.format(
                abs_err, x0, self.abs_tol)
            raise ValueError(abs_raise)
        elif self.rel_tol is not None and rel_err > self.rel_tol:
            rel_raise = 'Relative error is {0} at {1} which is larger than rel_tol of {2}.'.format(
                rel_err, x0, self.rel_tol)
            raise ValueError(rel_raise)
        return res_norm

    def results(self):
        print('Number of points evaluated:', self.count)
        print('Maximum absolute error:', np.max(self.abs_err))
        print('Maximum relative error:', np.max(self.rel_err))
        return self.abs_err, self.rel_err
