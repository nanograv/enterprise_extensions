import six
import collections

from enterprise.signals.parameter import ConstantParameter
import logging
import itertools

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

import numpy as np
import scipy.linalg as sl
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import jit

from enterprise_extensions.jax.unsubclass import add_matrices, inv_matrix

from enterprise import __version__
from sys import version
_py_version = version.split(" ")[0]

logger = logging.getLogger(__name__)

class MetaSignal(type):
    """Metaclass for Signals. Allows addition of ``Signal`` classes."""

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection([self, other])
        elif isinstance(other, MetaCollection):
            return SignalCollection([self] + other._metasignals)
        else:
            raise TypeError


class MetaCollection(type):
    """Metaclass for Signal collections. Allows addition of
    ``SignalCollection`` classes.
    """

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection(self._metasignals + [other])
        elif isinstance(other, MetaCollection):
            return SignalCollection(self._metasignals + other._metasignals)
        else:
            raise TypeError


@six.add_metaclass(MetaSignal)
class Signal(object):
    """Base class for Signal objects."""

    def __init__(self, psr):
        self.psrname = psr.name

    @property
    def params(self):
        # return only nonconstant parameters
        return [par for par in self._params.values() if not isinstance(par, ConstantParameter)]

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret

    def __repr__(self):
        return "<Enterprise Signal object " + self.signal_id + "[" + ", ".join(p.name for p in self.params) + "]>"

    def get(self, parname, params={}):
        try:
            return params[self._params[parname].name]
        except KeyError:
            return self._params[parname].value

    def set_default_params(self, params):
        """Set default parameters."""
        for kw, par in self._params.items():
            if par.name in params and isinstance(par, ConstantParameter):
                msg = "Setting {} to {}".format(par.name, params[par.name])
                logger.info(msg)
                self._params[kw].value = params[par.name]
            elif par.name not in params and isinstance(par, ConstantParameter):
                if par.value is None:
                    msg = "{} not set! Check parameter dict.".format(par.name)
                    logger.warning(msg)

    def get_ndiag(self, params):
        """Returns the diagonal of the white noise vector `N`.

        This method also supports block diagonal sparse matrices.
        """
        return None

    def get_delay(self, params):
        """Returns the waveform of a deterministic signal."""
        return 0

    def get_basis(self, params=None):
        """Returns the basis array of shape N_toa x N_basis."""
        return None

    def get_phi(self, params):
        """Returns a diagonal covariance matrix of the basis amplitudes."""
        return None

    def get_phiinv(self, params):
        """Returns inverse of the covaraince of basis amplitudes."""
        return None

    def get_logsignalprior(self, params):
        """Returns an additional prior/likelihood terms associated with a signal."""
        return 0


class CommonSignal(Signal):
    """Base class for CommonSignal objects."""

    def get_phiinv(self, params):
        msg = "You probably shouldn't be calling get_phiinv() "
        msg += "on a common red-noise signal."
        raise RuntimeError(msg)

    @classmethod
    def get_phicross(cls, signal1, signal2, params):
        return None


class JAXLogLikelihood(object):
    def __init__(self, pta):
        self.pta = pta

    def _block_TNT(self, TNTs):
        return sl.block_diag(*TNTs)
    
    def _block_TNr(self, TNrs):
        return np.concatenate(TNrs)

    def __call__(self, xs, phiinv_method="cliques"):
        # map parameter vector if needed
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        loglike = 0
         # phiinvs will be a list or may be a big matrix if spatially
        # correlated signals
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=True, method=phiinv_method)

        # get -0.5 * (rNr + logdet_N) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))

        # red noise piece
        if self.pta._commonsignals:
            phiinv, logdet_phi = phiinvs

            TNT = self._block_TNT(TNTs)
            TNr = self._block_TNr(TNrs)

            try:
                cf = jsl.cho_factor(TNT + phiinv)  # cf(Sigma)
                expval = jsl.cho_solve(cf, TNr)
                logdet_sigma = 2 * jnp.sum(jnp.log(jnp.diag(cf[0])))
            except sl.LinAlgError:  # pragma: no cover
                return -jnp.inf

            loglike += 0.5 * (jnp.dot(TNr, expval) - logdet_sigma - logdet_phi)
        else:
            for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
                if TNr is None:
                    continue

                phiinv, logdet_phi = pl
                Sigma = TNT + (jnp.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                try:
                    cf = jsl.cho_factor(Sigma)
                    expval = jsl.cho_solve(cf, TNr)
                except jsl.LinAlgError:  # pragma: no cover
                    return -jnp.inf

                logdet_sigma = jnp.sum(2 * jnp.log(jnp.diag(cf[0])))

                loglike += 0.5 * (jnp.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike



class JAXPTA(object):
    def __init__(self, init, lnlikelihood=JAXLogLikelihood):
        if isinstance(init, Sequence):
            self._signalcollections = list(init)
        else:
            self._signalcollections = [init]

        self.lnlikelihood = lnlikelihood

        # set signal dictionary
        self._set_signal_dict()

    def __add__(self, other):
        if hasattr(other, "_signalcollections"):
            return JAXPTA(self._signalcollections + other._signalcollections, lnlikelihood=self.lnlikelihood)
        else:
            return JAXPTA(self._signalcollections + [other], lnlikelihood=self.lnlikelihood)

    @property
    def params(self):
        ret = set()

        for signalcollection in self._signalcollections:
            for param in signalcollection.params:
                for par in param.params:
                    ret.add(par)

        return sorted(list(ret), key=lambda par: par.name)

        # return sorted({par for signalcollection in self._signalcollections
        #                    for par in signalcollection.params},
        #               key=lambda par: par.name)

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret

    @property
    def pulsarmodels(self):
        return self._signalcollections

    def __repr__(self):
        return "<Enterprise PTA object: " + ", ".join(self.keys()) + ">"

    # emulate a dictionary

    def __len__(self):
        return len(self._signalcollections)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._signalcollections[key]
        else:
            for sc in self._signalcollections:
                if sc.psrname == key:
                    return sc

            raise KeyError("Pulsar model not found")

    def keys(self):
        return [sc.psrname for sc in self._signalcollections]

    def values(self):
        return self._signalcollections

    def items(self):
        return [(sc.psrname, sc) for sc in self._signalcollections]

    # tensor quantities assembled from individual pulsar models

    def get_TNr(self, params):
        return [signalcollection.get_TNr(params) for signalcollection in self._signalcollections]

    def get_TNT(self, params):
        return [signalcollection.get_TNT(params) for signalcollection in self._signalcollections]

    def get_rNr_logdet(self, params):
        return [signalcollection.get_rNr_logdet(params) for signalcollection in self._signalcollections]

    def get_residuals(self):
        return [signalcollection._residuals for signalcollection in self._signalcollections]

    def get_ndiag(self, params={}):
        return [signalcollection.get_ndiag(params) for signalcollection in self._signalcollections]

    def get_delay(self, params={}):
        return [signalcollection.get_delay(params) for signalcollection in self._signalcollections]

    def get_logsignalprior(self, params):
        return [signalcollection.get_logsignalprior(params) for signalcollection in self._signalcollections]

    def set_default_params(self, params):
        for sc in self._signalcollections:
            sc.set_default_params(params)

    def get_basis(self, params={}):
        return [signalcollection.get_basis(params) for signalcollection in self._signalcollections]

    @property
    def _lnlikelihood(self):
        # instantiate on first use
        if not hasattr(self, "_lnlike"):
            self._lnlike = jit(self.lnlikelihood(self))

        return self._lnlike

    def get_lnlikelihood(self, params, **kwargs):
        return self._lnlikelihood(params, **kwargs)

    @property
    def _commonsignals(self):
        # cache the computation if we don't have it yet
        if not hasattr(self, "_cs"):
            commonsignals = collections.defaultdict(collections.OrderedDict)

            for signalcollection in self._signalcollections:
                # TODO: need a better signal that a
                # signalcollection provides a basis

                if signalcollection._Fmat is not None:
                    for signal in signalcollection._signals:
                        # if the CommonSignal is coefficient based we don't
                        # need to worry about it for get_phi and get_phiinv
                        if isinstance(signal, CommonSignal) and not getattr(signal, "_coefficients", {}):
                            commonsignals[signal.__class__][signal] = signalcollection

            # drop common signals that appear only once
            self._cs = {csclass: csdict for csclass, csdict in commonsignals.items() if len(csdict) > 1}

        return self._cs

    # return a dictionary (indexed by SignalCollection) of Python slices
    # corresponding to the span of each pulsar within a Phi matrix
    def _get_slices(self, phivecs):
        ret, offset = {}, 0
        for sc, phivec in zip(self._signalcollections, phivecs):
            # assume phi is either a column vector or a square matrix
            stop = 0 if phivec is None else phivec.shape[0]
            ret[sc] = slice(offset, offset + stop)
            offset = ret[sc].stop

        return ret

    def get_phiinv(self, params, logdet=False, method="cliques"):
        if method == "cliques":
            return self.get_phiinv_byfreq_cliques(params, logdet)
        else:
            raise NotImplementedError

    
    def get_phiinv_byfreq_cliques(self, params, logdet=False, cholesky=False):
        phi = self.get_phi(params, cliques=True)

        if isinstance(phi, list):
            return [None if phivec is None else inv_matrix(phivec) for phivec in phi]
        else:
            ld = 0

            # first invert all the cliques
            for clcount in range(self._clcount):
                idx = self._cliques == clcount

                if np.any(idx):
                    idx2 = np.ix_(idx, idx)

                    if cholesky:
                        cf = sl.cho_factor(phi[idx2])

                        if logdet:
                            ld += 2.0 * jnp.sum(np.log(np.diag(cf[0])))

                        phi[idx2] = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                    else:
                        phi2 = phi[idx2]

                        if logdet:
                            ld += jnp.linalg.slogdet(phi2)[1]

                        phi = phi.at[idx2].set(jnp.linalg.inv(phi2))

            # then do the pure diagonal terms
            idx = self._cliques == -1

            if logdet:
                ld += jnp.sum(jnp.log(phi[idx, idx]))

            phi = phi.at[idx, idx].set(1.0 / phi[idx, idx])

            return (phi, ld) if logdet else phi

    # we use "cliques" to account for sparse non-diagonal Phi matrices
    # for each value in self._cliques, the matrix indices with that value form
    # an independent submatrix that can be inverted separately

    # reset clique index
    def _resetcliques(self, n):
        self._cliques = -1 * np.ones(n)
        self._clcount = 0

    # update clique index by considering a common signal under
    # the assumption that the corresponding "big-Phi" matrix is block diagonal
    def _setcliques(self, slices, csdict):
        # each column in idxmatrix (mind the .T) corresponds to the indices
        # that participate in a common signal for a given pulsar
        idxmatrix = np.array([csc._idx[cs] for cs, csc in csdict.items()]).T

        # each row in the updated idxmatrix corresponds to a set of "global"
        # Phi indices that are correlated across pulsars
        idxmatrix = idxmatrix + np.array([slices[csc].start for cs, csc in csdict.items()])

        # loop over vectors of common-signal-correlated global-indices
        for idxs in idxmatrix:
            # find the existing cliques assigned to these global indices
            allidx = set(self._cliques[idxs])
            maxidx = max(allidx)

            if maxidx == -1:
                # if no clique is found, create a new one, and assign it
                # to the indices in idx

                self._cliques[idxs] = self._clcount

                # I don't think this code is ever exercised...
                # if maxidx == -1, then allidx = [-1]
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques, allidx)] = self._clcount

                self._clcount = self._clcount + 1
            else:
                # if we find at least one clique, assign all indices in idx
                # to the maximum clique index

                self._cliques[idxs] = maxidx

                # since cliques are "contagious", reassign all the other
                # clique indices that we found to maxidx
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques, allidx)] = maxidx

    # add cliques from individual pulsar phis; these will never overlap
    # TO DO: at this point Phi could be defined as a smarter KernelMatrix!
    def _setpulsarcliques(self, slices, phis):
        for sc, phi in zip(self._signalcollections, phis):
            if phi is not None:
                for clindex in range(getattr(phi, "_clcount", 0)):
                    phiind = np.where(phi._cliques == clindex)[0]

                    if len(phiind) > 0:
                        try:
                            self._cliques[slices[sc].start + phiind] = self._clcount
                            self._clcount = self._clcount + 1
                        except Exception:  # pragma: no cover
                            logger.exception("Exception raised in computing cliques")
                            logger.info(self._cliques.shape)
                            logger.info("phiind", phiind, len(phiind))
                            logger.info(slices)
                            raise

    def get_phi(self, params, cliques=False):
        phis = [signalcollection.get_phi(params) for signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            if np.any([phi.ndim == 2 for phi in phis if phi is not None]):
                # if we have any dense matrices,
                Phi = sl.block_diag(*[np.diag(phi) if phi.ndim == 1 else phi for phi in phis if phi is not None])
            else:
                Phi = jnp.diag(jnp.concatenate([phi for phi in phis if phi is not None]))

            # get a dictionary of slices locating each pulsar in Phi matrix
            slices = self._get_slices(phis)

            # self._cliques is a vector of the same size as the Phi matrix
            # for each Phi index i, self._cliques[i] is -1 if row/column
            # belong to no clique, or it gives the clique number otherwise
            if cliques:
                self._resetcliques(Phi.shape[0])
                self._setpulsarcliques(slices, phis)

            # iterate over all common signal classes
            for csclass, csdict in self._commonsignals.items():
                # first figure out which indices are used in this common signal
                # and update the clique index
                if cliques:
                    self._setcliques(slices, csdict)

                # now iterate over all pairs of common signal instances
                pairs = itertools.combinations(csdict.items(), 2)

                for (cs1, csc1), (cs2, csc2) in pairs:
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]
                    block1 = np.arange(block1.start, block1.stop)
                    block2 = np.arange(block2.start, block2.stop)

                    if crossdiag.ndim == 1:
                        Phi = Phi.at[block1[idx1], block2[idx2]].add(crossdiag)
                        Phi = Phi.at[block2[idx2], block1[idx1]].add(crossdiag)
                    else:
                        Phi[block1, block2][np.ix_(idx1, idx2)] += crossdiag
                        Phi[block2, block1][np.ix_(idx2, idx1)] += crossdiag

            return Phi
        else:
            return phis

    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct : ct + n] if n > 1 else float(xs[ct])
            ct += n
        return ret

    def get_lnprior(self, params):
        # map parameter vector if needed
        params = params if isinstance(params, dict) else self.map_params(params)

        return np.sum([p.get_logpdf(params=params) for p in self.params])

    @property
    def pulsars(self):
        return [p.psrname for p in self._signalcollections]

    def _set_signal_dict(self):
        """ Set signal dictionary"""

        self._signal_dict = {}
        sig_list = []
        for ct1, sc in enumerate(self._signalcollections):
            for ct2, sig in enumerate(sc._signals):
                if sig.name not in sig_list:
                    sig_list.append(sig.name)
                    self._signal_dict[sig.name] = sig
                else:
                    msg = "Duplicate signal {} from objects {} and {}."
                    msg += "\nThis functionality was added in v1.1.0 and may"
                    msg += " cause post v1.1.0 functionality to break."
                    msg += "\nThis may not cause other errors but it is"
                    msg += " recommended that you use a custom name for one"
                    msg += " of the duplicate signals.\n"
                    logger.warn(msg.format(sig.name, sig, self._signal_dict[sig.name]))

    @property
    def signals(self):
        """ Return signal dictionary."""
        return self._signal_dict

    def get_signal(self, name):
        """Returns ``Signal`` instance given the signal name."""
        return self._signal_dict[name]

    def summary(self, include_params=True, to_stdout=False):
        """generate summary string for PTA model

        :param include_params: [bool]
            list all parameters for each signal
        :param to_stdout: [bool]
            print summary to `stdout` instead of returning it
        :return: [string]
        """
        summary = "enterprise v" + __version__ + ",  "
        summary += "Python v" + _py_version + "\n"
        summary += "=" * 90 + "\n"
        summary += "\n"
        row = ["Signal Name", "Signal Class", "no. Parameters"]
        summary += "{: <40} {: <30} {: <20}\n".format(*row)
        summary += "=" * 90 + "\n"
        cpcount, copcount = 0, 0
        for sc in self._signalcollections:
            for sig in sc._signals:
                for p in sig.param_names:
                    if sc.psrname not in p:
                        cpcount += 1
                row = [sig.name, sig.__class__.__name__, len(sig.param_names)]
                summary += "{: <40} {: <30} {: <20}\n".format(*row)
                if include_params:
                    summary += "\n"
                    summary += "params:\n"
                    for par in sig._params.values():
                        if isinstance(par, ConstantParameter):
                            copcount += 1
                        summary += "{!s: <90}\n".format(par.__repr__())
                summary += "_" * 90 + "\n"
        summary += "=" * 90 + "\n"
        summary += "Total params: {}\n".format(len(self.param_names) + copcount)
        summary += "Varying params: {}\n".format(len(self.param_names))
        summary += "Common params: {}\n".format(cpcount)
        summary += "Fixed params: {}\n".format(copcount)
        summary += "Number of pulsars: {}\n".format(len(self._signalcollections))
        if to_stdout:
            logger.info(summary)
        else:
            return summary


def SignalCollection(metasignals):  # noqa: C901
    """Class factory for ``SignalCollection`` objects."""

    @six.add_metaclass(MetaCollection)
    class SignalCollection(object):
        _metasignals = metasignals

        def __init__(self, psr):
            self.psrname = psr.name
            # instantiate all the signals with a pulsar
            self._signals = [metasignal(psr) for metasignal in self._metasignals]

            self._residuals = psr.residuals

            self._set_cache_parameters()

        def __add__(self, other):
            return JAXPTA([self, other])

        # TODO: this could be implemented more cleanly
        def _set_cache_parameters(self):
            """ Sets the cache for various signal types."""

            self.white_params = []
            self.basis_params = []
            self.delay_params = []
            for signal in self._signals:
                if signal.signal_type == "white noise":
                    self.white_params.extend(signal.ndiag_params)
                elif signal.signal_type in ["basis", "common basis"]:
                    # to support GP coefficients, and yet do the right thing
                    # for common GPs, which do not have coefficients yet
                    self.delay_params.extend(getattr(signal, "delay_params", []))
                    self.basis_params.extend(signal.basis_params)
                elif signal.signal_type in ["deterministic"]:
                    self.delay_params.extend(signal.delay_params)
                else:
                    msg = "{} signal type not recognized! Caching ".format(signal.signal_type)
                    msg += "may not work correctly for this signal."
                    logger.error(msg)

        # def cache_clear(self):
        #     for instance in [self] + self.signals:
        #         kill = [attr for attr in instance.__dict__ if attr.startswith("_cache")]
        #
        #        for attr in kill:
        #            del instance.__dict__[attr]

        # a candidate for memoization
        @property
        def params(self):
            return sorted({param for signal in self._signals for param in signal.params}, key=lambda par: par.name)

        @property
        def param_names(self):
            ret = []
            for p in self.params:
                if p.size:
                    for ii in range(0, p.size):
                        ret.append(p.name + "_{}".format(ii))
                else:
                    ret.append(p.name)
            return ret

        @property
        def signals(self):
            return self._signals

        def __repr__(self):
            return "<Enterprise SignalCollection object " + self.psrname + ": " + ", ".join(self.keys()) + ">"

        # emulate a dictionary

        def __len__(self):
            return len(self._signals)

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._signals[key]
            else:
                for s in self._signals:
                    if s.signal_id == key:
                        return s

                raise KeyError("Signal model not found")

        def keys(self):
            return [s.signal_id for s in self._signals]

        def values(self):
            return self._signals

        def items(self):
            return [(s.signal_id, s) for s in self._signals]

        # set default parameters

        def set_default_params(self, params):
            for signal in self._signals:
                signal.set_default_params(params)

        def _combine_basis_columns(self, signals):
            """Given a set of Signal objects, each of which may return an
            Fmat (through get_basis()), return a dict (indexed by signal)
            of integer arrays that map individual Fmat columns to the
            combined Fmat.

            Note: The Fmat returned here is simply meant to initialize the
            matrix to save computations when calling `get_basis` later.
            """

            idx, hashlist, cc, nrow = {}, [], 0, None
            for signal in signals:
                Fmat = signal.get_basis()

                if Fmat is not None:
                    nrow = Fmat.shape[0]

                    if not signal.basis_params:
                        idx[signal] = []

                        for i, column in enumerate(Fmat.T):
                            colhash = hash(column.tobytes())

                            if signal.basis_combine and colhash in hashlist:
                                # if we're combining the basis for this signal
                                # and we have seen this column already, make a note
                                # of where it was

                                j = hashlist.index(colhash)
                                idx[signal].append(j)
                            else:
                                # if we're not combining or we haven't seen it already
                                # save the hash and make a note it's new

                                hashlist.append(colhash)
                                idx[signal].append(cc)
                                cc += 1
                    elif signal.basis_params:
                        nf = Fmat.shape[1]
                        idx[signal] = list(range(cc, cc + nf))
                        cc += nf

            if not idx:
                return {}, None
            else:
                ncol = len(np.unique(sum(idx.values(), [])))
                return ({key: np.array(idx[key]) for key in idx.keys()}, np.zeros((nrow, ncol)))

        # goofy way to cache _idx
        def __getattr__(self, par):
            if par in ("_idx", "_Fmat"):
                self._idx, self._Fmat = self._combine_basis_columns(self._signals)
                return getattr(self, par)
            else:
                raise AttributeError("{} object has no attribute {}".format(self.__class__, par))

        def get_ndiag(self, params):
            ndiags = [signal.get_ndiag(params) for signal in self._signals]
            return sum(ndiag for ndiag in ndiags if ndiag is not None)

        def get_delay(self, params):
            delays = [signal.get_delay(params) for signal in self._signals]
            return sum(delay for delay in delays if delay is not None)

        def get_detres(self, params):
            return self._residuals - self.get_delay(params)

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        def get_basis(self, params={}):
            if self._Fmat is None:
                return None

            Fmat = np.zeros_like(self._Fmat)

            for signal in self._signals:
                if signal in self._idx:
                    Fmat[:, self._idx[signal]] = signal.get_basis(params)

            return Fmat

        def get_phiinv(self, params):
            return self.get_phi(params).inv()

        def get_phi(self, params):
            if self._Fmat is None:
                return None

            phi = jnp.zeros(self._Fmat.shape[1])

            for signal in self._signals:
                if signal in self._idx:
                    phi = add_matrices(phi, signal.get_phi(params), self._idx[signal])

            return phi

        def get_TNr(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=T)

        def get_TNT(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            return Nvec.solve(T, left_array=T)

        def get_rNr_logdet(self, params):
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=res, logdet=True)

        # TO DO: cache how?
        def get_logsignalprior(self, params):
            return sum(signal.get_logsignalprior(params) for signal in self._signals)

    return SignalCollection


class ndarray_alt(np.ndarray):
    """Sub-class of ``np.ndarray`` with custom ``solve`` method."""

    def __new__(cls, inputarr):
        obj = np.asarray(inputarr).view(cls)
        return obj

    def __add__(self, other):
        try:
            ret = super(ndarray_alt, self).__add__(other)
        except:
            ret = other + self
        return ret

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            mult = np.array(other / self)
        elif other.ndim == 2:
            mult = np.array(other / self[:, None])
        if left_array is not None:
            mult = np.dot(left_array.T, mult)

        ret = (mult, float(np.sum(np.log(self)))) if logdet else mult
        return ret

class ShermanMorrison(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        self._jvec = jvec
        self._slices = slices
        self._nvec = nvec

    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison(self._jvec, self._slices, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                ZNX -= beta * np.outer(zn.T, xn)
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logdet = np.einsum("i->", np.log(self._nvec))
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                logdet += np.log(jv) - np.log(beta)
        return logdet

    def solve(self, other, left_array=None, logdet=False):

        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise TypeError
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret
