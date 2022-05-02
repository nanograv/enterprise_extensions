# Where we put all parts that need to be "unsubclassed,"
# because JAX doesn't play nicely with subclassed numpy arrays

import jax.numpy as jnp
import jax.scipy.linalg as jsl

# KernelMatrix functions


def add_matrices(self, other, idx):
    if other.ndim == 2 and self.ndim == 1:
        self = jnp.zeros(jnp.diag(self))

    if self.ndim == 1:
        self = self.at[idx].add(other)
        # self = self.at[idx].set(self[idx] + other)
    else:
        if other.ndim == 1:
            self[idx, idx] += other
        else:
            self._setcliques(idx)
            idx = (idx, idx) if isinstance(idx, slice) else (idx[:, None], idx)
            self[idx] += other

    return self


def inv_matrix(self):
    if self.ndim == 1:
        inv = 1.0 / self

        return inv, jnp.sum(jnp.log(self))
    else:
        cf = jsl.cho_factor(self)
        inv = jsl.cho_solve(cf, jnp.identity(cf[0].shape[0]))
        ld = 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0])))

        return inv, ld


def set_matrix(self, other, idx):
    if other.ndim == 2 and self.ndim == 1:
        self = jnp.diag(self)

    if self.ndim == 1:
        self = self.at[idx].set(other)

    else:
        if other.ndim == 1:
            self = self.at[idx, idx].set(other)
        else:
            self = matrix_setcliques(self, idx)
            idx = (idx, idx) if isinstance(idx, slice) else (idx[:, None], idx)
            self = self.at[idx].set(other)

    return self


def matrix_setcliques(self, idxs):
    allidx = set(self._cliques[idxs])
    maxidx = max(allidx)

    if maxidx == -1:
        self._cliques[idxs] = self._clcount
        self._clcount = self._clcount + 1
    else:
        self._cliques[idxs] = maxidx
        if len(allidx) > 1:
            self._cliques[jnp.in1d(self._cliques, allidx)] = maxidx
