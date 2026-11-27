# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)




from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv
from GPy.util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from GPy.inference.latent_function_inference.var_dtc import VarDTC as GPyVarDTC
from GPy.inference.latent_function_inference.posterior import Posterior

log_2_pi = np.log(2*np.pi)

from ..util.util import reject_mpi_comm

class VarDTC(GPyVarDTC):
    """
    Sparse variational DTC inference for deep GP layers.

    Subclasses :class:`GPy.inference.latent_function_inference.var_dtc.VarDTC`
    (CIP-0003 Phase 2). Compared with GPy's class this adds only:

    * ``psi_chunk_size`` / ``psi_threads``: chunked psi-statistics (CIP-0001).
    * ``Kuu_sigma``: per-inducing-point diagonal added to ``Kuu`` (hidden layers).
    * uncertain outputs: ``Y`` may be a ``VariationalPosterior`` (the latent
      variables of the layer below), which GPy's class does not support.
    * ``const_jitter = 1e-6`` and a noise-variance floor of ``1e-6``, matching
      the behaviour that the reference-value tests were generated with.

    Everything else (algebra, gradients, posterior, ``sparse_linalg`` modes)
    is GPy's. For deterministic ``Y`` the two classes agree to ~1e-11.
    """
    const_jitter = 1e-6
    _variance_floor = 1e-6

    def __init__(self, mpi_comm=None, psi_chunk_size=None, psi_threads=1,
                 sparse_linalg="stable"):
        reject_mpi_comm(mpi_comm)
        try:
            super(VarDTC, self).__init__(limit=1, sparse_linalg=sparse_linalg)
        except TypeError:
            # GPy releases without CIP-0013 have no ``sparse_linalg`` argument.
            if sparse_linalg != "stable":
                raise
            super(VarDTC, self).__init__(limit=1)
        # Newer GPy sets ``const_jitter`` per instance in ``__init__``, which would shadow the
        # class attribute above; reassert our convention.
        self.const_jitter = type(self).const_jitter
        self.psi_chunk_size = psi_chunk_size
        self.psi_threads = 1 if psi_threads is None else int(psi_threads)

    def __getstate__(self):
        state = super(VarDTC, self).__getstate__()
        return state, self.psi_chunk_size, self.psi_threads

    def __setstate__(self, state):
        inner, self.psi_chunk_size, self.psi_threads = state
        super(VarDTC, self).__setstate__(inner)

    def psi_statistics(self, kern, X, Z):
        """Unscaled ``(psi0, psi1, psi2)`` for uncertain inputs, chunked if requested.

        ``psi0`` is per-point, ``psi1`` is N x M, ``psi2`` is M x M (summed over points).
        """
        from ..util.psi_chunking import configure_kern_psicomp, row_slices

        # Prefer GPy R12 chunking/threading when the kernel supports it.
        psicomp_chunked = configure_kern_psicomp(kern, self.psi_chunk_size, self.psi_threads)
        N = X.shape[0]
        chunk = self.psi_chunk_size
        if chunk is not None and 0 < int(chunk) < N and not psicomp_chunked:
            # CIP-0001 R14: serial map-reduce over row chunks for any kernel.
            psi0, psi1, psi2 = [], [], None
            for sl in row_slices(N, chunk):
                Xc = X[sl]
                psi0.append(kern.psi0(Z, Xc))
                psi1.append(kern.psi1(Z, Xc))
                psi2_c = kern.psi2(Z, Xc)
                psi2 = psi2_c if psi2 is None else psi2 + psi2_c
            return np.hstack(psi0), np.vstack(psi1), psi2
        return kern.psi0(Z, X), kern.psi1(Z, X), kern.psi2(Z, X)

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None, Lm=None,
                  dL_dKmm=None, Kuu_sigma=None, **kwargs):
        if isinstance(Y, VariationalPosterior):
            return self._inference_uncertain_outputs(
                kern, X, Z, likelihood, Y, Y_metadata, Lm, dL_dKmm, Kuu_sigma)
        if Kuu_sigma is not None and Lm is None:
            Kmm = kern.K(Z).copy()
            diag.add(Kmm, Kuu_sigma)
            Lm = jitchol(Kmm)
        if isinstance(X, VariationalPosterior):
            psi0, psi1, psi2 = self.psi_statistics(kern, X, Z)
            kwargs.update(psi0=psi0, psi1=psi1, psi2=psi2)
        precision = 1. / np.fmax(likelihood.variance, self._variance_floor)
        post, logL, grad_dict = super(VarDTC, self).inference(
            kern, X, Z, likelihood, Y, Y_metadata=Y_metadata, Lm=Lm,
            dL_dKmm=dL_dKmm, precision=np.asarray(precision).reshape(-1), **kwargs)
        # Layers expect a length-1 vector (GPy returns a 1x1 matrix) and no dL_dm.
        grad_dict.pop('dL_dm', None)
        return post, np.asarray(logL).reshape(1), grad_dict

    def _inference_uncertain_outputs(self, kern, X, Z, likelihood, Y, Y_metadata=None,
                                     Lm=None, dL_dKmm=None, Kuu_sigma=None):
        """Uncertain outputs: Y is the q(X) of the layer below (not in GPy).

        Computes the statistics and delegates the algebra to the pure
        :func:`deepgp.functional.collapsed_bound`.
        """
        from ..functional import collapsed_bound
        beta = 1. / np.fmax(likelihood.variance, self._variance_floor)
        Kmm = kern.K(Z).copy()
        diag.add(Kmm, Kuu_sigma if Kuu_sigma is not None else self.const_jitter)
        if isinstance(X, VariationalPosterior):
            psi0, psi1, psi2 = self.psi_statistics(kern, X, Z)
        else:
            psi0, psi1, psi2 = kern.Kdiag(X), kern.K(X, Z), None
        r = collapsed_bound(Kmm, psi0, psi1, psi2, Y.mean, float(beta[0]) if np.ndim(beta) else float(beta),
                            Y_var=Y.variance)
        post = Posterior(woodbury_inv=r['woodbury_inv'], woodbury_vector=r['woodbury_vector'],
                         K=Kmm, mean=None, cov=r['cov'], K_chol=r['Lm'])
        grad_dict = {'dL_dKmm': r['dL_dKmm'], 'dL_dpsi0': r['dL_dpsi0'],
                     'dL_dpsi1': r['dL_dpsi1'], 'dL_dpsi2': r['dL_dpsi2'],
                     'dL_dthetaL': r['dL_dthetaL'],
                     'dL_dYmean': r['dL_dYmean'], 'dL_dYvar': r['dL_dYvar']}
        if r['dL_dpsi2'] is None:
            grad_dict = {'dL_dKmm': r['dL_dKmm'], 'dL_dKdiag': r['dL_dpsi0'],
                         'dL_dKnm': r['dL_dpsi1'], 'dL_dthetaL': r['dL_dthetaL'],
                         'dL_dYmean': r['dL_dYmean'], 'dL_dYvar': r['dL_dYvar']}
        return post, r['logL'], grad_dict
