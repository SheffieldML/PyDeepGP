"""Chunked + threaded RBF psi-statistics: pure NumPy, no MPI. NumPy ufuncs/BLAS release the GIL."""
import os, numpy as np
from concurrent.futures import ThreadPoolExecutor
from GPy.kern.src.psi_comp import rbf_psi_comp as R
p1 = getattr(R, "__psi1computations"); p2 = getattr(R, "__psi2computations")
T = int(os.environ.get("PSI_THREADS", "1")); CH = int(os.environ.get("PSI_CHUNK", "500"))
pool = ThreadPoolExecutor(max_workers=T)

def _chunks(N):
    k = max(T, -(-N // CH)); return np.array_split(np.arange(N), k)

def psicomputations(variance, lengthscale, Z, vp, return_psi2_n=False):
    mu, S = vp.mean.values, vp.variance.values; idx = _chunks(mu.shape[0])
    def work(i):
        a = p1(variance, lengthscale, Z, mu[i], S[i]); b = p2(variance, lengthscale, Z, mu[i], S[i])
        return a, (b if return_psi2_n else b.sum(0))
    res = list(pool.map(work, idx))
    psi1 = np.vstack([r[0] for r in res])
    psi2 = np.concatenate([r[1] for r in res]) if return_psi2_n else sum(r[1] for r in res)
    return np.full(mu.shape[0], np.ravel(variance)[0]), psi1, psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, vp):
    mu, S = vp.mean.values, vp.variance.values; idx = _chunks(mu.shape[0]); ARD = len(lengthscale) != 1
    d2 = dL_dpsi2 if dL_dpsi2.ndim == 2 else None
    def work(i):
        a = R._psi1compDer(dL_dpsi1[i], variance, lengthscale, Z, mu[i], S[i])
        b = R._psi2compDer(d2 if d2 is not None else dL_dpsi2[i], variance, lengthscale, Z, mu[i], S[i])
        return a, b
    res = list(pool.map(work, idx))
    dvar = np.sum(dL_dpsi0) + sum(a[0] + b[0] for a, b in res)
    dl = sum(a[1] + b[1] for a, b in res); dZ = sum(a[2] + b[2] for a, b in res)
    dmu = np.vstack([a[3] + b[3] for a, b in res]); dS = np.vstack([a[4] + b[4] for a, b in res])
    return dvar, (dl if ARD else dl.sum()), dZ, dmu, dS

R.psicomputations = psicomputations; R.psiDerivativecomputations = psiDerivativecomputations
