"""Generate ``gpc_reference.json``: GPc CIP-0035 reference values (not run in CI).

Run with the GPc virtual environment (it provides the compiled ``gpc._gpc``)::

    cd /path/to/GPc/python
    .venv/bin/python /path/to/PyDeepGP/deepgp/testing/gpc_reference_generate.py \
        /path/to/PyDeepGP/deepgp/testing/gpc_reference.json

For each case we build a GPc ``DeepGP`` on fixed data, overwrite its packed
stochastic parameter vector with random values, reload it, and record the
parameters (in GPc's parameterisation) with GPc's ELBO.  ``test_gpc_reference``
then evaluates the same parameters with ``doubly_stochastic_bound``.

Packed layout per layer (``CSvgp::getStochasticParams``): kernel log-parameters
(``inverseWidth``, ``variance``), inducing inputs ``Z`` (column major, M x Q),
log beta, then for each output the M posterior means followed by the lower
Cholesky factor of S in column-major order with the diagonal stored as logs.
GPc's RBF is ``k = variance * exp(-0.5 * inverseWidth * r^2)``.
"""
import json
import sys
import tempfile
import os

import importlib.util
import glob

import numpy as np

try:
    from gpc._gpc import DeepGP
except ImportError:     # package __init__ can be newer than the compiled module: load the module alone
    _so = glob.glob(os.path.join(os.getcwd(), 'gpc', '_gpc*.so'))[0]
    _spec = importlib.util.spec_from_file_location('gpc._gpc', _so)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DeepGP = _mod.DeepGP


def _build(X, Y, L, M, hidden, seed, likelihood='gaussian'):
    rng = np.random.RandomState(seed)
    m = DeepGP(X, Y, layers=L, hidden=hidden, num_inducing=M, likelihood=likelihood)
    path = os.path.join(tempfile.mkdtemp(), 'ref.model')
    m.save(path)
    t = open(path).read().split('\n')
    idx = [k for k, l in enumerate(t) if l.startswith('numStochParams')]
    widths = [X.shape[1]] + [hidden] * (L - 1) + [Y.shape[1]]
    layers = []
    for l, i in enumerate(idx):
        Q, P = widths[l], widths[l + 1]
        K = int(t[i].split('=')[1])
        old = [float.fromhex(v) for v in t[i + 6].split()]
        nq = P * (M + M * (M + 1) // 2)
        nk = K - M * Q - 1 - nq
        logk = rng.randn(nk) * 0.3
        Z = rng.randn(M, Q)
        logbeta = (np.log(5.) + 0.2 * rng.randn()) if l == L - 1 \
            else old[nk + M * Q]
        qm = 0.5 * rng.randn(M, P)
        chol = np.stack([np.tril(0.15 * rng.randn(M, M))
                         + np.diag(0.5 + 0.3 * rng.rand(M)) for _ in range(P)])
        v = list(logk) + list(Z.T.ravel()) + [logbeta]
        for p in range(P):
            v += list(qm[:, p])
            for c in range(M):
                for r in range(c, M):
                    v.append(np.log(chol[p][r, c]) if r == c else chol[p][r, c])
        assert len(v) == K
        t[i + 6] = ' '.join(float.hex(float(x)) for x in v) + ' '
        layers.append(dict(log_inverse_width=float(logk[0]),
                           log_variance=float(logk[1]),
                           Z=Z.tolist(), log_beta=float(logbeta),
                           q_mu=qm.tolist(), q_chol=chol.tolist()))
    out = os.path.join(os.path.dirname(path), 'ref2.model')
    open(out, 'w').write('\n'.join(t))
    return DeepGP.load(out), layers


def main(target, n_mc=4000):
    rng = np.random.RandomState(0)
    X = rng.randn(20, 2)
    Y = np.sin(X[:, :1]) + 0.1 * rng.randn(20, 1)
    Xt = rng.randn(5, 2)
    cases = []
    for name, L, seed in (('one_layer', 1, 1), ('two_layer', 2, 2),
                          ('three_layer', 3, 3)):
        g, layers = _build(X, Y, L, 4, 2, seed)
        case = dict(name=name, num_layers=L, layers=layers)
        if L == 1:
            case['elbo'] = g.elbo()
            case['layer_kl'] = [g.layer_kl(0)]
            mu, var = g.predict(Xt)
            case['pred_mean'] = np.asarray(mu).tolist()
            case['pred_var'] = np.asarray(var).tolist()
        else:
            vals = np.array([g.elbo() for _ in range(n_mc)])
            case['elbo_mean'] = float(vals.mean())
            case['elbo_stderr'] = float(vals.std() / np.sqrt(n_mc))
            case['layer_kl'] = [g.layer_kl(i) for i in range(L)]
        cases.append(case)
    # Non-Gaussian top layers. One layer is exact (same 20-point Gauss-Hermite rule); two layers
    # are a Monte Carlo mean as above. Labels and counts are drawn from the same inputs.
    r = np.random.RandomState(7)
    f0 = np.sin(X[:, :1]) + 0.5 * X[:, 1:]
    Ydata = {'probit': np.where(f0 + 0.3 * r.randn(20, 1) > 0, 1., -1.),
             'poisson': r.poisson(np.exp(0.5 * f0)).astype(float),
             'student_t': f0 + 0.3 * r.standard_t(3, (20, 1))}
    lik_cases = []
    for lik, Yl in Ydata.items():
        for L, seed in ((1, 11), (2, 12)):
            g, layers = _build(X, Yl, L, 4, 2, seed, likelihood=lik)
            case = dict(name='%s_%d' % (lik, L), likelihood=lik, num_layers=L, layers=layers,
                        Y=Yl.tolist(), layer_kl=[g.layer_kl(i) for i in range(L)])
            if L == 1:
                case['elbo'] = g.elbo()
            else:
                vals = np.array([g.elbo() for _ in range(n_mc)])
                case['elbo_mean'] = float(vals.mean())
                case['elbo_stderr'] = float(vals.std() / np.sqrt(n_mc))
            lik_cases.append(case)
    data = dict(likelihood_cases=lik_cases, X=X.tolist(), Y=Y.tolist(), Xtest=Xt.tolist(), hidden=2,
                num_inducing=4, cases=cases)
    with open(target, 'w') as f:
        json.dump(data, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
