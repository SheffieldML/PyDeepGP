"""MRD with the doubly stochastic model: two views, one shared and one private factor each.

Prints each view's ARD relevances (1 / lengthscale^2, normalised) over the three latent
dimensions. A view should rely on the shared factor and its own private one, not the other's.
"""
import sys
import numpy as np
from deepgp.models.uncollapsed import DoublyStochasticDeepGP


def main(steps=1500, seed=0, back_constraint=False):
    rng = np.random.RandomState(seed)
    N = 120
    shared, pa, pb = (np.sin(np.linspace(0, 8, N)), np.cos(np.linspace(0, 20, N)),
                      np.sign(np.sin(np.linspace(0, 13, N))) * 0.8)
    W = lambda d: rng.randn(d, 2)
    A = np.stack([shared, pa], 1).dot(W(6).T) + 0.05 * rng.randn(N, 6)
    B = np.stack([shared, pb], 1).dot(W(5).T) + 0.05 * rng.randn(N, 5)
    m = DoublyStochasticDeepGP([[6, 5], 3], [A, B], None, num_inducing=20, noise_var=0.05,
                               seed=seed, back_constraint=back_constraint,
                               q_sqrt_x=0.3)
    m.fit_minibatch(steps=steps, batch_size=60, lr=0.02)
    for v, h in enumerate(m.heads):
        rel = 1. / np.asarray(h.kern.lengthscale.values) ** 2
        print('view %d relevance' % v, np.round(rel / rel.sum(), 3))
    print('latent correlation with [shared, private A, private B]:')
    X = m.latent_mean()
    print(np.round(np.abs(np.corrcoef(X.T, np.stack([shared, pa, pb])) [:3, 3:]), 2))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500, back_constraint='bc' in sys.argv)
