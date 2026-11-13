# CIP-0001 artefacts

Supplementary materials for [CIP-0001](../cip0001.md) (not the CIP itself).

| File | Role |
|---|---|
| `discussion-draft.md` | Draft GitHub Discussion post |
| `pydeepgp-numpy-encoder-prototype.patch` | NumPy MLP encoder + remove Theano skip + `np.int`→`np.int64` in MPI typemap |
| `psi_threads_prototype.py` | Monkey-patch of GPy `rbf_psi_comp` for chunked/threaded psi (proof of seam) |

These are analysis prototypes. Landing work goes through normal PRs once the CIP is Accepted.
