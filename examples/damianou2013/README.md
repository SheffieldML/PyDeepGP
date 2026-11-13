# Damianou & Lawrence (AISTATS 2013) — PyDeepGP recreations

Scripts that recreate the **feasible** experiments from
[Damianou & Lawrence, *Deep Gaussian Processes*](https://proceedings.mlr.press/v31/damianou13a.html)
(AISTATS 2013) on the maintained `DeepGP` path.

These are **modern recreations**, not bit-identical paper scripts: optimiser
schedules, inducing-point counts, and initialisation differ from the 2013
Matlab stack. Use `--quick` for a short smoke run.

## Mapping

| Paper section | Script | Status |
| --- | --- | --- |
| §4.1 Toy hierarchy (fig. 2) | `toy_hierarchy.py` | Runnable |
| §4.1 Warped regression (fig. 3) | `toy_warped_regression.py` | Runnable |
| §4.3 USPS {0,1,6} depth / ARD / 1-NN (figs. 6–8) | `usps_digits.py` | Runnable (downloads ESL `zip.train`) |
| §4.2 CMU high-five shared views (figs. 4–5) | `highfive_shared.py` | Documented gap — MRD currently fails at bound aggregation; `--force-mrd` reports the error |

## Setup

From the repository root (Python ≥ 3.11, GPy ≥ 1.15):

```bash
python -m pip install -e .
# optional: matplotlib for figures; already a project dependency
```

USPS data are cached under `examples/damianou2013/data/zip.train.gz` on first run
(ElemStatLearn `zip.train`).

## Quick smoke tests

```bash
cd examples/damianou2013
python toy_hierarchy.py --quick --no-plot
python toy_warped_regression.py --quick --no-plot
python usps_digits.py --quick --no-plot --depths 1 2
python highfive_shared.py          # prints status; exit code 2
```

Save figures without interactive display:

```bash
python toy_hierarchy.py --quick --no-plot --outdir /tmp/dgp2013
```

## Paper-scale runs

Omit `--quick` (slower; especially USPS depth 1–5):

```bash
python toy_warped_regression.py --repeats 10 --outdir /tmp/dgp2013
python usps_digits.py --depths 1 2 3 4 5 --outdir /tmp/dgp2013
```

## High-five / MRD gap

The high-five experiment uses **conditionally independent observation views**
with a shared parent latent layer. That construction is still present as
`ObservedMRDLayer`, but it is outside CI coverage. `highfive_shared.py`
explains the gap; `--force-mrd` attempts an exploratory fit only.

## Citation

```
@inproceedings{damianou2013deep,
  title={Deep {G}aussian {P}rocesses},
  author={Damianou, Andreas and Lawrence, Neil},
  booktitle={AISTATS},
  year={2013}
}
```
