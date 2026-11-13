# Damianou & Lawrence (AISTATS 2013) — PyDeepGP recreations

Scripts and Colab notebooks that recreate experiments from
[Damianou & Lawrence, *Deep Gaussian Processes*](https://proceedings.mlr.press/v31/damianou13a.html)
(AISTATS 2013) on the maintained `DeepGP` path.

These are **modern recreations**, not bit-identical paper scripts: optimiser
schedules, inducing-point counts, and initialisation differ from the 2013
Matlab stack. Use `--quick` / `QUICK = True` for a short smoke run.

## Notebooks (Colab)

Each notebook starts with an **Open in Colab** badge and a cell that installs
GPy + PyDeepGP when run on Colab.

| Paper section | Notebook |
| --- | --- |
| §4.1 fig. 2 — toy hierarchy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/01_toy_hierarchy.ipynb) [`01_toy_hierarchy.ipynb`](notebooks/01_toy_hierarchy.ipynb) |
| §4.1 fig. 3 — warped regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/02_toy_warped_regression.ipynb) [`02_toy_warped_regression.ipynb`](notebooks/02_toy_warped_regression.ipynb) |
| §4.3 figs. 6–8 — USPS depth / ARD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/03_usps_digits.ipynb) [`03_usps_digits.ipynb`](notebooks/03_usps_digits.ipynb) |
| §4.2 figs. 4–5 — high-five / MRD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/04_highfive_mrd.ipynb) [`04_highfive_mrd.ipynb`](notebooks/04_highfive_mrd.ipynb) |

Locally: `jupyter notebook notebooks/01_toy_hierarchy.ipynb` (with PyDeepGP installed).

## Mapping (scripts)

| Paper section | Script | Status |
| --- | --- | --- |
| §4.1 Toy hierarchy (fig. 2) | `toy_hierarchy.py` | Runnable |
| §4.1 Warped regression (fig. 3) | `toy_warped_regression.py` | Runnable |
| §4.3 USPS {0,1,6} depth / ARD / 1-NN (figs. 6–8) | `usps_digits.py` | Runnable (downloads ESL `zip.train`) |
| §4.2 CMU high-five shared views (figs. 4–5) | `highfive_shared.py` | Runnable via CIP-0002 MRD (synthetic or pods stand-in) |

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
python highfive_shared.py --quick --no-plot --synthetic
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

## High-five / MRD

The high-five experiment uses **conditionally independent observation views**
with a shared parent latent layer (`ObservedMRDLayer`, CIP-0002).
`highfive_shared.py` fits that model; use `--synthetic` when MOCAP data are
unavailable. See also ``examples/example_mrd.py``.

## Citation

```
@inproceedings{damianou2013deep,
  title={Deep {G}aussian {P}rocesses},
  author={Damianou, Andreas and Lawrence, Neil},
  booktitle={AISTATS},
  year={2013}
}
```
