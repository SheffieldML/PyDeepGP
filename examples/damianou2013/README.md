# Damianou & Lawrence (AISTATS 2013) — PyDeepGP ports

Ports of the official Matlab demos in
[SheffieldML/deepGP](https://github.com/SheffieldML/deepGP)
(`deepGP/matlab/`). See [SOURCE.md](SOURCE.md) for the file-level mapping.

These are **not** bit-identical to Matlab (optimiser / kernel stack differ), but
they use the same reference data files, model widths, and evaluation metrics
as the demos — not invented protocols.

Paper: [Damianou & Lawrence, AISTATS 2013](https://proceedings.mlr.press/v31/damianou13a.html).

## Notebooks (Colab)

| Paper | Notebook | Matlab demo |
| --- | --- | --- |
| §4.1 fig. 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/01_toy_hierarchy.ipynb) [01](notebooks/01_toy_hierarchy.ipynb) | `demToyUnsupervised.m` |
| §4.1 fig. 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/02_toy_warped_regression.ipynb) [02](notebooks/02_toy_warped_regression.ipynb) | `demToyRegressionSimple.m` |
| §4.3 figs. 6–8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/03_usps_digits.ipynb) [03](notebooks/03_usps_digits.ipynb) | `demUsps.m` / `demDigitsDemonstration.m` |
| §4.2 figs. 4–5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SheffieldML/PyDeepGP/blob/main/examples/damianou2013/notebooks/04_highfive_mrd.ipynb) [04](notebooks/04_highfive_mrd.ipynb) | `demHighFiveDemonstration.m` |

Until this PR is on `main`, open Colab from the branch, e.g.
`…/blob/examples-damianou2013/examples/damianou2013/notebooks/…`.

## Scripts

| Script | Data | Notes |
| --- | --- | --- |
| `toy_hierarchy.py` | `data/hgplvmSampleDataTr1.mat` | Two-view DeepGP Q={6,4}; vs stacked PCA & Isomap |
| `toy_warped_regression.py` | `hierGpsNEW` generator | Defaults = Matlab simple (N=150,D=15,Ntr=25); `--paper` → N=120,D=10 |
| `highfive_shared.py` | `data/demHighFiveHgplvm1.mat` | YA/YB 78×62; plots vs stored hgplvm latents |
| `usps_digits.py` | ESL `zip.train` | `vercat2` 50×{0,6,1}; Q={15,12,10,8,6}; K=80 |

```bash
cd examples/damianou2013
python toy_hierarchy.py --quick --no-plot
python toy_warped_regression.py --quick --no-plot
python highfive_shared.py --quick --no-plot
python usps_digits.py --quick --no-plot --depths 1
```

Omit `--quick` for demo-scale optimisation (slow, especially USPS depth 5).

## Setup

```bash
python -m pip install -e .  # from repo root
# Isomap comparison in toy_hierarchy: pip install scikit-learn
```

## Citation

```
@inproceedings{damianou2013deep,
  title={Deep {G}aussian {P}rocesses},
  author={Damianou, Andreas and Lawrence, Neil},
  booktitle={AISTATS},
  year={2013}
}
```
