# Provenance: Matlab deepGP demos

Python scripts and notebooks in this directory are ports of the official
AISTATS 2013 Matlab package:

**https://github.com/SheffieldML/deepGP** (`deepGP/matlab/`)

| Paper | Matlab demo | Python entry |
| --- | --- | --- |
| §4.1 fig. 2 hierarchy | `demToyUnsupervised.m` + `hgplvmSampleDataTr1.mat` | `toy_hierarchy.py` |
| §4.1 fig. 3 regression | `demToyRegressionSimple.m` + `demToyDynamicsCreateData.m` (`hierGpsNEW`) | `toy_warped_regression.py` |
| §4.2 figs. 4–5 high-five | `demHighFiveDemonstration.m` + `demHighFiveHgplvm1.mat` | `highfive_shared.py` |
| §4.3 figs. 6–8 USPS | `demUsps.m` / `demDigitsDemonstration.m` (`vercat2`, H=5, Q={15,12,10,8,6}) | `usps_digits.py` |

Walkthrough narrative: `tutorial.m`.

These are still **not** bit-identical to Matlab (different optimiser stack,
kernel implementations, and PyDeepGP API), but data construction, model
depth/width, and evaluation metrics follow the demos above rather than
ad-hoc settings.
