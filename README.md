# Python workflow for benchmarking force fields
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/vtlim/benchmarkff.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/vtlim/benchmarkff/context:python)

README last updated: Jan 8 2020

## About

This repository houses code pertaining to extracting molecule datasets from [QCArchive](https://qcarchive.molssi.org/), running energy minimizations with various force fields, then analyzing the resulting geometries and energies with respect to the QM data from the archive.

## Major dependencies
* Python (numpy, matplotlib, seaborn)
* OpenEye
* RDKit (solely for TFD calculations)
* OpenMM
* OpenForceField
* QCFractal, QCPortal

## Conda setup
```
conda create -n parsley python=3.6 matplotlib numpy seaborn
conda activate parsley
conda install -c openeye -c conda-forge -c omnia rdkit openeye-toolkits qcfractal qcportal openforcefield cmiles openmm
```

## Contents

Directories in this repo:

* `01_setup`: Extract molecules from [QCArchive](https://qcarchive.molssi.org/), convert to OpenEye mols, and standardize conformers and titles.
* `02_calc`
* `03_analysis`
* `examples`: See this directory for example results and plots.

File descriptions:

| directory   | file                             | description |
|-------------|----------------------------------|-------------|
|`01_setup`   |`combine_conformers.ipynb`        | ...         |
|`01_setup`   |`extract_qcarchive_dataset.ipynb` | ...         |
|`02_calc`    |`minimize_ffs.py`                 | ...         |
|`03_analysis`|`cat_mols.py`                     |OpenEye script, modified to ... |

Other [OpenEye scripts](https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples_summary.html) helpful for setup:
* `molchunk.py` -- VTL modified
* `molextract.py`
* [`mols2pdf.py`](https://docs.eyesopen.com/toolkits/python/_downloads/mols2pdf.py)

