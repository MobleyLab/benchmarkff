# Python workflow for benchmarking force fields

Last updated: Nov 11 2019

Some relevant scripts by [OpenEye](https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples_summary.html).

## Conda setup

## `01_setup`
Extract molecules from [QCArchive](https://qcarchive.molssi.org/), convert to OpenEye mols, and standardize conformers and titles.

| file                             | description |
|----------------------------------|-------------|
|`combine_conformers.ipynb`        |             |
|`extract_qcarchive_dataset.ipynb` |             |

Also relevant from OpenEye:
* `molchunk.py` -- VTL modified to
* `molextract.py`
* [`mols2pdf.py`](https://docs.eyesopen.com/toolkits/python/_downloads/mols2pdf.py)

## `02_calc`
* `minimize_ffs.py`
