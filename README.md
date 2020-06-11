# BenchmarkFF
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/vtlim/benchmarkff.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/vtlim/benchmarkff/context:python)

README last updated: June 7 2020

## About

Overview: Compare optimized geometries and energies from various force fields with respect to a QM reference.

This repository comprises code to extract molecule datasets from [QCArchive](https://qcarchive.molssi.org/), run energy minimizations with various force fields, and analyze the resulting geometries and energies with respect to QM reference data from QCArchive.

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
* `02_calc`: Run energy minimizations for various force fields
* `03_analysis`: Analyze output energies and geometries
* `examples`: See this directory for example results and plots.

File descriptions:

| directory   | file                             | description |
|-------------|----------------------------------|-------------|
|`01_setup`   |`extract_qcarchive_dataset.ipynb` |write out molecules from a QCArchive database which have completed QM calculations|
|`01_setup`   |`combine_conformers.ipynb`        |of the molecules from `extract_qcarchive_dataset.ipynb`, combine conformers that are labeled as different molecules|
|`02_calc`    |`minimize_ffs.py`                 |minimize all molecules in an input SDF file with a specified force field|
|`03_analysis`|`color_by_moiety.py`              |generate ddE vs TFD (or RMSD) scatter plots highglighting specific moieties by color|
|`03_analysis`|`compare_ffs.py`                  |compare FF-minimized molecules on their geometries and energies (no conformer matching)|
|`03_analysis`|`match_minima.py`                 |similar to `compare_ffs` of comparing geometries and energies but analyzing RMSD-matched structures|
|`03_analysis`|`probe_parameter.py`              |find all molecules in a set that use certain specified parameter(s)|
|`03_analysis`|`reader.py`                       |reader for molecule sets and text input files called by the other analysis scripts|
|`03_analysis`|`tailed_parameters.py`            |identify parameters that may be overrepresented in high RMSD/TFD tails for FFXML force fields|


## Brief overview

### Setup
1. Write out from a QCArchive database which have completed QM calculations, using `extract_qcarchive_dataset.ipynb`.
2. Reorganize up the molecule set for conformers which are labeled as different molecules:
    1. Group all the same conformers together since they are separated by intervening molecules, using `combine_conformers.ipynb`.
    2. Of the full set, write out the good molecules that don't need conformer reorganization, using `molextract.py`\* from OEChem.
    3. Combine the results of steps 2.1 and 2.2: `cat whole_02_good.sdf whole_03_redosort.sdf > whole_04_combine.sdf`
    4. Read in the whole set with proper conformers, and rename titles for numeric order, using `combine_conformers.ipynb`.
    5. (opt) Write out associated SMILES: `awk '/SMILES/{getline; print}' whole_05_renew.sdf > whole_05_renew.smi`
    6. (opt) Generate molecular structures in PDF format, using [`mols2pdf.py`](https://docs.eyesopen.com/toolkits/python/_downloads/mols2pdf.py).

### FF calculations
3. (opt) Break up the full set into smaller chunks for managable computations, using `molchunk.py`.\*
4. Run the minimizations, using `minimize_ffs.py`.
5. Remove the molecules that were unable to minimize (e.g., due to missing parameters) from all files, using `cat_mols.py`.\*
6. (opt) If the full set was broken up, concatenate all the constituent files back together, using `cat` or `cat_mols.py`.\*

### Analysis
7. Analyze geometries and relative energies with `compare_ffs.py`.
    1. (opt) If some conformers (not full molecules) are outliers, can remove using `get_by_tag.py`.
    2. Mark certain moieties of interest in energy v. geometry scatter plots using `color_by_moiety.py`.
8. Analyze energies of structurally similar conformers using `match_minima.py`.
9. Explore parameters overrepresented in high TFD/RMSD regions using `tailed_parameters.py` and `probe_parameter.py`.

\*The OEChem scripts referred to above are located [here](https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples_summary.html).
* `molextract.py`
* `molchunk.py` -- VTL modified to use `OEAbsCanonicalConfTest`

**Note**: Some of the analysis can take a long time for multiple force fields and many molecules
(e.g. up to 2 hours on `compare_ffs.py` or 30-45 min on `tailed_parameters.py`).
To explore the analyzed data, adjust plots, etc. without re-analyzing data, you can
input the pickle file written out from the previously run analysis.

## Contributors
* Victoria Lim (author)
* David Mobley (advising)
* Jeffrey Wagner, Daniel Smith (code review)
* Jessica Maat, Caitlin Bannan, Hyesu Jang, Lee-Ping Wang, Chris Bayly (discussions)

## Big picture wish list / to do tasks
See more science-focused issues in the Github issue tracker.
* Format code with YAPF/Black
* Use logging module instead of print statements
* Look into automatically serializable representations (e.g., Pydantic) instead of pickle
* Use type hints for functions
* Allow user to pass in dict for plotting parameters (i.e., talk or paper font sizes)
* Generate plots with Plotly

