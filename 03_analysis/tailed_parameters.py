#!/usr/bin/env python

"""
tailed_parameters.py

For distributions of RMSD or TFDs of force field geometries with respect to
reference geometries, identify outlier molecules in the high RMSD (TFD) tails
above a predetermined cutoff value. For each force field parameter in the
molecule set, determine the fraction of outlier molecules with that parameter.
Compare that to the fraction of all molecules that use that parameter.

By:      Victoria T. Lim
Version: Feb 6 2020

References:
https://github.com/openforcefield/openforcefield/blob/master/examples/inspect_assigned_parameters/inspect_assigned_parameters.ipynb

Note:
- Make sure to use the same FFXML file corresponding to the minimized geometries.

Examples:
$ python tailed_parameters.py -i refdata_parsley.sdf -f openff_unconstrained-1.0.0-RC2.offxml --rmsd --cutoff 1.0  --tag "RMSD to qcarchive.sdf" --tag_smiles "SMILES QCArchive" > tailed.dat
$ python tailed_parameters.py -i refdata_parsley.sdf -f openff_unconstrained-1.0.0-RC2.offxml --tfd  --cutoff 0.12 --tag "TFD to qcarchive.sdf"  --tag_smiles "SMILES QCArchive" >> tailed.dat

"""

import os
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

import openeye.oechem as oechem

from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils.structure import get_molecule_parameterIDs

import reader


### ------------------- Functions -------------------


def natural_keys(text):
    """
    Natural sorting of strings containing numbers.
    https://stackoverflow.com/a/5967539/8397754
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def write_mols(mols_dict, outfile):
    """
    Save all mols in the given dictionary to 'outfile'.

    Parameters
    ----------
    mols_dict : dict of dicts
        the first level key is the SMILES string and the value of that key is
        a dict with the following key/value pairs--
            metric      geometric measurement
            structure   OEGraphMol of the structure
    outfile : string
        name of output file
    """

    # open an outstream file
    ofs = oechem.oemolostream()
    if not ofs.open(outfile):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % outfile)

    # go through all the molecules
    for key in mols_dict:
        print(f"writing out {key}")
        mymol = mols_dict[key]['structure']
        oechem.OEWriteConstMolecule(ofs, mymol)


def get_parameters(mols_dict, ffxml):
    """
    For a group of structures, call the Open Force Field function
    get_molecule_parameterIDs to identify parameter assignment, grouped
    by molecule and grouped by parameter.

    Parameters
    ----------
    mols_dict : dict of dicts
        the first level key is the SMILES string and the value of that key is
        a dict with the following key/value pairs--
            metric      geometric measurement
            structure   OEGraphMol of the structure
    ffxml : string
        name of FFXML force field file

    Returns
    -------
    parameters_by_molecule : dict
        key is isosmiles generated by Open Force Field internal code;
        value is a list of parameter IDs associated with this molecule
    parameters_by_ID : dict
        key is parameter ID;
        value is a list of isosmiles for all the molecules that have this ID
    smi_dict : dict
        key is isosmiles;
        value is the molecular identifier from the input SDF file
    """

    # load in force field
    ff = ForceField(ffxml)

    # convert OEMols to open force field molecules
    off_mols = []
    smi_dict = {}

    for i, key in enumerate(mols_dict):
        # get mol from the dict
        mymol = mols_dict[key]['structure']

        # create openforcefield molecule from OEMol
        # note: stereo error raised even though coordinates present (todo?)
        off_mol = Molecule.from_openeye(mymol, allow_undefined_stereo=True)
        off_mols.append(off_mol)

        # form a dictionary to backtrace the iso_smiles to original molecule
        smi_dict[off_mol.to_smiles()] = key

    # remove duplicate molecules (else get_molecule_parameterIDs gives err)
    iso_smiles = [ molecule.to_smiles() for molecule in off_mols ]
    idx_of_duplicates = [idx for idx, item in enumerate(iso_smiles) if item in iso_smiles[:idx]]
    for index in sorted(idx_of_duplicates, reverse=True):
        del off_mols[index]

    # create dictionaries describing parameter assignment,
    # grouped both by molecule and by parameter
    parameters_by_molecule, parameters_by_ID = get_molecule_parameterIDs(off_mols, ff)

    return parameters_by_molecule, parameters_by_ID, smi_dict


def count_mols_by_param(full_params, params_id_all, params_id_out):
    """
    Count the number of mols from the full set and the outlier set
    that have each parameter.

    Parameters
    ----------
    full_params : list
        alphabetized list of all parameters in the whole set of molecules
    params_id_all : dict
        key is parameter ID;
        value is a list of isosmiles for all the molecules that have this ID
    params_id_out : dict
        same format as params_id_all but with only the subset of outlier mols

    Returns
    -------
    nmols_cnt_all : 1D numpy array
        nmols_cnt_all[i] is the count of molecules that have the parameter
        found in full_params[i]
    nmols_cnt_out : 1D numpy array
        same format as nmols_cnt_all but with only the subset of outlier mols
    """

    nmols_cnt_all = []
    nmols_cnt_out = []

    for i, p in enumerate(full_params):

        # count number of mols in the COMPLETE set which use this parameter
        cnt_all = len(params_id_all[p])

        # count number of mols in the OUTLIER set which use this parameter
        try:
            cnt_out = len(params_id_out[p])
        except KeyError:
            cnt_out = 0

        nmols_cnt_all.append(cnt_all)
        nmols_cnt_out.append(cnt_out)

    return np.array(nmols_cnt_all), np.array(nmols_cnt_out)


def plot_param_bars(plot_data, labels, max_ratio, suffix, num_sort=False, what_for='talk'):
    """
    Generate bar plots of the ratio of (fraction of outlier mols with
    a given parameter) to (fraction of all mols with a given parameter).
    The ratios should be pre-computed before calling this function.

    Parameters
    ----------
    plot_data : 1D numpy array
        array of ratios to plot
    labels : 1D numpy array
        array of strings that correspond to plot_data
    max_ratio : float
        max ratio of all the data that will be plotted to generate consistent
        plot limits for all outputs
    suffix: string
        label to append to end of plot filename
    num_sort : Boolean
        True to sort bar plot numerically, False for alphabetical order
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"
    """
    def single_subplot(ax, y, plot_data, labels):
        ax.barh(y, plot_data, bar_width, color='darkcyan')

        # add plot labels, ticks, and tick labels
        ax.set_xlabel('fraction', fontsize=fs1)
        ax.tick_params(axis='x', labelsize=fs2)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=fs2)

        # invert for horizontal bars
        ax.invert_yaxis()

        # set plot limits by rounding max_ratio to the nearest 0.5
        x_max = round(max_ratio * 2) / 2
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.linspace(0, x_max, 6))

        # set grid
        ax.grid(True)

        # add a reference line at ratio = 1.0
        ax.axvline(1.0, ls='--', c='purple', alpha=0.6)

        # set alternating colors for background for ease of visualizing
        locs, values = plt.xticks()
        for i in range(1, len(locs)-1, 2):
            ax.axvspan(locs[i], locs[i+1], facecolor='lightgrey', alpha=0.25)

    n_bars = len(plot_data)
    if what_for == 'talk':
        width = 5
        height = n_bars/2
        fs1 = 20
        fs2 = 16
    elif what_for == 'paper':
        width = 3
        height = n_bars/2.5
        fs1 = 14
        fs2 = 12

    # set y (parameter) locations and bar widths
    y = np.arange(n_bars)
    bar_width = 0.3

    # sort data numerically
    if num_sort:
        idx = plot_data.argsort()
        plot_data, labels = plot_data[idx], labels[idx]

    # dynamically create subplots if lots of data
    n_groups = 1
    if not n_bars <= 20:
        n_groups = round(n_bars/20)
        y_array = np.array_split(y, n_groups)
        data_array = np.array_split(plot_data, n_groups)
        label_array = np.array_split(labels, n_groups)
        print(f"\nsplitting {suffix} data into {n_groups} subplots")

    if n_groups == 1:

        # create figure
        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        # plot the bars
        single_subplot(ax, y, plot_data, labels)

    else:

        # create figure
        fig, axs = plt.subplots(ncols=n_groups, sharex=True,
            figsize=(width*n_groups, height/n_groups))

        # plot the bars
        for i, ax in enumerate(axs):
            single_subplot(ax, y_array[i], data_array[i], label_array[i])

    # finish tweaking and save figure
    fig.tight_layout()
    plt.savefig(f'barparams_{suffix}.png', bbox_inches='tight')


def tailed_parameters(in_sdf, ffxml, cutoff, tag, tag_smiles, metric_type):
    """
    Extract data from SD tags, identify outlier molecules above cutoff,
    and get the associated force field parameters for each structure.

    Parameters
    ----------
    in_sdf : string
        name of the input SDF molecule file with RMSD or TFD info as SD tags
    ffxml : string
        name of the FFXML force field file
    cutoff : float
        cutoff value to use for the metric; structures with value above
        the cutoff are considered outliers
    tag : string
        name of the SD tag in the SDF file with the metric information
    tag_smiles : string
        name of the SD tag in the SDF file with the molecule identifier
    metric_type : string
        what metric the tag and cutoff refer to (e.g., TFD or RMSD)
        for plot and figure labeling

    Returns
    -------
    data_all : dict
        key of 'count' has int for count of all structures
        key of 'mols_dict' has dict for dictionary of mols (see note)
        key of 'params_mol' has dict of isosmiles and list of parameter IDs
        key of 'params_id' has dict of parameter IDs and list of isosmiles
        key of 'smi_dict' has dict of isosmiles keys and SD tag identifier
    data_out : dict
        same format as data_all but only containing outlier molecules

    Note
    ----
    The mols_dict dictionary is itself a dict of dicts where the first
    level key is the SMILES string (or specified molecular identifier),
    and the value of that key is a dict with the following key/value pairs:
        metric      geometric measurement
        structure   OEGraphMol of the structure

    """

    # load molecules from open reference and query files
    print(f"Opening SDF file {in_sdf}...")
    mols = reader.read_mols(in_sdf)
    print(f"Looking for outlier molecules with {metric_type.upper()} above {cutoff}...\n")

    # find the molecules with the metric above the cutoff
    all_smiles    = []
    mols_all      = OrderedDict()
    mols_out      = OrderedDict()
    count_all     = 0
    count_out     = 0

    for mol in mols:
        for conf in mol.GetConfs():

            smiles = oechem.OEGetSDData(conf, tag_smiles)

            try:
                value  = float(oechem.OEGetSDData(conf, tag))
            except ValueError as e:
                raise ValueError("There was an error while obtaining the SD "
                     f"tag value of '{oechem.OEGetSDData(conf, tag)}'. Did you "
                     f"specify the correct SD tag of '{tag}'?")

            if value >= cutoff:
                mols_out[smiles] = {'metric': value, 'structure': oechem.OEGraphMol(conf)}
                count_out += 1

            mols_all[smiles] = {'structure': oechem.OEGraphMol(conf)}
            all_smiles.append(smiles)
            count_all += 1

    # save outliers molecules to file
    write_mols(mols_out, f'outliers_{metric_type}.mol2')

    # analyze parameters in the outlier and full sets
    params_mol_out, params_id_out, smi_dict_out = get_parameters(mols_out, ffxml)
    params_mol_all, params_id_all, smi_dict_all = get_parameters(mols_all, ffxml)

    # organize all computed data to encompassing dictionary
    # all values in data_* are dictionaries except for data_*['count']
    data_all = {'count': count_all, 'mols_dict': mols_all,
        'params_mol': params_mol_all, 'params_id': params_id_all,
        'smi_dict': smi_dict_all}
    data_out = {'count': count_out, 'mols_dict': mols_out,
        'params_mol': params_mol_out, 'params_id': params_id_out,
        'smi_dict': smi_dict_out}

    # save the params organized by id to pickle
    with open(f'tailed_{metric_type}.pickle', 'wb') as f:
        pickle.dump((data_all, data_out), f)

    return data_all, data_out


def main(in_sdf, ffxml, cutoff, tag, tag_smiles, metric_type, inpickle=None):
    """
    For distributions of RMSD or TFDs of force field geometries with respect to
    reference geometries, identify outlier molecules in the high RMSD (TFD) tails
    above a predetermined cutoff value. For each force field parameter in the
    molecule set, determine the fraction of outlier molecules with that parameter.
    Compare that to the fraction of all molecules that use that parameter.

    Parameters
    ----------
    in_sdf : string
        name of the input SDF molecule file with RMSD or TFD info as SD tags
    ffxml : string
        name of the FFXML force field file
    cutoff : float
        cutoff value to use for the metric; structures with value above
        the cutoff are considered outliers
    tag : string
        name of the SD tag in the SDF file with the metric information
    tag_smiles : string
        name of the SD tag in the SDF file with the molecule identifier
    metric_type : string
        what metric the tag and cutoff refer to (e.g., TFD or RMSD)
        for plot and figure labeling
    inpickle : string
        name of the pickle file with already-computed parameter analysis

    """

    if inpickle is not None and os.path.exists(inpickle):

        # load in analysis from pickle file
        with open(inpickle, 'rb') as f:
            data_all, data_out = pickle.load(f)

    else:
        data_all, data_out = tailed_parameters(
            in_sdf, ffxml, cutoff, tag, tag_smiles, metric_type)

    # count number of unique mols
    params_mol_all = data_all['params_mol']
    params_mol_out = data_out['params_mol']
    uniq_n_all = len(params_mol_all)
    uniq_n_out = len(params_mol_out)

    # count number of unique params
    params_id_all = data_all['params_id']
    params_id_out = data_out['params_id']

    full_params = list(set(params_id_all.keys()))
    full_params.sort(key=natural_keys)

    uniq_p_all = len(full_params)
    uniq_p_out = len(list(set(params_id_out.keys())))

    # print stats on number of outliers
    print(f"\nNumber of structures in full set: {data_all['count']} ({uniq_n_all} unique)")
    print(f"Number of structures in outlier set: {data_out['count']} ({uniq_n_out} unique)")
    print(f"Number of unique parameters in full set: {uniq_p_all}")
    print(f"Number of unique parameters in outlier set: {uniq_p_out}")

    # go through all parameters and find number of molecules which use each one
    nmols_cnt_all, nmols_cnt_out = count_mols_by_param(full_params, params_id_all, params_id_out)
    write_data = np.column_stack((full_params, nmols_cnt_out, nmols_cnt_all))
    with open(f'params_{metric_type}.dat', 'w') as f:
        f.write("# param\tnmols_out\tnmols_all\n")
        f.write(f"NA_total\t{uniq_n_out}\t{uniq_n_all}\n")
        np.savetxt(f, write_data, fmt='%-8s', delimiter='\t')

    # compare fractions in the all set vs the outliers set
    fraction_cnt_all = nmols_cnt_all/uniq_n_all
    fraction_cnt_out = nmols_cnt_out/uniq_n_out

    # exclude parameters for which outliers set AND full set
    # both have only 1 match; do this BEFORE excluding nonzero_inds
    # TODO: this could be made more general. e.g., nmols_cnt_all < nsamples
    ones_nmols_all = np.where(nmols_cnt_all == 1)[0]
    ones_nmols_out = np.where(nmols_cnt_out == 1)[0]
    ones_both = np.intersect1d(ones_nmols_all, ones_nmols_out)
    fraction_cnt_all = np.delete(fraction_cnt_all, ones_both)
    fraction_cnt_out = np.delete(fraction_cnt_out, ones_both)
    full_params = [v for index, v in enumerate(full_params) if index not in ones_both] # exclude ones

    # exclude parameters which are not used in outliers set
    nonzero_inds = np.nonzero(fraction_cnt_out)
    fraction_cnt_all = fraction_cnt_all[nonzero_inds]
    fraction_cnt_out = fraction_cnt_out[nonzero_inds]
    full_params = [full_params[i] for i in nonzero_inds[0]] # keep nonzeroes

    # get ratio of fraction_outliers to fraction_all
    fraction_ratio = fraction_cnt_out / fraction_cnt_all
    max_ratio = np.max(fraction_ratio)

    # plot fraction of molecules which use each parameter
    # separate plot by parameter type
    for t in ['a', 'b', 'i', 'n', 't']:

        # get the subset of data based on parameter type
        plot_inds = [full_params.index(i) for i in full_params if i.startswith(t)]
        subset_data = fraction_ratio[plot_inds]
        subset_label = np.array(full_params)[plot_inds]

        plot_param_bars(subset_data, subset_label, max_ratio,
            suffix=metric_type+f'_{t}',
            num_sort=True,
            what_for='talk')


### ------------------- Parser -------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile", required=True,
            help="Input molecule file")

    parser.add_argument("-f", "--ffxml", required=True,
            help="Open force field ffxml file")

    parser.add_argument("--cutoff", required=True, type=float,
            help="Cutoff value for which to separate outliers")

    parser.add_argument("--tag", required=True,
            help="SDF tag from which to obtain RMSDs or TFDs")

    parser.add_argument("--tag_smiles", required=True,
            help="SDF tag from which to identify conformers")

    parser.add_argument("--metric", default='metric',
        help="Specify 'RMSD' or 'TFD' for which the tag and cutoff value refer")

    parser.add_argument("--inpickle", default=None,
        help="Name of pickle file with already-computed data")


    args = parser.parse_args()
    if args.metric == 'metric':
        print("WARNING: No metric label of 'RMSD' or 'TFD' specified. "
              "Will apply generic label.")
    metric_type = args.metric.lower()

    main(args.infile, args.ffxml,
        args.cutoff, args.tag, args.tag_smiles, metric_type,
        args.inpickle)

