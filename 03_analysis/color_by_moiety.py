#!/usr/bin/env python

"""
color_by_moiety.py

Generate scatter plots of ddE vs. RMSD/TFD in which a subset of data
(e.g., representing a particular chemical moiety) is highlighted.
This script takes in data from the pickle file generated from compare_ffs.py

By:      Victoria T. Lim
Version: Apr 11 2020

Examples:
python color_by_moiety.py -i match.in -p metrics.pickle -s moiety1.dat [moiety2.dat ...] -o scatter_tfd_r4

"""

import os
import re
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import openeye.oechem as oechem
import reader
import seaborn as sns
import pandas as pd

sns.set(rc={'text.usetex' : True})

def draw_scatter_moiety(x_data, y_data, all_x_subset, all_y_subset,
    labels_subset, method_label, x_label, y_label, out_file, what_for='talk',
    x_range=None, y_range=None):

    """
    Draw a scatter plot and color only a subset of points.

    Parameters
    ----------
    x_data : 1D np array
        represents x-axis data for all molecules of a given method
    all_x_subset : list of np arrays
        subsets of x_data with points to highlight in color
    y_data : 1D np array
        should have same shape and correspond to x_data
    all_y_subset : list of np arrays
        subsets of y_data with points to highlight in color
    labels_subset : list of strings
        labels taken from subset file names to label on plot legend
    x_label : string
        name of the x-axis label
    y_label : string
        name of the y-axis label
    out_file : string
        name of the output plot
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"
    x_range : tuple of two floats
        min and max values of x-axis
    y_range : tuple of two floats
        min and max values of y-axis

    """
    print(f"Number of data points in full scatter plot: {len(x_data)}")

    num_methods = len(x_data)

    # set plot limits if specified
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # set log scaling but use symmetric log for negative values
    #plt.yscale('symlog')

    if what_for == 'paper':
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt_options = {'s':10, 'alpha':1.0}
        plt.rc('legend', fontsize=10)

    elif what_for == 'talk':
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt_options = {'s':10, 'alpha':1.0}
        plt.rc('legend', fontsize=14)

    # generate the plot with full set
    plt.scatter(x_data, y_data,
        c='white', edgecolors='grey', **plt_options)

    # generate the plot with subset(s)
    subset_colors = ['#01959f', '#614051', '#e96058']  # todo generalize
    with open('statistics.dat', 'a') as file:
        file.write(f"{method_label} all {len(x_data)} {np.average(x_data[~np.isnan(x_data)])} {np.std(x_data[~np.isnan(x_data)])} {np.average(y_data[~np.isnan(y_data)])} {np.std(y_data[~np.isnan(y_data)])}\n")
        for i, (xs, ys, lab) in enumerate(zip(all_x_subset, all_y_subset, labels_subset)):
            print(f"Number of data points in subset {i}: {len(xs)}")
            print(f"{method_label} {lab} {len(xs)} {np.average(xs)} {np.std(xs)} {np.average(ys)} {np.std(ys)}")
            file.write(f"{method_label} {lab} {len(xs)} {np.average(xs)} {np.std(xs)} {np.average(ys)} {np.std(ys)}\n")
            plt.scatter(xs, ys, label=lab, c=subset_colors[i], zorder=2, **plt_options)

    plt.title(method_label)
    plt.legend(loc=(0.35, 0.02))
    plt.savefig(out_file, bbox_inches='tight')
    plt.clf()
    #plt.show()


def main(in_dict, pickle_file, smi_files, out_prefix):
    """
    For 2+ SDF files that are analogous in terms of molecules and their
    conformers, assess them with respective to a reference SDF file (e.g., QM).
    Metrics include RMSD of conformers, TFD, and relative energy differences.

    Parameter
    ---------
    in_dict : OrderedDict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'
    pickle_file : string
        name of the pickle file from compare_ffs.py
    smi_files : list of strings
        list of filenames with SMILES strings to be colored in plot
    out_prefix : string
        prefix of the names of the output plots

    """
#    method_labels = list(in_dict.keys())
    method_labels =[k[:-2] if (k.endswith('.0') or k.endswith('.1')) else k for k in in_dict.keys()]
    num_methods = len(method_labels)

    # enes_full[i][j][k] = ddE of ith method, jth mol, kth conformer.
    enes_full, rmsds_full, tfds_full, smiles_full = pickle.load(
        open(pickle_file, 'rb'))

    all_smi_subsets = []
    labels_subset = []
    for sf in smi_files:
        labels_subset.append(os.path.splitext(os.path.basename(sf))[0])
        with open(sf) as f:
            smiles_subset = f.readlines()
        smiles_subset = [x.strip() for x in smiles_subset]
        all_smi_subsets.append(smiles_subset)

    x_subset = []
    y_subset = []
    all_inds_subset = [] # list[i][j] has the smiles indices for ith subset

    # flatten list of lists of smiles from pickle file
    smi_flat = [val for sublist in smiles_full[0] for val in sublist]

    # go through each subset file
    for i, smiles_subset in enumerate(all_smi_subsets):
        inds_subset = []

        # go through pickle file smiles
        for j, smi in enumerate(smi_flat):

            # store index if this pickle smiles is in the subset smiles
            if smi in smiles_subset:
                inds_subset.append(j)

        # store and move onto next subset file
        all_inds_subset.append(inds_subset)

    for method in ['GAFF', 'GAFF2', 'MMFF94', 'MMFF94S', 'OPLS3e', 'Smirnoff99Frosst', 'OpenFF-1.0', 'OpenFF-1.1', 'OpenFF-1.2']:
        i = method_labels.index(method) - 1
        # get output filename and make sure it has no forbidden characters
        out_file = out_prefix + method_labels[i+1] + '.png'
        out_file = re.sub(r'[\\/*?:"<>|]', "", out_file)
        print(f"\n{out_file}")

        # assign full data variables
        x_data = np.concatenate(tfds_full[i]).ravel()
        y_data = np.concatenate(enes_full[i]).ravel()

        # take subset(s) of the full data
        all_x_subset = []
        all_y_subset = []
        for j, inds_subset in enumerate(all_inds_subset):
            all_x_subset.append(x_data[inds_subset])
            all_y_subset.append(y_data[inds_subset])

        # print out min/max
        print(f"min/max x: {np.nanmin(x_data):10.4f}\t{np.nanmax(x_data):10.4f}")
        print(f"min/max y: {np.nanmin(y_data):10.4f}\t{np.nanmax(y_data):10.4f}")

        draw_scatter_moiety(
            x_data, y_data,
            all_x_subset, all_y_subset, labels_subset, method_labels[i+1],
            "TFD",
            "ddE (kcal/mol)",
            out_file,
            what_for='paper',
            x_range=(0, 0.8),
            y_range=(-50, 30))
    
        
    data = pd.read_csv('statistics.dat', sep=' ', header=None)
    print(data)
    for i, dat, lab, unit in zip(range(4), ['avg_tfd', 'std_tfd', 'avg_ene', 'std_ene'], ['$\overline{\mathrm{TFD}}$', '$\sigma(\mathrm{TFD})$', '$\overline{\mathrm{dd}E}$', '$\sigma(\mathrm{dd}E)$'], ['', '', ' [kcal/mol]', ' [kcal/mol]']):
        sns.barplot(x=data.iloc[:,0], y=data.iloc[:,i+3], hue=data.iloc[:,1])
        plt.xticks(rotation=90)
        plt.xlabel('')
        plt.ylabel(lab + unit)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(f'{dat}.png', bbox_inches='tight')
        plt.clf()

### ------------------- Parser -------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # parse slice if not analyzing full set
    # https://stackoverflow.com/questions/18632320/numpy-array-indices-via-argparse-how-to-do-it-properly
    def _parse_slice(inslice):
        if inslice == 'all':
            return slice(None)
        try:
            section = int(inslice)
        except ValueError:
            section = [int(s) if s else None for s in inslice.split(':')]
            if len(section) > 3:
                raise ValueError('error parsing input slice')
            section = slice(*section)
        return section

    parser.add_argument("-i", "--infile",
        help="Name of text file with force field in first column and molecule "
             "file in second column. Columns separated by commas.")

    parser.add_argument("-p", "--picklefile",
        help="Pickle file from compare_ffs.py analysis")

    parser.add_argument("-s", "--smifiles", nargs='+',
        help="One or more text files with SMILES from SD tags for "
             "molecules to color in plot. If multiple files, list them "
             "in order from bottommost color to topmost color.")

    parser.add_argument("-o", "--out_prefix",
        help="Prefix of the names of the output plots")

    # parse arguments
    args = parser.parse_args()
    if not os.path.exists(args.infile):
        parser.error(f"Input file {args.infile} does not exist.")

    # suppress the following repeated warning
    # Warning: Using automorph=true and heavyOnly=false in OERMSD.
    # Warning: In some cases, this can lead to long runtimes due to a large number of automorph matches.
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)

    # read main input file and check that files within exist
    in_dict = reader.read_check_input(args.infile)

    # run main
    print("Log file from color_by_moiety.py\n")
    with open('statistics.dat', 'w') as file:
        file.write('')
    main(in_dict, args.picklefile, args.smifiles, args.out_prefix)

