#!/usr/bin/env python

"""
compare_ffs.py

For 2+ SDF files that are analogous in terms of molecules and their conformers,
assess them (e.g., having FF geometries) with respective to a reference SDF
file (e.g., having QM geometries). Metrics include: RMSD of conformers, TFD
(another geometric evaluation), and relative energy differences.

By:      Victoria T. Lim
Version: Jan 10 2020

Examples:
python compare_ffs.py -i match.in -t 'SMILES QCArchive' --plot
python compare_ffs.py -i match.in -t 'SMILES QCArchive' --plot --molslice 25 26 3:5 6::3

"""

import os
import numpy as np
from scipy.interpolate import interpn
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import openeye.oechem as oechem
import rdkit.Chem as Chem
from rdkit.Chem import TorsionFingerprints
import reader
from collections import OrderedDict
### ------------------- Functions -------------------


def calc_tfd(ref_mol, query_mol, conf_id_tag):
    """
    Calculate Torsion Fingerprint Deviation between two molecular structures.
    RDKit is required for TFD calculation.

    References
    ----------
    Modified from the following code:
    https://github.com/MobleyLab/off-ffcompare

    TFD reference:
    https://pubs.acs.org/doi/10.1021/ci2002318

    Parameters
    ----------
    ref_mol : OEMol
    query_mol : OEMol
    conf_id_tag : string
        label of the SD tag that should be the same for matching conformers
        in different files

    Returns
    -------
    tfd : float
        Torsion Fingerprint Deviation between ref and query molecules

    """
    # convert refmol to one readable by RDKit
    ref_rdmol = reader.rdmol_from_oemol(ref_mol)

    # convert querymol to one readable by RDKit
    que_rdmol = reader.rdmol_from_oemol(query_mol)

    # check if the molecules are the same
    # tfd requires the two molecules must be instances of the same molecule
    rsmiles = Chem.MolToSmiles(ref_rdmol)
    qsmiles = Chem.MolToSmiles(que_rdmol)
    if rsmiles != qsmiles:
        print(f"- WARNING: The reference mol \'{ref_mol.GetTitle()}\' and "
                 f"query mol \'{query_mol.GetTitle()}\' do NOT have the same "
                 "SMILES strings as determined by RDKit MolToSmiles. It is "
                 "possible that they did not have matching SMILES even before "
                 "conversion from OEMol to RDKit mol. Listing in order the "
                 "QCArchive SMILES string, RDKit SMILES for ref mol, and "
                 "RDKit SMILES for query mol:"
                 f"\n {oechem.OEGetSDData(ref_mol, conf_id_tag)}"
                 f"\n {rsmiles}\n {qsmiles}")
        tfd = np.nan

    # calculate the TFD
    else:
        try:
            tfd = TorsionFingerprints.GetTFDBetweenMolecules(ref_rdmol, que_rdmol)
        # triggered for molecules such as urea
        except IndexError:
            print(f"- Error calculating TFD on molecule '{ref_mol.GetTitle()}'."
                  " Possibly no non-terminal rotatable bonds found.")
            tfd = np.nan

    return tfd


def compare_ffs(in_dict, conf_id_tag, out_prefix, keep_ref_conf=False, mol_slice=None):
    """
    For 2+ SDF files that are analogous in terms of molecules and their
    conformers, assess them by RMSD, TFD, and relative energy differences.

    Parameters
    ----------
    in_dict : OrderedDict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'
    conf_id_tag : string
        label of the SD tag that should be the same for matching conformers
        in different files
    out_prefix : string
        prefix appended to sdf file name to write out new SDF file
        with RMSD and TFD info added as SD tags
    keep_ref_conf : Boolean
        True to keep reference conformer energy for each molecule
        False to remove reference conformer energy;
        note that qm ref conf defines where dE=0 for a certain molecule
        so if qm ref conf is same as ff ref conf, ddE=0 may be inflated;
        qm ref conf may or may not be the same as ff ref conf;
        ref conf data also removed for RMSD/TFD data (for scatter plots)
    mol_slice : numpy slice object
        The resulting integers are numerically sorted and duplicates removed.
        e.g., slices = np.s_[0, 3:5, 6::3] would be parsed to return
        [0, 3, 4, 6, 9, 12, 15, 18, ...]
        Can also parse from end: [-3:] gets the last 3 molecules, and
        [-2:-1] is the same as [-2] to get just next to last molecule.

    Returns
    -------
    enes_full : 3D list
        enes_full[i][j][k] = ddE of ith method, jth mol, kth conformer.
        ddE = (dE of query method) - (dE of ref method),
        where the dE is computed as conformer M - conformer N,
        and conformer N is chosen from the lowest energy of the ref confs.
        the reference method is not present; i.e., self-comparison is skipped,
        so the max i value represents total number of files minus one.
    rmsds_full : 3D list
        same format as that of enes_full but with conformer RMSDs
    tfds_full : 3D list
        same format as that of enes_full but with conformer TFDs
    smiles_full : 3D list
        same format as that of enes_full but with conformer SMILES strings

    """
    # set RMSD calculation parameters
    automorph = True   # take into acct symmetry related transformations
    heavyOnly = False  # do consider hydrogen atoms for automorphisms
    overlay = True     # find the lowest possible RMSD

    # initiate final data lists
    enes_full = []
    rmsds_full = []
    tfds_full = []
    smiles_full = []

    # get first filename representing the reference geometries
    sdf_ref = list(in_dict.values())[0]['sdfile']
    tag_ref = list(in_dict.values())[0]['sdtag']

    # assess each file against reference
    for ff_label, ff_dict in in_dict.items():

        # get details of queried file
        sdf_que = ff_dict['sdfile']
        tag_que = ff_dict['sdtag']

        if sdf_que == sdf_ref:
            continue

        # initiate new sublists
        enes_method = []
        rmsds_method = []
        tfds_method = []
        smiles_method = []

        # open an output file to store query molecules with new SD tags
        out_file = f'{out_prefix}_{os.path.basename(sdf_que)}'
        ofs = oechem.oemolostream()
        if not ofs.open(out_file):
            oechem.OEThrow.Fatal(f"Unable to open {out_file} for writing")

        # load molecules from open reference and query files
        print(f"\n\nOpening reference file {sdf_ref}")
        mols_ref = reader.read_mols(sdf_ref, mol_slice)

        print(f"Opening query file {sdf_que} for [ {ff_label} ] energies")
        mols_que = reader.read_mols(sdf_que, mol_slice)

        # loop over each molecule in reference and query files
        for rmol, qmol in zip(mols_ref, mols_que):

            # initial check that they have same title and number of confs
            rmol_name = rmol.GetTitle()
            rmol_nconfs = rmol.NumConfs()
            if (rmol_name != qmol.GetTitle()) or (rmol_nconfs != qmol.NumConfs()):
                raise ValueError("ERROR: Molecules not aligned in iteration. "
                                "Offending molecules and number of conformers:\n"
                                f"\'{rmol_name}\': {rmol_nconfs} nconfs\n"
                                f"\'{qmol.GetTitle()}\': {qmol.NumConfs()} nconfs")

            # initialize lists to store conformer energies
            enes_ref = []
            enes_que = []
            rmsds_mol = []
            tfds_mol = []
            smiles_mol = []

            # loop over each conformer of this mol
            for ref_conf, que_conf in zip(rmol.GetConfs(), qmol.GetConfs()):

                # check confomer match from the specified tag
                ref_id = oechem.OEGetSDData(ref_conf, conf_id_tag)
                que_id = oechem.OEGetSDData(que_conf, conf_id_tag)
                if ref_id != que_id:
                    raise ValueError("ERROR: Conformers not aligned in iteration"
                                    f" for mol: '{rmol_name}'. The conformer "
                                    f"IDs ({conf_id_tag}) for ref and query are:"
                                    f"\n{ref_id}\n{que_id}.")

                # note the smiles id
                smiles_mol.append(ref_id)

                # get energies
                enes_ref.append(float(oechem.OEGetSDData(ref_conf, tag_ref)))
                enes_que.append(float(oechem.OEGetSDData(que_conf, tag_que)))

                # compute RMSD between reference and query conformers
                rmsd = oechem.OERMSD(ref_conf, que_conf, automorph,
                                     heavyOnly, overlay)
                rmsds_mol.append(rmsd)

                # compute TFD between reference and query conformers
                tfd = calc_tfd(ref_conf, que_conf, conf_id_tag)
                tfds_mol.append(tfd)

                # store data in SD tags for query conf, and write conf to file
                oechem.OEAddSDData(que_conf, f'RMSD to {sdf_ref}', str(rmsd))
                oechem.OEAddSDData(que_conf, f'TFD to {sdf_ref}', str(tfd))
                oechem.OEWriteConstMolecule(ofs, que_conf)

            # compute relative energies against lowest E reference conformer
            lowest_ref_idx = enes_ref.index(min(enes_ref))
            rel_enes_ref = np.array(enes_ref) - enes_ref[lowest_ref_idx]
            rel_enes_que = np.array(enes_que) - enes_que[lowest_ref_idx]

            # remove the reference conformer of dE = 0
            if not keep_ref_conf:
                rel_enes_ref = np.delete(rel_enes_ref, lowest_ref_idx)
                rel_enes_que = np.delete(rel_enes_que, lowest_ref_idx)
                rmsds_mol.pop(lowest_ref_idx)
                tfds_mol.pop(lowest_ref_idx)
                smiles_mol.pop(lowest_ref_idx)

            # subtract them to get ddE = dE (query method) - dE (ref method)
            enes_mol = np.array(rel_enes_que) - np.array(rel_enes_ref)

            # store then move on
            enes_method.append(enes_mol)
            rmsds_method.append(np.array(rmsds_mol))
            tfds_method.append(np.array(tfds_mol))
            smiles_method.append(smiles_mol)
            #print(rmsds_method, len(rmsds_method))
            #print(enes_method, len(enes_method))

        enes_full.append(enes_method)
        rmsds_full.append(rmsds_method)
        tfds_full.append(tfds_method)
        smiles_full.append(smiles_method)

    ofs.close()

    return enes_full, rmsds_full, tfds_full, smiles_full


def flatten(list_of_lists):
    """
    Flatten one level of nesting.

    Parameter
    ---------
    list_of_lists

    Returns
    -------
    1D numpy array

    """
    return np.concatenate(list_of_lists).ravel()


def draw_scatter(x_data, y_data, method_labels, x_label, y_label, out_file, what_for='talk'):
    """
    Draw scatter plot, such as of (ddE vs RMSD) or (ddE vs TFD).

    Parameters
    ----------
    x_data : list of lists
        x_data[i][j] represents ith method and jth molecular structure
    y_data : list of lists
        should have same shape and correspond to x_data
    method_labels : list
        list of all the method names including reference method first
    x_label : string
        name of the x-axis label
    y_label : string
        name of the y-axis label
    out_file : string
        name of the output file
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"

    """
    print(f"\nNumber of data points in full scatter plot: {len(flatten(x_data))}")
    markers = ["o", "^", "d", "x", "s", "p", "P", "3", ">"]

    num_methods = len(x_data)
    plist = []
    for i in range(num_methods):
        p = plt.scatter(x_data[i], y_data[i], marker=markers[i],
            label=method_labels[i+1], alpha=0.6)
        plist.append(p)

    if what_for == 'paper':
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
        plt.subplots_adjust(left=0.16, right=.72,top=0.9, bottom=0.2)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc=(1.04,0.4), fontsize=10)
        # make the marker size smaller
        for p in plist:
            p.set_sizes([8.0])

    elif what_for == 'talk':
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc=(1.04,0.5), fontsize=12)
        # make the marker size smaller
        for p in plist:
            p.set_sizes([4.0])

    # set log scaling but use symmetric log for negative values
#    plt.yscale('symlog')

    plt.savefig(out_file, bbox_inches='tight')
    plt.clf()
    #plt.show()

def draw_corr(x_data, y_data, method_labels, x_label, y_label, out_file, what_for='talk'):
    """
    Draw scatter plot, such as of (ddE vs RMSD) or (ddE vs TFD).

    Parameters
    ----------
    x_data : list of lists
        x_data[i][j] represents ith method and jth molecular structure
    y_data : list of lists
        should have same shape and correspond to x_data
    method_labels : list
        list of all the method names including reference method first
    x_label : string
        name of the x-axis label
    y_label : string
        name of the y-axis label
    out_file : string
        name of the output file
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"

    """
    print(f"\nNumber of data points in full scatter plot: {len(flatten(x_data))}")
    markers = ["o", "^", "d", "x", "s", "p", "P", "3", ">"]

    num_methods = len(x_data)
    plist = []
    for i in range(num_methods):
        p = plt.scatter(x_data[i], y_data[i], marker=markers[i],
            label=method_labels[i+1], alpha=0.6)
        plist.append(p)

    if what_for == 'paper':
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.subplots_adjust(left=0.16, right=.72,top=0.9, bottom=0.2)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc=(1.04,0.4), fontsize=10)
        # make the marker size smaller
        for p in plist:
            p.set_sizes([8.0])

    elif what_for == 'talk':
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc=(1.04,0.5), fontsize=12)
        # make the marker size smaller
        for p in plist:
            p.set_sizes([4.0])

    plt.savefig(out_file, bbox_inches='tight')
    plt.clf()
    #plt.show()


def draw_ridgeplot(mydata, method_labels, x_label, out_file, what_for='paper',
                   bw='scott', same_subplot=False, sym_log=False, hist_range=(-15,15)):

    """
    Draw ridge plot of data (to which kernel density estimate is applied)
    segregated by each method (representing a different color/level).

    Modified from the following code:
    https://seaborn.pydata.org/examples/kde_ridgeplot.html

    Parameters
    ----------
    mydata : list of lists
        mydata[i][j] represents ith method and jth molecular structure
    method_labels : list
        list of all the method names including reference method first
    x_label : string
        name of the x-axis label
        also used for pandas dataframe column name
    out_file : string
        name of the output file
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"
    bw : string or float
        defines bandwidth for KDE as called in seaborn.kdeplot, OR don't use
        kde at all and just histogram the data;
        options: 'scott' (KDE), 'silverman' (KDE), scalar value (KDE), 'hist'
    same_subplot : Boolean
        False is default to have separate and slightly overlapping plots,
        True to plot all of them showing on the same subplot (no fill)
    sym_log : Boolean
        False is default to plot density estimate as is,
        True to plot x-axis on symmetric log scale
    hist_range : tuple
        tuple of min and max values to use for histogram;
        only needed if bw is set to 'hist'

    """
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()

        # set axis font size
        if what_for == 'paper': fs = 14
        elif what_for == 'talk': fs = 14

        ax.text(0, .2, label, fontweight="bold", color=color, fontsize=fs,
                ha="left", va="center", transform=ax.transAxes)

    if what_for == 'paper':
        ridgedict = {
            "h":0.9,
            "lw":2.0,
            "vl":1.0,
            "xfontsize":14,
        }
    elif what_for == 'talk':
        ridgedict = {
            "h":2.0,
            "lw":3.0,
            "vl":1.0,
            "xfontsize":16,
        }

    num_methods = len(mydata)

    # Initialize the FacetGrid object
    my_cmap = "tab10"
    pal = sns.palplot(sns.color_palette(my_cmap))
    colors = sns.color_palette(my_cmap)
    all_labels = ['MMFF94S', 'GAFF2', 'OPLS3e', 'OpenFF-1.2', 'OpenFF-1.0', 'Smirnoff99Frosst', 'MMFF94', 'GAFF',  'B3LYP-D3BJ/DZVP', 'OpenFF-1.1']
    cdict = {m: c for m, c in zip(all_labels, colors)}

    # convert data to dataframes for ridge plot
    temp = []
    for method in ['GAFF', 'GAFF2', 'MMFF94', 'MMFF94S', 'OPLS3e', 'Smirnoff99Frosst', 'OpenFF-1.0', 'OpenFF-1.1', 'OpenFF-1.2']:
#range(num_methods):
        if method in method_labels:
            index = method_labels.index(method) - 1
            print(method, index)
            df = pd.DataFrame(mydata[index], columns = [x_label])
            df['method'] = method_labels[index+1]
            temp.append(df)

    print(temp)
    # list of dataframes concatenated to single dataframe
    df = pd.concat(temp, ignore_index = True)
    print(df)
#    print(method_labels)
    g = sns.FacetGrid(df, row="method", hue="method", aspect=10,
        height=ridgedict["h"], palette=cdict)

    if not same_subplot:

        # draw filled-in densities
        if bw=='hist':
            histoptions = {"histtype":"bar", "alpha":0.6, "linewidth":ridgedict["lw"],
                "range":hist_range, "align":"mid"}
            g.map(sns.distplot, x_label, hist=True, kde=False, bins=15, norm_hist=True, hist_kws=histoptions)
        else:
            g.map(sns.kdeplot, x_label, clip_on=False, shade=True, alpha=0.5,
                lw=ridgedict["lw"], bw=bw)

        # draw colored horizontal line below densities
        g.map(plt.axhline, y=0, lw=ridgedict["lw"], clip_on=False)

    else:

        # draw black horizontal line below densities
        plt.axhline(y=0, color='black')

    # draw outline around densities; can also single outline color: color="k"
    if bw=='hist':
        histoptions = {"histtype":"step", "alpha":1.0, "linewidth":ridgedict["lw"],
            "range":hist_range, "align":"mid"}
        g.map(sns.distplot, x_label, hist=True, kde=False, bins=15, norm_hist=True, hist_kws=histoptions)


    else:
        g.map(sns.kdeplot, x_label, clip_on=False, lw=ridgedict["lw"], bw=bw)

    # draw a vertical line at x=0 for visual reference
    g.map(plt.axvline, x=0, lw=ridgedict["vl"], ls='--', color='gray', clip_on=False)

    # optional: add custom vertical line
    #g.map(plt.axvline, x=0.12, lw=1, ls='--', color='gray', clip_on=False)

    # add labels to each level
    if not same_subplot:
        g.map(label, x_label)

    # else if single subplot, generate a custom legend
    else:
        cmap = mpl.cm.tab10
        patches = []
        n_ffs = len(method_labels)-1
        #for i in range(n_ffs):
        for method in ['GAFF', 'GAFF2', 'MMFF94', 'MMFF94S', 'OPLS3e', 'Smirnoff99Frosst', 'OpenFF-1.0', 'OpenFF-1.1', 'OpenFF-1.2']:
            if method in method_labels:
                index = method_labels.index(method) - 1
                patches.append(mpl.patches.Patch(color=cdict[method_labels[index+1]],
                label=method_labels[index+1]))
        plt.legend(handles=patches, fontsize=ridgedict["xfontsize"]/1.2)

    # optional: set symmetric log scale on x-axis
    if sym_log:
        g.set(xscale = "symlog")

    # Set the subplots to overlap
    if not same_subplot:
        g.fig.subplots_adjust(hspace=-0.45)
    else:
        g.fig.subplots_adjust(hspace=-1.0)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
#    g.set(yticks=[])
    g.despine(bottom=True)#, left=True)
    # ax = plt.gca()
    # ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_position('zero')
    # ax.set_yticks([0.4])
    if what_for == 'paper':
        plt.gcf().set_size_inches(7, 3)
    elif what_for == 'talk':
        plt.gcf().set_size_inches(12, 9)

    # adjust font sizes
    plt.xlabel(x_label, fontsize=ridgedict["xfontsize"])
    plt.ylabel('Density', fontsize=ridgedict["xfontsize"])
    plt.xticks(fontsize=ridgedict["xfontsize"])


    # save with transparency for overlapping plots
    plt.savefig(out_file, transparent=True, bbox_inches='tight')
    plt.clf()
    #plt.show()


def draw_density2d(x_data, y_data, title, x_label, y_label, out_file, what_for='talk',
                   bins=20, x_range=None, y_range=None, z_range=None, z_interp=True, symlog=False):

    """
    Draw a scatter plot colored smoothly to represent the 2D density.
    Based on: https://stackoverflow.com/a/53865762/8397754

    Parameters
    ----------
    x_data : 1D list
        represents x-axis data for all molecules of a given method
    y_data : 1D list
        should have same shape and correspond to x_data
    title : string
        title of the plot
    x_label : string
        name of the x-axis label
    y_label : string
        name of the y-axis label
    out_file : string
        name of the output file
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"
    bins : int
        number of bins for np.histogram2d
    x_range : tuple of two floats
        min and max values of x-axis
    y_range : tuple of two floats
        min and max values of y-axis
    z_range : tuple of two floats
        min and max values of density for setting a uniform color bar;
        these should be at or beyond the bounds of the min and max
    z_interp : Boolean
        True to smoothen the color scale for the scatter plot points;
        False to plot 2d histograms colored by cells (no scatter plot)

    """

    def colorbar_and_finish(labelsize, fname):
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=labelsize)
        cb.ax.set_title('counts', size=labelsize)

        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
        #plt.show()

    fig = plt.gcf()
    if what_for == 'paper':
        ms = 1
        size1 = 10
        size2 = 10
        fig.set_size_inches(4, 3)
    elif what_for == 'talk':
        ms = 4
        size1 = 14
        size2 = 16
        fig.set_size_inches(9, 6)
    plt_options = {'s':ms, 'cmap':'coolwarm_r'}

    # label and adjust plot
    plt.title(title, fontsize=size2)
    plt.xlabel(x_label, fontsize=size2)
    plt.ylabel(y_label, fontsize=size2)
    plt.xticks(fontsize=size1)
    plt.yticks(fontsize=size1)

    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])

    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # remove any nans from x_data, such as TFD score for urea-like mols
    nan_inds = np.isnan(x_data)
    x_data = x_data[~nan_inds]
    y_data = y_data[~nan_inds]
    print(f"\nNumber of data points in FF scatter plot: {len(x_data)}")

    # compute histogram in 2d
    data, x_e, y_e = np.histogram2d(x_data, y_data, bins=bins)

    # plot colored 2d histogram if z_interp not specified
    if not z_interp:
        extent = [x_e[0], x_e[-1], y_e[0], y_e[-1]]
        plt.imshow(data.T, extent=extent, origin='lower', aspect='auto',
            cmap=plt_options['cmap'], vmin=z_range[0], vmax=z_range[1])

        colorbar_and_finish(size1, out_file)
        return

    # smooth/interpolate data
    z = interpn( ( 0.5*(x_e[1:]+x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ), data,
        np.vstack([x_data, y_data]).T, method="splinef2d", bounds_error=False)

    # sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x_data[idx], y_data[idx], z[idx]

    print(f"{title} ranges of data in density plot:\n\t\tmin\t\tmax"
          f"\nx\t{np.min(x):10.4f}\t{np.max(x):10.4f}"
          f"\ny\t{np.min(y):10.4f}\t{np.max(y):10.4f}"
          f"\nz\t{np.min(data):10.4f}\t{np.max(data):10.4f} (histogrammed)"
          f"\nz'\t{np.nanmin(z):10.4f}\t{np.nanmax(z):10.4f} (interpolated)")

    # add dummy points of user-specified min/max z for uniform color scaling
    # similar to using vmin/vmax in pyplot pcolor
    x = np.append(x, [-1, -1])
    y = np.append(y, [0, 0])
    z = np.append(z, [z_range[0], z_range[1]])

    print(f"z''\t{np.nanmin(z):10.4f}\t{np.nanmax(z):10.4f} (interp, bounded)")

    # generate the plot
    plt.scatter(x, y, c=z, vmin=z_range[0], vmax=z_range[1], **plt_options)

    # set log scaling but use symmetric log for negative values
    if symlog:
        plt.yscale('symlog')

    # configure color bar and finish plotting
    colorbar_and_finish(size1, out_file)


def main(in_dict, read_pickle, conf_id_tag, plot=False, mol_slice=None):
    """
    For 2+ SDF files that are analogous in terms of molecules and their
    conformers, assess them with respective to a reference SDF file (e.g., QM).
    Metrics include RMSD of conformers, TFD, and relative energy differences.

    Parameter
    ---------
    in_dict : Orderedict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'
    read_pickle : Boolean
        read in data from metrics.pickle
    conf_id_tag : string
        label of the SD tag that should be the same for matching conformers
        in different files
    plot : Boolean
        generate line plots of conformer energies
    mol_slice : numpy slice object
        The resulting integers are numerically sorted and duplicates removed.
        e.g., slices = np.s_[0, 3:5, 6::3] would be parsed to return
        [0, 3, 4, 6, 9, 12, 15, 18, ...]
        Can also parse from end: [-3:] gets the last 3 molecules, and
        [-2:-1] is the same as [-2] to get just next to last molecule.

    """
    print(in_dict)
    # remove last digit of OpenFF version
    in_dict = OrderedDict([(k[:-2], v) if (k.endswith('.0') or k.endswith('.1')) else (k, v) for k, v in in_dict.items()])
    print(in_dict)
    method_labels = list(in_dict.keys())

    # run comparison, unless reading in from pickle file
    if read_pickle:
        enes_full, rmsds_full, tfds_full, smiles_full = pickle.load(
            open('metrics.pickle', 'rb'))
    else:
        # enes_full[i][j][k] = ddE of ith method, jth mol, kth conformer.
        enes_full, rmsds_full, tfds_full, smiles_full = compare_ffs(
                                                            in_dict,
                                                            conf_id_tag,
                                                            'refdata',
                                                            False,
                                                            mol_slice)

        # save results in pickle file
        pickle.dump((enes_full, rmsds_full, tfds_full, smiles_full),
            open('metrics.pickle', 'wb'))

        # write enes_full to file since not so easy to save as SD tag
        with open('ddE.dat', 'w') as outfile:
            outfile.write("# Relative energies (kcal/mol) of ddE = dE (query method) - dE (ref method)\n")
            outfile.write("# Each dE is the current conformer's energy minus the lowest energy conformer of the same molecule\n")
            outfile.write("# ==================================================\n")

            for i, (ddE_slice, smi_slice) in enumerate(zip(enes_full, smiles_full)):
                outfile.write(f"# Relative energies for FF {method_labels[i+1]}\n")

                # flatten the mol/conformer array
                flat_enes =   np.array([item for sublist in ddE_slice for item in sublist])
                flat_smiles = np.array([item for sublist in smi_slice for item in sublist])

                # combine label and data, then write to file
                smiles_and_enes = np.column_stack((flat_smiles, flat_enes))
                np.savetxt(outfile, smiles_and_enes, fmt='%-60s', delimiter='\t')

    energies = []
    rmsds = []
    tfds = []

    # flatten all confs/molecules into same list but keep methods distinct
    # energies and rmsds are now 2d lists
    enes_full=np.array(enes_full)
    print(enes_full)
#    print(np.isclose(enes_full, np.zeros_like(enes_full)))
#    print(np.sum(np.argwhere(np.array(enes_full)==0.)))
    print(np.array(enes_full).shape)
        
    for a, b, c in zip(enes_full, rmsds_full, tfds_full):
        print(np.array(a).shape)
        isclose = np.isclose(flatten(a), np.zeros_like(flatten(a)), rtol=1e-23)
        print(np.sum(isclose))
        energies.append(flatten(a))
        rmsds.append(flatten(b))
        tfds.append(flatten(c))

    
    # 95 % percentiles
    print("95 % percentiles of energies")
    for m, enes in zip(method_labels[1:], energies):
        print(m, np.percentile(enes, 2.5), np.percentile(enes, 97.5))
    print("Fraction of ddE between -1 <= ddE <= 1 kcal/mol")
    for m, enes in zip(method_labels[1:], energies):
        samples = []
        for i in range(1000):
            s = np.random.choice(enes, replace=True, size=len(enes))
            samples.append(np.sum(np.logical_and(s >= -1.0, s <= 1.0))/float(len(s)))
        print(m, np.sum(np.logical_and(enes >= -1.0, enes <= 1.0))/float(len(enes)), np.std(samples))
    if plot:
        for i in range(len(energies)):
            for j in range(i+1, len(energies)):
                draw_corr(
                    [rmsds[i]],
                    [rmsds[j]],
                    [None, f'{method_labels[j+1]} vs. {method_labels[i+1]}'],
                    f"RMSD {method_labels[i+1]}"+"($\mathrm{\AA}$)",
                    f"RMSD {method_labels[j+1]}"+"($\mathrm{\AA}$)",
                    f"fig_scatter_rmsd_{method_labels[j+1]}_{method_labels[i+1]}.png".replace("/", ""),
                    "paper")
                draw_corr(
                    [tfds[i]],
                    [tfds[j]],
                    [None, f'{method_labels[j+1]} vs. {method_labels[i+1]}'],
                    f"TFD {method_labels[i+1]}",
                    f"TFD {method_labels[j+1]}",
                    f"fig_scatter_tfd_{method_labels[j+1]}_{method_labels[i+1]}.png".replace("/", ""),
                    "paper")
                draw_corr(
                    [energies[i]],
                    [energies[j]],
                    [None, f'{method_labels[j+1]} vs. {method_labels[i+1]}'],
                    f"ddE {method_labels[i+1]} (kcal/mol)",
                    f"ddE {method_labels[j+1]} (kcal/mol)",
                    f"fig_scatter_energies_{method_labels[j+1]}_{method_labels[i+1]}.png".replace("/", ""),
                    "paper")
        draw_scatter(
            rmsds,
            energies,
            method_labels,
            "RMSD ($\mathrm{\AA}$)",
            "ddE (kcal/mol)",
            "fig_scatter_rmsd.png",
            "paper")
        draw_scatter(
            tfds,
            energies,
            method_labels,
            "TFD",
            "ddE (kcal/mol)",
            "fig_scatter_tfd.png",
            "paper")
        draw_ridgeplot(
            energies,
            method_labels,
            "ddE (kcal/mol)",
            "fig_ridge_dde.png",
            "paper",
            bw='hist',
            same_subplot=True,
            sym_log=False,
            hist_range=(-15,15))
        draw_ridgeplot(
            rmsds,
            method_labels,
            "RMSD ($\mathrm{\AA}$)",
            "fig_ridge_rmsd.png",
            "paper",
            #bw='scott',
            bw='hist', hist_range=(0,3),
            same_subplot=True,
            sym_log=False)
        draw_ridgeplot(
            tfds,
            method_labels,
            "TFD",
            "fig_ridge_tfd.png",
            "paper",
            #bw='scott',
            bw='hist', hist_range=(0,.5),
            same_subplot=True,
            sym_log=False)

        for i, ml in enumerate(method_labels[1:]):
            draw_density2d(
                rmsds[i],
                energies[i],
                ml,
                "RMSD ($\mathrm{\AA}$)",
                "ddE (kcal/mol)",
                f"fig_density_rmsd_linear_{ml}.png",
                "paper",
                x_range=(0, 3.7),
                y_range=(-30, 30),
                z_range=(-260, 5200),
                z_interp=True,
                symlog=False)

            draw_density2d(
                tfds[i],
                energies[i],
                ml,
                "TFD",
                "ddE (kcal/mol)",
                f"fig_density_tfd_linear_{ml}.png",
                "paper",
                x_range=(0, .8),
                y_range=(-30, 30),
                z_range=(-302, 7060),
                z_interp=True,
                symlog=False)

            draw_density2d(
                rmsds[i],
                energies[i],
                ml,
                "RMSD ($\mathrm{\AA}$)",
                "ddE (kcal/mol)",
                f"fig_density_rmsd_{ml}.png",
                "paper",
                x_range=(0, 3.7),
                y_range=(-30, 30),
                z_range=(-260, 5200),
                z_interp=True,
                symlog=True)

            draw_density2d(
                tfds[i],
                energies[i],
                ml,
                "TFD",
                "ddE (kcal/mol)",
                f"fig_density_tfd_{ml}.png",
                "paper",
                x_range=(0, .8),
                y_range=(-30, 30),
                z_range=(-302, 7060),
                z_interp=True,
                symlog=True)





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

    parser.add_argument("--readpickle", action="store_true", default=False,
        help="Read in already-computed data from pickle file named \"metrics.pickle\"")

    parser.add_argument("-t", "--conftag",
        help="Name of the SD tag that distinguishes conformers. Within the "
             "same SDF file, no other conformer should have the same tag value."
             " Between two SDF files, the matching conformers should have the "
             "same SD tag and value.")

    parser.add_argument("--plot", action="store_true", default=False,
        help="Generate line plots for every molecule with the conformer "
             "relative energies.")

    parser.add_argument("--molslice", nargs='+', type=_parse_slice, default=None,
        help="Only analyze the selected molecules of the given slices. Slices "
             "are integer-sorted upon processing, and duplicates removed. Internal"
             " code can slice negative values like -2:-1 but argparse cannot."
             "Example: --molslice 25 26 3:5 6::3")

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
    print("Log file from compare_ffs.py")
    main(in_dict, args.readpickle, args.conftag, args.plot, args.molslice)

