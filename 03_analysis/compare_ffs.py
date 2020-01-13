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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import openeye.oechem as oechem
import rdkit.Chem as Chem
from rdkit.Chem import TorsionFingerprints
import reader

### ------------------- Functions -------------------


def calc_tfd(ref_mol, query_mol):
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

    Returns
    -------
    tfd : float
        Torsion Fingerprint Deviation between ref and query molecules

    """
    # convert refmol to one readable by RDKit
    ref_rdmol = reader.rdmol_from_oemol(ref_mol)

    # convert querymol to one readable by RDKit
    que_rdmol = reader.rdmol_from_oemol(query_mol)

    # check if there was a mistake in the conversion process
    rsmiles = Chem.MolToSmiles(ref_rdmol)
    qsmiles = Chem.MolToSmiles(que_rdmol)
    if rsmiles != qsmiles:
        raise ValueError("ERROR: SMILES strings no longer match after "
                 f"conversion of offending molecules: \'{ref_mol.GetTitle()}\'"
                 f" and \'{query_mol.GetTitle()}\'\n"
                 f"Their SMILES strings are:\n{rsmiles}\n{qsmiles}")

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


def compare_ffs(in_dict, conf_id_tag, out_prefix, mol_slice=None):
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
                tfd = calc_tfd(ref_conf, que_conf)
                tfds_mol.append(tfd)

                # store data in SD tags for query conf, and write conf to file
                oechem.OEAddSDData(que_conf, f'RMSD to {sdf_ref}', str(rmsd))
                oechem.OEAddSDData(que_conf, f'TFD to {sdf_ref}', str(tfd))
                oechem.OEWriteConstMolecule(ofs, que_conf)

            # compute relative energies against lowest E reference conformer
            lowest_ref_idx = enes_ref.index(min(enes_ref))
            rel_enes_ref = np.array(enes_ref) - enes_ref[lowest_ref_idx]
            rel_enes_que = np.array(enes_que) - enes_que[lowest_ref_idx]

            # subtract them to get ddE = dE (ref method) - dE (query method)
            enes_mol = np.array(rel_enes_ref) - np.array(rel_enes_que)

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
    print(f"\nNumber of data points in scatter plot: {len(flatten(x_data))}")
    markers = ["o", "^", "d", "x", "s", "p"]

    num_methods = len(x_data)
    plist = []
    for i in range(num_methods):
        p = plt.scatter(x_data[i], y_data[i], marker=markers[i],
            label=method_labels[i+1], alpha=0.6)
        plist.append(p)

    if what_for == 'paper':
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc=(1.04,0.5), fontsize=8)
        # make the marker size smaller
        for p in plist:
            p.set_sizes([12.0])

    elif what_for == 'talk':
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc=(1.04,0.5), fontsize=12)

    plt.savefig(out_file, bbox_inches='tight')
    plt.clf()
    #plt.show()


def draw_ridgeplot(mydata, datalabel, method_labels, out_file, what_for='talk'):
    """
    Draw ridge plot of data (to which kernel density estimate is applied)
    segregated by each method (representing a different color/level).

    Modified from the following code:
    https://seaborn.pydata.org/examples/kde_ridgeplot.html

    Parameters
    ----------
    mydata : list of lists
        mydata[i][j] represents ith method and jth molecular structure
    datalabel : string
        name of the x-axis label
        also used for pandas dataframe column name
    method_labels : list
        list of all the method names including reference method first
    out_file : string
        name of the output file
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"

    """
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()

        # set axis font size
        if what_for == 'paper': fs = 8
        elif what_for == 'talk': fs = 14

        ax.text(0, .2, label, fontweight="bold", color=color, fontsize=fs,
                ha="left", va="center", transform=ax.transAxes)

    num_methods = len(mydata)

    # convert data to dataframes for ridge plot
    temp = []
    for i in range(num_methods):
        df = pd.DataFrame(mydata[i], columns = [datalabel])
        df['method'] = method_labels[i+1]
        temp.append(df)

    # list of dataframes concatenated to single dataframe
    df = pd.concat(temp, ignore_index = True)

    if what_for == 'paper':
        ridgedict = {
            "h":0.5,
            "lw":0.7,
            "vl":0.25,
            "xfontsize":8,
        }
    elif what_for == 'talk':
        ridgedict = {
            "h":1.0,
            "lw":1.5,
            "vl":0.5,
            "xfontsize":16,
        }

    # Initialize the FacetGrid object
    pal = sns.palplot(sns.color_palette("tab10"))
    g = sns.FacetGrid(df, row="method", hue="method", aspect=15,
        height=ridgedict["h"], palette=pal)

    # draw filled-in densities, change bw for smoothing parameter
    g.map(sns.kdeplot, datalabel, clip_on=False, shade=True, alpha=0.5,
        lw=ridgedict["lw"], bw=.2)

    # draw outline around densities; can also single outline color: color="k"
    g.map(sns.kdeplot, datalabel, clip_on=False, lw=ridgedict["lw"], bw=.2)

    # draw horizontal line below densities
    g.map(plt.axhline, y=0, lw=ridgedict["lw"], clip_on=False)

    # draw a vertical line at x=0 for visual reference
    g.map(plt.axvline, x=0, lw=ridgedict["vl"], ls='--', color='gray', clip_on=False)

    # add labels to each level and to whole x-axis
    g.map(label, datalabel)

    # Set the subplots to overlap
    #g.fig.subplots_adjust(hspace=0.05)
    g.fig.subplots_adjust(hspace=-0.45)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # adjust font sizes
    plt.xlabel(datalabel, fontsize=ridgedict["xfontsize"])
    plt.xticks(fontsize=ridgedict["xfontsize"])

    # save with transparency for overlapping plots
    plt.savefig(out_file, bbox_inches='tight', transparent=True)
    plt.clf()
    #plt.show()


def main(in_dict, conf_id_tag, plot=False, mol_slice=None):
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
    method_labels = list(in_dict.keys())

    # enes_full[i][j][k] = ddE of ith method, jth mol, kth conformer.
    enes_full, rmsds_full, tfds_full, smiles_full = compare_ffs(
                                                        in_dict,
                                                        conf_id_tag,
                                                        'refdata',
                                                        mol_slice)

    # write enes_full to file since not so easy to save as SD tag
    with open('ddE.dat', 'w') as outfile:
        outfile.write("# Relative energies (kcal/mol) of ddE = dE (ref method) - dE (query method)\n")
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
    for a, b, c in zip(enes_full, rmsds_full, tfds_full):
        energies.append(flatten(a))
        rmsds.append(flatten(b))
        tfds.append(flatten(c))

    if plot:
        draw_scatter(
            rmsds,
            energies,
            method_labels,
            "RMSD ($\mathrm{\AA}$)",
            "ddE (kcal/mol)",
            "scatter_rmsd.png",
            "talk")
        draw_scatter(
            tfds,
            energies,
            method_labels,
            "TFD",
            "ddE (kcal/mol)",
            "scatter_tfd.png",
            "talk")
        draw_ridgeplot(
            energies,
            "ddE (kcal/mol)",
            method_labels,
            "ridge_dde.png",
            "talk")
        draw_ridgeplot(
            rmsds,
            "RMSD ($\mathrm{\AA}$)",
            method_labels,
            "ridge_rmsd.png",
            "talk")


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
    main(in_dict, args.conftag, args.plot, args.molslice)

