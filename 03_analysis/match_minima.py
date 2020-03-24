#!/usr/bin/env python

"""
match_minima.py

Match conformers from sets of different optimizations.
Compute relative energies of corresponding conformers to a reference conformer.
Generate plots for relative conformer energies (one plot per mol).

By:      Victoria T. Lim
Version: Dec 2 2019

"""

import os
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import openeye.oechem as oechem
import reader

### ------------------- Functions -------------------


def compare_two_mols(rmol, qmol, rmsd_cutoff):
    """
    For two identical molecules, with varying conformers,
    make an M by N comparison to match the M minima of
    rmol to the N minima of qmol. Match is declared
    for lowest RMSD between the two conformers and
    if the RMSD is below rmsd_cutoff.

    Parameters
    ----------
    rmol : OEMol
        reference OEChem molecule with all its filtered conformers
    qmol : OEMol
        query OEChem molecule with all its filtered conformers
    rmsd_cutoff : float
        cutoff above which two structures are considered diff conformers

    Returns
    -------
    molIndices : list
        1D list of qmol conformer indices that correspond to rmol confs

    """

    automorph = True   # take into acct symmetry related transformations
    heavyOnly = False  # do consider hydrogen atoms for automorphisms
    overlay = True     # find the lowest possible RMSD

    molIndices = []    # 1D list, stores indices of matched qmol confs wrt rmol

    for ref_conf in rmol.GetConfs():
        print(f">>>> Matching {qmol.GetTitle()} conformers to minima: "
              f"{ref_conf.GetIdx()+1} <<<<")

        # for this ref_conf, calculate/store RMSDs with all qmol's conformers
        thisR_allQ = []
        for que_conf in qmol.GetConfs():
            rms = oechem.OERMSD(ref_conf, que_conf, automorph, heavyOnly, overlay)
            thisR_allQ.append(rms)

        # for this ref_conf, get qmol conformer index of min RMSD if <=cutoff
        lowest_rmsd_index = [i for i, j in enumerate(thisR_allQ) if j == min(thisR_allQ)][0]
        if thisR_allQ[lowest_rmsd_index] <= rmsd_cutoff:
            molIndices.append(lowest_rmsd_index)
        else:
            print('no match bc rmsd is ', thisR_allQ[lowest_rmsd_index])
            molIndices.append(None)

    return molIndices


def plot_violin_signed(mses, ff_list, what_for='talk'):
    """
    Generate violin plots of the mean signed errors
    of force field energies with respect to QM energies.

    Parameters
    ----------
    mses : 2D list
        Mean signed errors for each method with reference to first input method
        mses[i][j] represents ith mol, jth method's MSE
    ff_list : list
        list of methods corresponding to energies in mses
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"

    """

    # create dataframe from list of lists
    df = pd.DataFrame.from_records(mses, columns=ff_list)
    medians = df.median(axis=0)

    print("\n\nDataframe of mean signed errors for each molecule, separated by force field\n")
    print(df)

    # reshape for grouped violin plots
    df = df.melt(var_name='groups', value_name='values')

    # set grid background style
    sns.set(style="whitegrid")

    if what_for == 'paper':
        f, ax = plt.subplots(figsize=(4, 3))
        large_font = 10
        small_font = 8
        lw = 1
        med_pt = 2
        xrot = 45
        xha = 'right'
    elif what_for == 'talk':
        f, ax = plt.subplots(figsize=(10, 8))
        large_font = 16
        small_font = 14
        lw = 2
        med_pt = 10
        xrot = 0
        xha = 'center'

        # overlapping violins
        #lw=1.0
        #f, ax = plt.subplots(figsize=(4, 8))

    # show each distribution with both violins and points
    sns.violinplot(x="groups", y="values", data=df, inner="box",
        palette="tab10", linewidth=lw)

    # replot the median point for larger marker, zorder to plot points on top
    xlocs = ax.get_xticks()
    for i, x in enumerate(xlocs):
        plt.scatter(x, medians[i], marker='o', color='white', s=med_pt, zorder=100)

    # represent the y-data on log scale
    plt.yscale('symlog')

    # set alpha transparency
    plt.setp(ax.collections, alpha=0.8)

    # hide vertical plot boundaries
    sns.despine(left=True)

    # add labels and adjust font sizes
    ax.set_xlabel("")
    ax.set_ylabel("mean signed error (kcal/mol)", size=large_font)
    plt.xticks(fontsize=small_font, rotation=xrot, ha=xha)
    plt.yticks(fontsize=large_font)

    # settings for overlapping violins
    #plt.xticks([])
    #plt.xlim(-1, 1)

    # save and close figure
    plt.savefig('violin.png', bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.close(plt.gcf())

    # reset plot parameters (white grid)
    sns.reset_orig()


def plot_mol_rmses(mol_name, rmses, xticklabels, eff_nconfs, ref_nconfs, what_for='talk'):
    """
    Generate bar plot of RMSEs of conformer energies for this molecule of
    all methods compared to reference method.

    Number of conformers used to calculate the RMSE is also plotted
    as a solid line. The number of possible conformers available
    by the reference method is plotted as a dashed line.

    Parameters
    ----------
    mol_name : string
        title of the mol being plotted
    rmses : list
        rmses[i] is the RMSE of this mol of reference compared to ith method
    xticklabels : list
        list of methods of the same length as rmses list; should not include
        reference method label
    eff_nconfs : list
        effective number of conformers with non-nan values;
        same format and length as rmses
    ref_nconfs : int
        number of conformers in the reference method
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"

    """

    # set figure-related labels
    plttitle = f"RMSEs of relative energies for\nmolecule {mol_name}"
    ylabel = "RMSE (kcal/mol)"
    figname = f"barRMSE_{mol_name}.png"

    # define x locations by integer number of methods
    x_locs = list(range(len(xticklabels)))

    if what_for == 'paper':
        fig = plt.gcf()
        fig.set_size_inches(4, 3)
        large_font = 14
        small_font = 10
    elif what_for == 'talk':
        large_font = 18
        small_font = 16

    # label figure; label xticks before plot for better spacing
    plt.title(plttitle, fontsize=large_font)
    plt.ylabel(ylabel, fontsize=large_font)
    plt.xticks(x_locs, xticklabels, fontsize=small_font, rotation=-30, ha='left')
    plt.yticks(fontsize=small_font)

    # define custom colors here if desired
    #colors = ['tab:blue']*len(xticklabels)
    colors = ['tab:blue']*2 + ['tab:orange']*2 + ['tab:green']*2

    # plot rmses as bars
    plt.bar(x_locs, rmses, color=colors, align='center', label='RMSE')

    # plot number of conformers as lines
    ax2 = plt.twinx()
    ax2.plot(x_locs, eff_nconfs, color='k', alpha=0.5,
             label='actual num confs')
    ax2.axhline(ref_nconfs, color='k', alpha=0.5, ls='--',
                label='reference num confs')

    # format line graph properties, then add plot legend
    ax2.set_ylabel('Number of conformers', fontsize=large_font)
    ax2.tick_params(axis='y', labelsize=small_font)
    ax2.yaxis.set_ticks(np.arange(min(eff_nconfs)-1, ref_nconfs+2, 1))
    plt.legend()

    # save and close figure
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.close(plt.gcf())


def plot_mol_minima(mol_name, minimaE, legend, what_for='talk', selected=None):
    """
    Generate line plot of conformer energies of all methods (single molecule).

    Parameters
    ----------
    mol_name : string
        title of the mol being plotted
    minimaE : list of lists
        minimaE[i][j] represents ith method and jth conformer energy
    legend : list
        list of strings with all method names in same order as minimaE
    what_for : string
        dictates figure size, text size of axis labels, legend, etc.
        "paper" or "talk"
    selected : list
        list of indices for methods to be plotted; e.g., [0], [0, 4]

    """

    # get details of reference method
    ref_nconfs = len(minimaE[0])
    ref_file = legend[0]
    num_files = len(minimaE)

    # flatten the 2D list into 1D to find min and max for plot
    flatten = [item for sublist in minimaE for item in sublist]
    floor = min(flatten)
    ceiling = max(flatten)

    # set figure-related labels
    plttitle = f"Relative Energies of {mol_name} Minima"
    plttitle += f"\nby Reference: {ref_file}"
    ylabel = "Relative energy (kcal/mol)"
    figname = f"minimaE_{mol_name}.png"

    # set xtick labels by either letters or numbers
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    rpt = int((len(minimaE[0]) / 26) + 1)
    xlabs = [''.join(i)
             for i in itertools.product(letters, repeat=rpt)][:ref_nconfs]
    # xticks by numbers instead
    #xlabs = range(len(minimaE[0]))

    if what_for == 'paper':
        fig = plt.figure(figsize=(7.5, 3))
        large_font = 12
        small_font = 10
        xaxis_font = 6
        mark_size = 5
    elif what_for == 'talk':
        fig = plt.figure(figsize=(20, 8))
        large_font = 18
        small_font = 14
        xaxis_font = 10
        mark_size = 9

    # create figure
    ax = fig.gca()
    ax.set_xticks(np.arange(-1, ref_nconfs + 1, 2))

    # label figure; label xticks before plotting for better spacing.
    plt.title(plttitle, fontsize=large_font)
    plt.ylabel(ylabel, fontsize=large_font)
    plt.xlabel("conformer minimum", fontsize=large_font)
    plt.xticks(list(range(ref_nconfs)), xlabs, fontsize=xaxis_font)
    plt.yticks(fontsize=small_font)

    # define line colors and markers
    colors = mpl.cm.rainbow(np.linspace(0, 1, num_files))
    markers = [
        "x", "^", "8", "d", "o", "s", "*", "p", "v", "<", "D", "+", ">", "."
    ] * 10

    # plot the data
    for i, file_ene in enumerate(minimaE):

        # skip the non-selected ones if defined
        if selected is not None and i not in selected:
            continue

        # define x's from integer range with step 1
        xi = list(range(ref_nconfs))

        # generate plot
        plt.plot(xi, file_ene, color=colors[i], label=legend[i],
            marker=markers[i], markersize=mark_size, alpha=0.6)

    # add legend, set plot limits, add grid
    plt.legend(bbox_to_anchor=(0.96, 1), loc=2, prop={'size': small_font})
    plt.xlim(-1, ref_nconfs + 1)
    ax.set_yticks(
        np.arange(int(round(floor)) - 2,
                  int(round(ceiling)) + 2))
    plt.grid()

    # save and close figure
    plt.savefig(figname, bbox_inches='tight')
    #plt.show()
    fig.clear()
    plt.close(fig)


def match_minima(in_dict, rmsd_cutoff):
    """
    For different methods, match the conformer minima to those of the reference
    method. Ex. Conf G of reference method matches with conf R of method 2.

    Parameters
    ----------
    in_dict : OrderedDict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'
    rmsd_cutoff : float
        cutoff above which two structures are considered diff conformers

    Returns
    -------
    mol_dict : dict of dicts
        mol_dict['mol_name']['energies'] =
            [[file1_conf1_E file1_conf2_E] [file2_conf1_E file2_conf2_E]]
        An analogous structure is followed for mol_dict['mol_name']['indices'].

    """

    # nested dictionary: 1st layer of mol names, 2nd layer of method energies
    mol_dict = {}

    # get first filename representing the reference geometries
    sdf_ref = list(in_dict.values())[0]['sdfile']

    # assess each file against reference
    for ff_label, ff_dict in in_dict.items():
        sdf_query = ff_dict['sdfile']
        sdf_tag = ff_dict['sdtag']

        # load molecules from open reference and query files
        print(f"\n\nOpening reference file {sdf_ref}")
        mols_ref = reader.read_mols(sdf_ref)

        print(f"Opening query file {sdf_query} for [ {ff_label} ] energies")
        mols_query = reader.read_mols(sdf_query)

        # loop over each molecule in reference and query files
        for rmol in mols_ref:
            mol_name = rmol.GetTitle()
            ref_nconfs = rmol.NumConfs()
            run_match = False

            for qmol in mols_query:

                # same mol titles should mean same molecular identity;
                # when same molecular identity found, break out of loop to
                # start matching conformers
                if rmol.GetTitle() == qmol.GetTitle():
                    run_match = True
                    break

            # create entry for this mol in mol_dict if not already present
            # energies [i][j] will be 2d list of ith method and jth conformer
            if mol_name not in mol_dict:
                mol_dict[mol_name] = {'energies': [], 'indices': []}

            # no same molecules were found bt ref and query methods
            # for N reference minima of each mol, P matching indices for each ref minimia
            if not run_match:
                print(f"No \"{mol_name}\" molecule found in {sdf_query}")

                # fill in -2 error values for conformer indices
                mol_dict[mol_name]['indices'].append([-2] * ref_nconfs)

                # fill in nan values for conformer energies and ref_nconfs
                mol_dict[mol_name]['energies'].append([np.nan] * ref_nconfs)

                # reset mols_query generator
                mols_query = reader.read_mols(sdf_query)

                # continue with the next rmol
                continue

            # get data from specified sd tag for all conformers
            data_confs = reader.get_sd_list(qmol, sdf_tag)

            # format sd tag data to float types
            float_data_confs = list(map(float, data_confs))

            # store sd data from tags into dictionary
            mol_dict[mol_name]['energies'].append(float_data_confs)

            # don't run match if query method is same as reference method
            # keep this section after sd tag extraction of energies
            if sdf_query == sdf_ref:
                print("Skipping comparison against self.")
                mol_dict[mol_name]['indices'].append([-1] * ref_nconfs)
                continue

            # run the match here
            # get indices of qmol conformers that match rmol conformers
            molIndices = compare_two_mols(rmol, qmol, rmsd_cutoff)
            mol_dict[mol_name]['indices'].append(molIndices)

    return mol_dict


def calc_rms_error(rel_energies, lowest_conf_indices):
    """
    From relative energies with respect to some conformer from calc_rel_ene,
    calculate the root mean square error with respect to the relative
    conformer energies of the first (reference) method.

    Parameters
    ----------
    rel_energies : 3D list
        energies, where rel_energies[i][j][k] represents ith mol, jth method,
        kth conformer rel energy
    lowest_conf_indices : 1D list
        indices of the lowest energy conformer per each mol

    Returns
    -------
    rms_array : 2D list
        RMS errors for each method with reference to first input method
        rms_array[i][j] represents ith mol, jth method's RMSE
    mse_array : 2D list
        same layout as that of rms_array except containing mean squared errors

    """
    rms_array = []
    mse_array = []

    # iterate over each molecule
    for i, mol_array in enumerate(rel_energies):
        mol_rmses = []
        mol_mses = []

        # iterate over each file (method)
        for j, filelist in enumerate(mol_array):

            # subtract query file minus reference file
            errs = np.asarray(filelist) - np.asarray(mol_array[0])

            # delete reference conformer since it has zero relative energy
            errs = np.delete(errs, lowest_conf_indices[i])

            # square
            sqrs = errs**2.

            # also delete any nan values (TODO: treat this differently?)
            sqrs = sqrs[~np.isnan(sqrs)]

            # mean, root, store
            mse = np.mean(sqrs)
            rmse = np.sqrt(mse)
            mol_rmses.append(rmse)

            # also calculate mse
            sum_errs = np.sum(errs)
            mse = sum_errs/len(errs)
            mol_mses.append(mse)

        rms_array.append(mol_rmses)
        mse_array.append(mol_mses)

    return rms_array, mse_array


def calc_rel_ene(matched_enes):
    """
    Calculate conformer relative energies of matching conformers.
    For each method, subtract minimum conformer energy from all conformers.
    The minimum-energy conformer minimum is chosen from first
    finding the method with the least number of missing energies,
    then of that method, choosing the lowest energy conformer.

    For mols with a single conformer it doesn't make sense to calculate
    relative energies. These mols are removed from matched_enes.

    Parameters
    ----------
    matched_enes : 3D list
        energies, matched_enes[i][j][k] represents energy of
        ith mol, jth method, kth conformer

    Returns
    -------
    rel_energies : 3D list
        energies in same format as matched_enes except with relative energies
    lowest_conf_indices : 1D list
        indices of the lowest energy conformer by reference mols
    eff_nconfs : 2D list
        effective number of conformers with non-nan values
        eff_nconfs[i][j] is for ith mol, jth method

    """

    lowest_conf_indices = []
    eff_nconfs = []

    # loop over molecules
    for i, mol_array in enumerate(matched_enes):

        # get number of conformers in reference method
        ref_nconfs = len(mol_array[0])

        # for this mol, count number of nans for each conf of all methods
        # 1d list of length ref_nconfs, where max value is num_methods
        nan_cnt = []
        for j in range(ref_nconfs):
            nan_cnt.append(sum(np.isnan([file_enes[j] for file_enes in mol_array])))
        #print("mol {} nan_cnt: {}".format(i, nan_cnt))

        # store the effective number of conformers per method
        eff_cnt = sum(~np.isnan(np.array(mol_array).T))
        eff_nconfs.append(eff_cnt)

        # find which method has fewest nans; equiv to finding which query
        # method has most number of conformer matches

        # no_nan_conf_inds: list of conf indices with no nans across all files
        no_nan_conf_inds = np.empty(0)
        cnt = 0

        # check which confs have 0 method nans. if none of them, check which
        # have 1 nan across all files, etc. repeat until you find the smallest
        # nan value for which you get conformer indices with that number nans.
        # leave loop when 'no_nan_conf_inds' is assigned or if nothing left in nan_cnt
        while no_nan_conf_inds.size == 0 and cnt < ref_nconfs:
            no_nan_conf_inds = np.where(np.asarray(nan_cnt) == cnt)[0]
            cnt += 1

        # for an existing no_nan_conf_inds, get lowest energy conf of reference method
        if no_nan_conf_inds.size > 0:

            # get energies from reference [0] file
            leastNanEs = np.take(mol_array[0], no_nan_conf_inds)

            # find index of the lowest energy conformer
            lowest_conf_idx = no_nan_conf_inds[np.argmin(leastNanEs)]
            lowest_conf_indices.append(lowest_conf_idx)

        # if no_nan_conf_inds not assigned, this means all methods had all nans
        # in this case just get index of conformer with lowest number of nans
        else:
            lowest_conf_indices.append(nan_cnt.index(min(nan_cnt)))

    # after going thru all mols, calc relative energies by subtracting lowest
    rel_energies = []
    for z, molE in zip(lowest_conf_indices, matched_enes):

        # temp list for this mol's relative energies
        temp = []

        # subtract energy of lowest_conf_index; store in list
        for fileE in molE:
            temp.append(
                [(fileE[i] - fileE[z]) for i in range(len(fileE))])
        rel_energies.append(temp)

    return rel_energies, lowest_conf_indices, eff_nconfs


def write_rel_ene(mol_name, rmse, rel_enes, low_ind, ff_list, prefix='relene'):
    """
    Write the relative energies and RMSEs in an output text file.

    Parameters
    ----------
    mol_name : string
        title of the mol being written out
    rmse : list
        1D list of RMSEs for all the compared methods against ref method
    rel_enes : 2D list
        rel_enes[i][j] represents energy of ith method and jth conformer
    low_ind : int
        integer of the index of the lowest energy conformer
    ff_list : list
        list of methods including reference as the first
    prefix : string
        prefix of output dat file

    """

    ofile = open(f"{prefix}_{mol_name}.dat", 'w')
    ofile.write(f"# Molecule {mol_name}\n")
    ofile.write("# Energies (kcal/mol) for conformers matched to first method.")
    ofile.write(f"\n# Energies are relative to conformer {low_ind}.\n")
    ofile.write("# Rows represent conformers; columns represent methods.\n")

    # write methods, RMSEs, integer column header
    rmsheader = "\n# "
    colheader = "\n\n# "
    for i, method in enumerate(ff_list):
        ofile.write(f"\n# {i+1} {method}")
        rmsheader += f"\t{rmse[i]:.4f}"
        colheader += f"\t\t{i+1}"

    ofile.write("\n\n# RMS errors by method, with respect to the " +
                "first method listed:")
    ofile.write(rmsheader)
    ofile.write(colheader)

    # write each ff's relative energies
    for i in range(len(rel_enes[0])):

        # label conformer row
        ofile.write(f"\n{i}\t")

        # write energies for this conformer of all methods
        thisline = [x[i] for x in rel_enes]
        thisline = [f"{elem:.4f}" for elem in thisline]
        thisline = '\t'.join(map(str, thisline))
        ofile.write(thisline)

    ofile.close()


def extract_matches(mol_dict):
    """
    This function checks if minima are matched, using conformer indices.
    For example, mol_dict[example_mol]['indices'] = [ 0 2 1 ] means that,
    compared to the reference conformers, the queried method has confs 1 and 2
    switched. The switch will be made to compare corresponding energies
    for matching conformers.

    If the query method doesn't have a match to one of the reference conformers,
    then 'None' will be in the list of indices. Other indices of NON-match
    are -1 (query method the same as reference method so everything matches)
    and -2 (query method missing the whole molecule of the reference method).

    If a match is found, store the corresponding energy under a new key with
    value of "energies_matched". If there is no match, the energy listed in the
    dict is not used; rather, nan is added as placeholder.

    Parameter
    ---------
    mol_dict : dict of dicts
        mol_dict['mol_name']['energies'] =
            [[file1_conf1_E file1_conf2_E] [file2_conf1_E file2_conf2_E]]
        An analogous structure is followed for mol_dict['mol_name']['indices'].

    Returns
    -------
    mol_dict : dict of dicts
        mol_dict['mol_name']['energies']
        mol_dict['mol_name']['energies_matched']

    """

    # iterate over each molecule
    for m in mol_dict:

        # 2D list, [i][j] i=filename (method), j=index
        # index represents queried mol's conformation location that matches
        # the ref mol's conformer
        queried_indices = mol_dict[m]['indices']

        # 2D list, [i][j] i=filename (method), j=energy
        energy_array = mol_dict[m]['energies']

        # based on the indices, extract the matching energies
        updated = []

        for i, file_indices in enumerate(queried_indices):
            fileData = []

            for j, conf_index in enumerate(file_indices):

                # conformers were matched but all RMSDs were beyond cutoff
                # as set in the compare_two_mols function
                if conf_index is None:
                    print(f"No matching conformer within RMSD cutoff for {j}th"
                          f" conf of {m} mol in {i}th file.")
                    fileData.append(np.nan)

                # the query molecule was missing
                elif conf_index == -2:
                    # only print this warning message once per mol
                    if j == 0:
                        print(f"!!!! The entire {m} mol is not found in "
                              f"{i}th file. !!!!")
                    fileData.append(np.nan)

                # energies are missing somehow?
                elif len(energy_array[i]) == 0:
                    print(f"!!!! Mol {m} was found and confs were matched by "
                          f"RMSD but there are no energies of {i}th method. !!!!")
                    fileData.append(np.nan)

                # conformers not matched bc query file = reference file
                # reference indices therefore equals query indices
                elif conf_index == -1:
                    fileData.append(float(energy_array[i][j]))

                # conformers matched and there exists match within cutoff
                else:
                    fileData.append(float(energy_array[i][conf_index]))

            updated.append(fileData)

        mol_dict[m]['energies_matched'] = updated

    return mol_dict


def main(in_dict, read_pickle, plot, rmsd_cutoff):
    """
    Match conformers from sets of different optimizations.
    Compute relative energies of corresponding confs to a reference conformer.
    Generate plots for relative conformer energies (one plot per mol).

    Parameter
    ---------
    in_dict : OrderedDict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'
    read_pickle : Boolean
        read in data from match.pickle
    plot : Boolean
        generate line plots of conformer energies
    rmsd_cutoff : float
        cutoff above which two structures are considered diff conformers

    """

    # run matching, unless reading in from pickle file
    if read_pickle:
        mol_dict = pickle.load(open('match.pickle', 'rb'))
    else:
        # match conformers
        mol_dict = match_minima(in_dict, rmsd_cutoff)

        # save results in pickle file
        pickle.dump(mol_dict, open('match.pickle', 'wb'))

    # process dictionary to match the energies by RMSD-matched conformers
    mol_dict = extract_matches(mol_dict)

    # collect the matched energies into a list of lists
    matched_enes = []
    for m in mol_dict:
        matched_enes.append(mol_dict[m]['energies_matched'])

    # with matched energies, calculate relative values and RMS error
    rel_energies, lowest_conf_indices, eff_nconfs = calc_rel_ene(matched_enes)
    rms_array, mse_array = calc_rms_error(rel_energies, lowest_conf_indices)

    # write out data file of relative energies
    mol_names = mol_dict.keys()
    ff_list = list(in_dict.keys())

    for i, mn in enumerate(mol_names):
        write_rel_ene(mn, rms_array[i], rel_energies[i], lowest_conf_indices[i], ff_list)

    if plot:

        # customize: exclude outliers from violin plots
        mol_names = list(mol_names)
        violin_exclude = ['full_549', 'full_590', 'full_802', 'full_1691', 'full_1343', 'full_2471']
        idx_of_exclude = [mol_names.index(x) for x in violin_exclude]
        for idx in sorted(idx_of_exclude, reverse=True):
            del mse_array[idx]

        # plots combining all molecules -- skip reference bc 0 rmse to self
        # mse_array[i][j] represents ith mol, jth method's RMSE
        plot_violin_signed(np.array(mse_array)[:, 1:], ff_list[1:], 'talk')

        # molecule-specific plots
        print("\nGenerating bar and line plots for individual mols. This might take a while.")
        for i, mol_name in enumerate(mol_dict):
            print(mol_name)

            # optional: only plot single molecule by specified title
            #if mol_name != 'full_549': continue
            if mol_name in violin_exclude: continue

            # optional: only plot selected force fields by index
            #plot_mol_minima(mol_name, rel_energies[i], ff_list, selected=[0])

            # line plots of relative energies
            plot_mol_minima(mol_name, rel_energies[i], ff_list, 'talk')

            # bar plots of RMSEs by force field -- skip reference bc 0 rmse to self
            ref_nconfs = eff_nconfs[i][0]
            plot_mol_rmses(mol_name, rms_array[i][1:], ff_list[1:],
                           eff_nconfs[i][1:], ref_nconfs, 'talk')



### ------------------- Parser -------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile",
        help="Name of text file with force field in first column and molecule "
             "file in second column. Columns separated by commas.")

    parser.add_argument("--readpickle", action="store_true", default=False,
        help="Read in already-matched data from pickle file named \"match.pickle\"")

    parser.add_argument("--plot", action="store_true", default=False,
        help="Generate line plots for every molecule with the conformer "
             "relative energies.")

    parser.add_argument("--cutoff", type=float, default=1.0,
        help="RMSD cutoff above which conformers are considered different "
             "enough to be distinct. Corresponding energies not considered. "
             "Measured by OpenEye in with automorph=True, heavyOnly=False, "
             "overlay=True. Units in Angstroms.")


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

    # run match_minima
    main(in_dict, args.readpickle, args.plot, args.cutoff)

