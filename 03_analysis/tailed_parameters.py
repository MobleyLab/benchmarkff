#!/usr/bin/env python

"""
tailed_parameters.py

For distributions of RMSD or TFDs of force field geometries with respect to
reference geometries, identify outlier molecules in the high RMSD (TFD) tails
above a predetermined cutoff value. For each force field parameter in the
molecule set, determine the fraction of outlier molecules with that parameter.
Compare that to the fraction of all molecules that use that parameter.

By:      Victoria T. Lim
Version: Jan 28 2020

References:
https://github.com/openforcefield/openforcefield/blob/master/examples/inspect_assigned_parameters/inspect_assigned_parameters.ipynb

Note:
- Make sure to use the same FFXML file that was used to generate the minimized geometries.

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
import openeye.oedepict as oedepict

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
    mols_dict : dict
        TODO
    outfile : string
        name of output file
    """

    # open an outstream file
    ofs = oechem.oemolostream()
#    if os.path.exists(outfile):
#        raise FileExistsError("Output file {} already exists in {}".format(
#            outfile, os.getcwd()))
    if not ofs.open(outfile):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % outfile)

    # go through all the molecules
    for key in mols_dict:
        print(f"writing out {key}")
        mymol = mols_dict[key]['structure']
        oechem.OEWriteConstMolecule(ofs, mymol)


def get_parameters(mols_dict, ffxml):
    """
    Parameters
    ----------
    mols_dict : dict
        TODO
    ffxml : string
        name of FFXML force field file
    """

    # load in force field
    ff = ForceField(ffxml)

    # convert OEMols to open force field molecules
    off_mols = []

    for i, key in enumerate(mols_dict):
        # get mol from the dict
        mymol = mols_dict[key]['structure']

        # create openforcefield molecule from OEMol
        # note: stereo error raised even though coordinates present (todo?)
        off_mol = Molecule.from_openeye(mymol, allow_undefined_stereo=True)
        off_mols.append(off_mol)

    # form a dictionary to backtrace the iso_smiles to original molecule
    iso_smiles = [ molecule.to_smiles() for molecule in off_mols ]
    smi_list = mols_dict.keys()
    smi_dict = dict(zip(iso_smiles, smi_list))

    # remove duplicate molecules (else get_molecule_parameterIDs gives err)
    idx_of_duplicates = [idx for idx, item in enumerate(iso_smiles) if item in iso_smiles[:idx]]
    for index in sorted(idx_of_duplicates, reverse=True):
        del off_mols[index]

    # create dictionaries describing parameter assignment,
    # grouped both by molecule and by parameter
    parameters_by_molecule, parameters_by_ID = get_molecule_parameterIDs(off_mols, ff)

    return parameters_by_molecule, parameters_by_ID, smi_dict


def param_num_mols(full_params, params_id_all, params_id_out):
    """
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

    return nmols_cnt_all, nmols_cnt_out


def plot_by_paramtype(prefix, full_params, fraction_cnt_all, fraction_cnt_out, metric_type, exclude_empty_outliers=False):
    """
    prefix : string
        specify parameter type to generate plot.
        options: 'a' 'b' 'i' 'n' 't'
    """

    if exclude_empty_outliers:
        pass
        # TODO

    plot_inds = [full_params.index(i) for i in full_params if i.startswith(prefix)]

    # create the plot and set label sizes
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    size1 = 20
    size2 = 16

    # set x locations and bar widths
    x = np.arange(len(plot_inds))
    width = 0.3

    # plot the bars
    rects1 = ax.bar(x - width/2, fraction_cnt_all[plot_inds], width, label='all', color='darkcyan')
    rects2 = ax.bar(x + width/2, fraction_cnt_out[plot_inds], width, label=f'{metric_type} outliers', color='chocolate')

    # add plot labels
    ax.set_ylabel('fraction', fontsize=size1)
    ax.set_xlabel('force field parameter', fontsize=size1)
    ax.set_title('fraction of molecules having given parameter', fontsize=size1)
    ax.set_xticks(x)
    ax.set_xticklabels([full_params[index] for index in plot_inds], fontsize=size2)
    plt.yticks(fontsize=size2)
    ax.legend(fontsize=size2)
    fig.tight_layout()
    plt.savefig(f'bars_{metric_type.lower()}_params_{prefix}.png', bbox_inches='tight')


def tailed_parameters(in_sdf, ffxml, cutoff, tag, tag_smiles, metric_type):
    """
    """

    # load molecules from open reference and query files
    print(f"Opening SDF file {in_sdf}...")
    mols = reader.read_mols(in_sdf)
    print(f"Looking for outlier molecules with {metric_type} above {cutoff}...\n")


    # find the molecules with the metric above the cutoff
    all_smiles    = []
    mols_all      = OrderedDict()
    mols_out      = OrderedDict()
    count_all     = 0
    count_out     = 0

    for mol in mols:
        for conf in mol.GetConfs():

            value  = float(oechem.OEGetSDData(conf, tag))
            smiles = oechem.OEGetSDData(conf, tag_smiles)

            if value >= cutoff:
                mols_out[smiles] = {'metric': value, 'structure': oechem.OEGraphMol(conf)}
                count_out += 1

            mols_all[smiles] = {'structure': oechem.OEGraphMol(conf)}
            all_smiles.append(smiles)
            count_all += 1

    # save outliers molecules to file
    write_mols(mols_out, f'outliers_{metric_type.lower()}.mol2')

    # analyze parameters in the outlier and full sets
    params_mol_out, params_id_out, smi_dict_out = get_parameters(mols_out, ffxml)
    params_mol_all, params_id_all, smi_dict_all = get_parameters(mols_all, ffxml)

    # save the params organized by id to pickle
    pickle.dump(
        (mols_out, params_id_out, smi_dict_out, mols_all, params_id_all, smi_dict_all),
        open(f'tailed_{metric_type.lower()}.pickle', 'wb'))

    # count number of unique mol and unique params
    uniq_n_all = len(params_mol_all)
    uniq_n_out = len(params_mol_out)

    full_params = list(set(params_id_all.keys()))
    full_params.sort(key=natural_keys)

    uniq_p_all = len(full_params)
    uniq_p_out = len(list(set(params_id_out.keys())))

    # print stats on number of outliers
    print(f"\nNumber of structures in full set: {count_all} ({uniq_n_all} unique)")
    print(f"Number of structures in outlier set: {count_out} ({uniq_n_out} unique)")
    print(f"Number of unique parameters in full set: {uniq_p_all}")
    print(f"Number of unique parameters in outlier set: {uniq_p_out}")

    # go through all parameters and find number of molecules which use each one
    nmols_cnt_all, nmols_cnt_out = param_num_mols(full_params, params_id_all, params_id_out)
    fraction_cnt_all = np.array(nmols_cnt_all)/uniq_n_all
    fraction_cnt_out = np.array(nmols_cnt_out)/uniq_n_out

    # plot fraction of molecules which use each parameter
    # separate plot by parameter type
    for t in ['a', 'b', 'i', 'n', 't']:
        plot_by_paramtype(t, full_params, fraction_cnt_all, fraction_cnt_out, metric_type)


### ------------------- Parser -------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile", required=True,
            help="Input molecule file")

    parser.add_argument("-f", "--ffxml", required=True,
            help="Open force field ffxml file")

    parser.add_argument("--rmsd", action="store_true", default=False,
        help="Tag and cutoff value refer to RMSD metrics.")

    parser.add_argument("--tfd", action="store_true", default=False,
        help="Tag and cutoff value refer to TFD metrics.")

    parser.add_argument("--cutoff", required=True, type=float,
            help="Cutoff value for which to separate outliers")

    parser.add_argument("--tag", required=True,
            help="SDF tag from which to obtain RMSDs or TFDs")

    parser.add_argument("--tag_smiles", required=True,
            help="SDF tag from which to identify conformers")

    args = parser.parse_args()
    if args.rmsd:
        metric_type = 'RMSD'
    elif args.tfd:
        metric_type = 'TFD'
    else:
        pass
        # TODO

    tailed_parameters(args.infile, args.ffxml,
        args.cutoff, args.tag, args.tag_smiles, metric_type)

