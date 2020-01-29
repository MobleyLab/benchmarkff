#!/usr/bin/env python

"""
probe_parameter.py

For specific force field parameters identified from the analysis of
tailed_parameters.py, (1) find all molecules that use this parameter,
and (2) save them to a single file labeled with the parameter ID.
Then (3) generate a PDF report of all molecules together, color-coded
by parameter ID and labeled with parameter ID and SMILES tag.

By:      Victoria T. Lim
Version: Jan 28 2020

Examples:
$ python probe_parameter.py -f openff_unconstrained-1.0.0-RC2.offxml -k tailed_rmsd.pickle -s probe_params_rmsd -p a20 b9 n3 t96

"""

import os
import pickle

import openeye.oechem as oechem
import openeye.oedepict as oedepict

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField


### ------------------- Functions -------------------


def probe_by_parameter(probe_param, ffxml, all_probe_mols, subdir, pickle_file):
    """
    probe_param : string
        Name of the parameter to investigate
    ffxml : string
        Name of the FFXML force field file
    pickle_file : string
        Name of the pickle file from output of tailed_parameters.py
    """
    prefix_dict = {'a':'Angles', 'b':'Bonds', 'i':'ImproperTorsions', 'n':'vdW', 't':'ProperTorsions'}

    # load parameter dictionaries from pickle
    mols_out, params_id_out, smi_dict_out, mols_all, params_id_all, smi_dict_all = pickle.load(open(pickle_file, 'rb'))

    # find the first outlier mol that has given param
    mols_with_probe = list(params_id_out[probe_param])
    probe_mol = Molecule.from_smiles(mols_with_probe[0], allow_undefined_stereo=True)
    topology = Topology.from_molecules([probe_mol])

    # load in force field
    ff = ForceField(ffxml)

    # run molecule labeling
    molecule_force_list = ff.label_molecules(topology)

    # get the smirks pattern associated with param
    prefix = probe_param[0]
    force_dict = molecule_force_list[0][prefix_dict[prefix]]
    for (k, v) in force_dict.items():
        if v.id == probe_param:
            probe_smirks = v.smirks
            break
    print(f"\n=====\n{probe_param}: {probe_smirks}\n=====")

    # find all molecules with this parameter and save to file.
    # conformers are not considered here so these smiles refer to
    # arbitrary conformer from dict of zip
    ofs = oechem.oemolostream()
    if not ofs.open(f'{subdir}/param_{probe_param}.mol2'):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % outfile)

    for m in mols_with_probe:
        key = smi_dict_out[m]
        print(f"writing out {key}")
        mymol = mols_out[key]['structure']
        oechem.OEWriteConstMolecule(ofs, mymol)

        # save to write full pdf
        all_probe_mols[probe_param].append(oechem.OEGraphMol(mymol))

    return all_probe_mols


def oedepict_pdf(all_probe_mols):
    multi = oedepict.OEMultiPageImageFile(oedepict.OEPageOrientation_Landscape,
                                          oedepict.OEPageSize_US_Letter)
    image = multi.NewPage()

    opts = oedepict.OE2DMolDisplayOptions()

    rows, cols = 4, 4
    grid = oedepict.OEImageGrid(image, rows, cols)
    grid.SetCellGap(20)
    grid.SetMargins(20)
    citer = grid.GetCells()

    colors = list(oechem.OEGetContrastColors())

    for i, (param, mol_list) in enumerate(all_probe_mols.items()):

        pen = oedepict.OEPen(oechem.OEWhite, colors[i], oedepict.OEFill_Off, 4.0)

        for mol in mol_list:

            # go to next page
            if not citer.IsValid():
                image = multi.NewPage()
                grid = oedepict.OEImageGrid(image, rows, cols)
                grid.SetCellGap(20)
                grid.SetMargins(20)
                citer = grid.GetCells()

            cell = citer.Target()
            mol.SetTitle(f"{param}   {oechem.OEGetSDData(mol, 'SMILES QCArchive')}")
            oedepict.OEPrepareDepiction(mol)
            opts.SetDimensions(cell.GetWidth(), cell.GetHeight(), oedepict.OEScale_AutoScale)
            disp = oedepict.OE2DMolDisplay(mol, opts)
            oedepict.OERenderMolecule(cell, disp)
            oedepict.OEDrawBorder(cell, pen)
            citer.Next()

    oedepict.OEWriteMultiPageImage("{subdir}/results.pdf", multi)




### ------------------- Parser -------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--ffxml", required=True,
            help="Open force field ffxml file")

    parser.add_argument("-k", "--pickle", required=True,
            help="Name of the pickle file from output of tailed_parameters.py")

    parser.add_argument("-s", "--subdir", default='probe_params',
            help="Name of subdirectory in which to save mol2 and pdf files")

    parser.add_argument("-p", "--params", required=True, nargs='+',
            help="One or more parameters to investigate")

    args = parser.parse_args()

    if not os.path.exists(args.subdir):
        os.mkdir(args.subdir)

    all_probe_mols = {}

    for p in args.params:
        all_probe_mols[p] = []
        all_probe_mols = probe_by_parameter(p, args.ffxml, args.subdir, all_probe_mols)

    oedepict_pdf(all_probe_mols)
