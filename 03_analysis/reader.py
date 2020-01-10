#!/usr/bin/env python

"""
reader.py

Functions to parse input files (for Quanformer / BenchmarkFF) or
OEMols from multi-molecule SDF files.

By:      Victoria T. Lim
Version: Jan 10 2020

"""

import os
import numpy as np
import copy
import collections

import openeye.oechem as oechem
from rdkit import Chem, Geometry

def read_mols(in_file, mol_slice=None):
    """
    Open a molecule file and return molecules and conformers as OEMols.
    Provide option to slice the mols to return only a chunk from the
    specified indices.

    Parameters
    ----------
    in_file : string
        name of input file with molecules
    mol_slice : numpy slice object
        The resulting integers are numerically sorted and duplicates removed.
        e.g., slices = np.s_[0, 3:5, 6::3] would be parsed to return
        [0, 3, 4, 6, 9, 12, 15, 18, ...]
        Can also parse from end: [-3:] gets the last 3 molecules, and
        [-2:-1] is the same as [-2] to get just next to last molecule.

    Returns
    -------
    mols : OEMols

    """
    def flatten(x):
        # https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]

    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(in_file):
        raise FileNotFoundError(f"Unable to open {in_file} for reading")
    mols = ifs.GetOEMols()

    if mol_slice is None:
        return mols

    # set max number of molecules for decoding slices
    # TODO: how to get num_mols without re-reading file and loading all mols
    ifs2 = oechem.oemolistream()
    ifs2.SetConfTest(oechem.OEAbsCanonicalConfTest())
    ifs2.open(in_file)
    mols2 = ifs2.GetOEMols()
    num_mols = len(list(mols2))

    # parse mol_slice for multiple slice definitions provided
    # e.g., (1, 4, 8) for second, fifth, and ninth molecules
    # e.g., (0, slice(3, 5, None), slice(6, None, 3)) for example in docs

    if isinstance(mol_slice, tuple) or isinstance(mol_slice, list):
        idx_to_keep = []
        for s in mol_slice:

            # parse the slice object
            if isinstance(s, slice):
                idx_to_keep.append(list(range(num_mols))[s])

            # else decode the negative int to positive int
            elif isinstance(s, int) and s<0:
                idx_to_keep.append(s + num_mols)

            # else just append the positive int
            elif isinstance(s, int):
                idx_to_keep.append(s)

            else:
                raise ValueError(f"ERROR in parsing 'mol_slice' from {mol_slice}"
                                 f" due to {s} being neither slice nor int")

        # flatten to 1d, use set to remove duplicates, then sort list
        idx_to_keep = list(set(flatten(idx_to_keep)))
        idx_to_keep.sort()
        #print(idx_to_keep)

    elif isinstance(mol_slice, slice):
        # parse the slice object
        idx_to_keep = list(range(num_mols))[mol_slice]

    # else just store the single value in a list
    elif isinstance(mol_slice, int):
        if mol_slice < 0:
            mol_slice = mol_slice + num_mols
        idx_to_keep = list(mol_slice)

    else:
        raise ValueError(f"ERROR in parsing 'mol_slice' from {mol_slice}")

    # go through the generator and retrive the specified slices
    mlist = []
    for i, m in enumerate(mols):
        if i in idx_to_keep:

            # append a copy else still linked to orig generator
            mlist.append(copy.copy(m))

            # if this index is the last one in idx_to_keep, finish now
            if i == idx_to_keep[-1]:
                return mlist

    return mlist


def get_sd_list(mol, taglabel):
    """
    Get list of specified SD tag for all confs in mol.

    Parameters
    ----------
    mol : OEMol with N conformers
    taglabel : string
        tag from which to extract SD data

    Returns
    -------
    sdlist : list
        N-length list with value from SD tag

    """

    sd_list = []

    for j, conf in enumerate(mol.GetConfs()):
        for x in oechem.OEGetSDDataPairs(conf):
            if taglabel.lower() in x.GetTag().lower():
                sd_list.append(x.GetValue())
                break

    return sd_list


def read_check_input(in_file):
    """
    Read input file into an ordered dictionary.

    Parameters
    ----------
    in_file : string
        name of input file to match script

    Returns
    -------
    in_dict : OrderedDict
        dictionary from input file, where key is method and value is dictionary
        first entry should be reference method
        in sub-dictionary, keys are 'sdfile' and 'sdtag'

    """

    in_dict = collections.OrderedDict()

    # read input file
    contents = np.genfromtxt(in_file, delimiter=',', unpack=True,
        dtype='unicode', autostrip=True)

    # store each file's information in dictionary of dictionaries
    for i in range(len(contents[0])):
        in_dict[contents[0][i]] = {'sdfile': contents[1][i], 'sdtag': contents[2][i]}

    # check that each file exists before starting
    list1 = []
    for vals in in_dict.values():
        list1.append(os.path.isfile(vals['sdfile']))

    # if all elements are True then all files exist
    if all(list1):
        return in_dict
    else:
        print(list1)
        raise ValueError("One or more listed files not found")


def rdmol_from_oemol(oemol):
    """
    Create an RDKit molecule identical to the input OpenEye molecule.

    Reference
    ---------
    Written by Caitlin Bannan:
    https://gist.github.com/bannanc/810ccc4636b930a4522636baab1965a6

    May not be needed in newer openforcefield versions. See:
    https://github.com/openforcefield/openforcefield/issues/135

    Parameters
    ----------
    oemol : OEMol

    """
    #print("Starting OpenEye molecule: ", oechem.OEMolToSmiles(oemol))

    # start function
    rdmol = Chem.RWMol()

    # RDKit keeps bond order as a type instead using these values, I don't really understand 7,
    # I took them from Shuzhe's example linked above
    _bondtypes = {1: Chem.BondType.SINGLE,
                  1.5: Chem.BondType.AROMATIC,
                  2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE,
                  4: Chem.BondType.QUADRUPLE,
                  5: Chem.BondType.QUINTUPLE,
                  6: Chem.BondType.HEXTUPLE,
                  7: Chem.BondType.ONEANDAHALF,}

    # atom map lets you find atoms again
    map_atoms = dict() # {oe_idx: rd_idx}
    for oea in oemol.GetAtoms():
        oe_idx = oea.GetIdx()
        rda = Chem.Atom(oea.GetAtomicNum())
        rda.SetFormalCharge(oea.GetFormalCharge())
        rda.SetIsAromatic(oea.IsAromatic())

        # unlike OE, RDK lets you set chirality directly
        cip = oechem.OEPerceiveCIPStereo(oemol, oea)
        if cip == oechem.OECIPAtomStereo_S:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
        if cip == oechem.OECIPAtomStereo_R:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

        map_atoms[oe_idx] = rdmol.AddAtom(rda)

    # As discussed above, setting bond stereochemistry requires neighboring bonds
    # so we will store that information by atom index in this list
    stereo_bonds = list()
    # stereo_bonds will have tuples with the form (rda1, rda2, rda3, rda4, is_cis)
    # where rda[n] is an atom index for a double bond of form 1-2=3-4
    # and is_cis is a Boolean is True then onds 1-2 and 3-4 are cis to each other

    for oeb in oemol.GetBonds():
        # get neighboring rd atoms
        rd_a1 = map_atoms[oeb.GetBgnIdx()]
        rd_a2 = map_atoms[oeb.GetEndIdx()]

        # AddBond returns the total number of bonds, so addbond and then get it
        rdmol.AddBond(rd_a1, rd_a2)
        rdbond = rdmol.GetBondBetweenAtoms(rd_a1, rd_a2)

        # Assign bond type, which is based on order unless it is aromatic
        order = oeb.GetOrder()
        if oeb.IsAromatic():
            rdbond.SetBondType(_bondtypes[1.5])
            rdbond.SetIsAromatic(True)
        else:
            rdbond.SetBondType(_bondtypes[order])
            rdbond.SetIsAromatic(False)

        # If the bond has specified stereo add the required information to stereo_bonds
        if oeb.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            # OpenEye determined stereo based on neighboring atoms so get two outside atoms
            n1 = [n for n in oeb.GetBgn().GetAtoms() if n != oeb.GetEnd()][0]
            n2 = [n for n in oeb.GetEnd().GetAtoms() if n != oeb.GetBgn()][0]

            rd_n1 = map_atoms[n1.GetIdx()]
            rd_n2 = map_atoms[n2.GetIdx()]

            stereo = oeb.GetStereo([n1,n2], oechem.OEBondStereo_CisTrans)
            if stereo == oechem.OEBondStereo_Cis:
                print('cis')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, True))
            elif stereo == oechem.OEBondStereo_Trans:
                print('trans')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, False))

    # add bond stereochemistry:
    for (rda1, rda2, rda3, rda4, is_cis) in stereo_bonds:
        # get neighbor bonds
        bond1 = rdmol.GetBondBetweenAtoms(rda1, rda2)
        bond2 = rdmol.GetBondBetweenAtoms(rda3, rda4)

        # Since this is relative, the first bond always goes up
        # as explained above these names come from SMILES slashes so UP/UP is Trans and Up/Down is cis
        bond1.SetBondDir(Chem.BondDir.ENDUPRIGHT)
        if is_cis:
            bond2.SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        else:
            bond2.SetBondDir(Chem.BondDir.ENDUPRIGHT)

    # if oemol has coordinates (The dimension is non-zero)
    # add those coordinates to the rdmol
    if oechem.OEGetDimensionFromCoords(oemol) > 0:
        conformer = Chem.Conformer()
        oecoords = oemol.GetCoords()
        for oe_idx, rd_idx in map_atoms.items():
            (x,y,z) = oecoords[oe_idx]
            conformer.SetAtomPosition(rd_idx, Geometry.Point3D(x,y,z))
        rdmol.AddConformer(conformer)

    # Save the molecule title
    rdmol.SetProp("_Name", oemol.GetTitle())

    # Cleanup the rdmol
    # Note I copied UpdatePropertyCache and GetSSSR from Shuzhe's code to convert oemol to rdmol here:
    rdmol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(rdmol)
    # I added AssignStereochemistry which takes the directions of the bond set
    # and assigns the stereochemistry tags on the double bonds
    Chem.AssignStereochemistry(rdmol, force=False)

    #print("Final RDKit molecule: ", Chem.MolToSmiles(Chem.RemoveHs(rdmol), isomericSmiles=True))
    return rdmol.GetMol()

