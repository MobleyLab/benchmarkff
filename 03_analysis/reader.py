#!/usr/bin/env python

"""
reader.py

Functions to parse input files or OEMols from those files.

By:      Victoria T. Lim
Version: Dec 11 2019

"""

import os
import copy
import collections

import openeye.oechem as oechem
from rdkit import Chem, Geometry

def read_mols(infile, mol_slice=None):
    """
    Open a molecule file and return molecules and conformers as OEMols.
    Provide option to slice the mols to return only a chunk from the
    specified indices.

    Parameters
    ----------
    infile : string
        name of input file with molecules
    mol_slice : list[int]
        list of indices from which to slice mols generator for read_mols
        [start, stop, step]

    Returns
    -------
    mols : OEMols

    """
    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(infile):
        raise FileNotFoundError(f"Unable to open {infile} for reading")
    mols = ifs.GetOEMols()

    if mol_slice is not None:
        if len(mol_slice) != 3 or mol_slice[0] >= mol_slice[1] or mol_slice[2] <= 0:
            raise ValueError("Check input to mol_slice. Should have len 3, "
                "start value < stop value, step >= 1.")

        # TODO more efficient. can't itertools bc lost mol info (name, SD) after next()
        # adding copy/deepcopy doesnt work on generator objects
        # also doesn't work to convert generator to list then slice list
        #mols = itertools.islice(mols, mol_slice[0], mol_slice[1], mol_slice[2])
        #mlist = mlist[mol_slice[0]:mol_slice[1]:mol_slice[2]]

        def incrementer(count, mols, step):
            if step == 1:
                count += 1
                return count
            # use step-1 because for loop already increments once
            for j in range(step-1):
                count += 1
                next(mols)
            return count

        mlist = []
        count = 0
        for i, m in enumerate(mols):

            if count >= mol_slice[1]:
                return mlist
            elif count < mol_slice[0]:
                count += 1
                continue
            else:
                # important to append copy else still linked to orig generator
                mlist.append(copy.copy(m))
                try:
                    count = incrementer(count, mols, mol_slice[2])
                except StopIteration:
                    return mlist

        return mlist

    return mols


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

def read_check_input(infile):
    """
    Read input file into an ordered dictionary.

    Parameters
    ----------
    infile : string
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
    with open(infile) as f:
        for line in f:

            # skip commented lines or empty lines
            if line.startswith('#'):
                continue
            dataline = [x.strip() for x in line.split(',')]
            if dataline == ['']:
                continue

            # store each file's information in dictionary of dictionaries
            in_dict[dataline[0]] = {'sdfile': dataline[1], 'sdtag': dataline[2]}

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

    aro_bond = 0
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

