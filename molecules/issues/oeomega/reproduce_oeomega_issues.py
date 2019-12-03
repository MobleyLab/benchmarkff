
from openeye import oechem
import openmoltools

def charge_mol(mol):

    # make copy of the input mol
    oe_mol = oechem.OEMol(mol)

    # charging with openmoltools wrapper generates 800 confs per mol for ELF
    # https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html
    # openmoltools returns a charged copy of the mol
    chg_mol = openmoltools.openeye.get_charges(
        oe_mol,
        normalize=False, # already assigned aromatic flags and have hydrogens
        keep_confs=None, # keep the input conformation only
    )
    return chg_mol

def find_unspecified_stereochem(mol):
    """
    Debugging for frequent stereochem issues.
    https://docs.eyesopen.com/toolkits/python/oechemtk/stereochemistry.html
    """

    for atom in mol.GetAtoms():
        chiral = atom.IsChiral()
        stereo = oechem.OEAtomStereo_Undefined
        if atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
            v = []
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, oechem.OEAtomStereo_Tetrahedral)

        if chiral or stereo != oechem.OEAtomStereo_Undefined:
            print("Atom:", atom.GetIdx(), "chiral=", chiral, "stereo=", end=" ")
            if stereo == oechem.OEAtomStereo_RightHanded:
                print("right handed")
            elif stereo == oechem.OEAtomStereo_LeftHanded:
                print("left handed")
            else:
                print("undefined")
    # =========================================================================
    for bond in mol.GetBonds():
        chiral = bond.IsChiral()
        if chiral and bond.GetOrder() == 2 and not bond.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            print("atoms of UNSPECIFIED chiral bond: ", bond.GetBgn().GetIdx(), bond.GetEnd().GetIdx())

def main(infile):

    # open multi-molecule, multi-conformer file
    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(infile):
        raise FileNotFoundError(f"Unable to open {infile} for reading")
    mols = ifs.GetOEMols()

    for i, mol in enumerate(mols):

        # perceive stereochemistry for mol
        oechem.OEPerceiveChiral(mol)
        oechem.OEAssignAromaticFlags(mol, oechem.OEAroModel_MDL)

        # assign charges to copy of mol
        # note that chg_mol does NOT have conformers
        try:
            chg_mol = charge_mol(mol)

        except RuntimeError:
            find_unspecified_stereochem(mol)

            # perceive stereochem
            #find_unspecified_stereochem(mol)
            oechem.OE3DToInternalStereo(mol)

            # reset perceived and call OE3DToBondStereo, since it may be missed
            # by OE3DToInternalStereo if it thinks mol is flat
            mol.ResetPerceived()
            oechem.OE3DToBondStereo(mol)

            try:
                chg_mol = charge_mol(mol)
                print(f'fixed stereo: {mol.GetTitle()}')
            except RuntimeError:
                find_unspecified_stereochem(mol)

                title = mol.GetTitle()
                smilabel = oechem.OEGetSDData(mol, "SMILES QCArchive")
                print( ' >>> Charge assignment failed due to unspecified '
                      f'stereochemistry {title} {smilabel}')


                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile",
            help="Input molecule file")

    args = parser.parse_args()

    main(args.infile)

