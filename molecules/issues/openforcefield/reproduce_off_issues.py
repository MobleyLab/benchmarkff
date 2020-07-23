
from openeye import oechem
import simtk.openmm as mm
from simtk.openmm import app

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Molecule, Topology

def min_ffxml(mol, ffxml):

    # make copy of the input mol
    oe_mol = oechem.OEGraphMol(mol)

    try:
        # create openforcefield molecule ==> prone to triggering Exception
        off_mol = Molecule.from_openeye(oe_mol)

        # load in force field
        ff = ForceField(ffxml)

        # create components for OpenMM system
        topology = Topology.from_molecules(molecules=[off_mol])

        # create openmm system ==> prone to triggering Exception
        #system = ff.create_openmm_system(topology, charge_from_molecules=[off_mol])
        system = ff.create_openmm_system(topology)

    except Exception as e:
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> openforcefield failed to create OpenMM system: '
               f"'{oe_mol.GetTitle()}' '{smilabel}'")
        print(f"{e}\n")
        return

    print(" >>> successful OpenMM system creation for openforcefield "
         f"mol \"{oe_mol.GetTitle()}\"")

def main(infile, ffxml):

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

        for j, conf in enumerate(mol.GetConfs()):

            # perceive sterochemistry for conf coordinates
            oechem.OE3DToInternalStereo(conf)

            min_ffxml(conf, ffxml)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile",
            help="Input molecule file")

    parser.add_argument("-f", "--ffxml",
            help="Open force field ffxml file",
            default=None)

    args = parser.parse_args()

    main(args.infile, args.ffxml)

