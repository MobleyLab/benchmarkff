
import sys
from openeye.oechem import *

"""
Count the number of rings in each structure.
Usage: python find_num_rings.py input.sdf > output.dat
Reference: https://ctr.fandom.com/wiki/Ring_counts_in_a_SMILES_file
"""

def find_num_rings(infile):

    # read input file
    ifs = oemolistream()
    ifs.open(infile)

    # loop through conformers and count rings
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():
            num_components, component_membership = OEDetermineComponents(mol)
            num_rings = mol.NumBonds() - mol.NumAtoms() + num_components
            print(f"{conf.GetTitle()}\t{num_rings}")

if __name__ == "__main__":
    find_num_rings(sys.argv[1])
