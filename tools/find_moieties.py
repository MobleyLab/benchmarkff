
import sys
from openeye.oechem import *

"""
Count the number of specific chemical moieties in a group of structures.
Usage: python find_moieties.py input.sdf output.sdf
Reference: https://ctr.fandom.com/wiki/Ring_counts_in_a_SMILES_file
"""

def find_moieties(infile, outfile):

    # read input file
    ifs = oemolistream()
    ifs.open(infile)

    # open output sdf file
    ofs = oemolostream()
    ofs.open(outfile)

    # open output text files
    file_s = open('moiety_s.dat', 'w')
    file_cn = open('moiety_cn.dat', 'w')

    # loop through conformers
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():

            # create a copy of the molecule and copy SD tags
            newconf = OEGraphMol(conf)
            OECopySDData(conf, newconf)

            # parse smiles string in SD tag
            mytag = OEGetSDData(conf, 'SMILES QCArchive')
            count_s  = mytag.count('S')
            count_p  = mytag.count('P')
            count_cn = mytag.count('C#N')
            count_nn = mytag.count('N/N')

            # count rings
            num_components, component_membership = OEDetermineComponents(mol)
            num_rings = mol.NumBonds() - mol.NumAtoms() + num_components

            # write text files
            if count_s > 0:
                file_s.write(mytag + "\n")
            if count_cn > 0:
                file_cn.write(mytag + "\n")

            # store data as SD tag and write out the copied mol
            OESetSDData(newconf, "Number of rings", str(num_rings))
            OESetSDData(newconf, "Number of S",     str(count_s))
            OESetSDData(newconf, "Number of P",     str(count_p))
            OESetSDData(newconf, "Number of C#N",   str(count_cn))
            OESetSDData(newconf, "Number of N/N",   str(count_nn))
            OEWriteConstMolecule(ofs, newconf)

    ifs.close()
    ofs.close()
    file_s.close()
    file_cn.close()

if __name__ == "__main__":
    find_moieties(sys.argv[1], sys.argv[2])
