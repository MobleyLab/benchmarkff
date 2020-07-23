
import sys
import openeye.oechem as oechem

"""
Find all molecules with some chemical moiety by parsing SMILES data in SD tag.
Usage: python find_string_tag.py input.sdf > output.dat
"""

def find_string_tag(infile):
    # read input file
    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(infile):
        oechem.OEThrow.Warning("Unable to open input file for reading")
    # loop through and evaluate tags
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():
            mytag = oechem.OEGetSDData(conf, 'SMILES QCArchive')
            count1 = mytag.count('S')
            count2 = mytag.count('P')
            count3 = mytag.count('C#N')
            count4 = mytag.count('N/N')
            print(f"{conf.GetTitle()}\t{count1}\t{count2}\t{count3}\t{count4}")

if __name__ == "__main__":
    find_string_tag(sys.argv[1])
