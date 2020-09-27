#!/usr/bin/env python

"""
get_by_tag.py

Purpose: From a multi-molecule, multi-conformer SDF file, either extract to
KEEP or extract to REMOVE certain conformers based on the value in a unique
SD tag.

Examples:
 - python get_by_tag.py -i trim1.sdf -s "SMILES QCArchive" -e -list  gaff_outliers.txt   -o trim2.sdf
 - python get_by_tag.py -i file1.sdf -s some_sd_tag           -title tagvalue1 tagvalue2 -o output.sdf

"""

import re
import sys
import os
from openeye.oechem import *

def get_mols(infnames, outfname, conf_id_tag, nameset, exclude=False):
    ofs = oemolostream()
    if not ofs.open(outfname):
        OEThrow.Fatal("Unable to open %s for writing" % outfname)

    for i, fname in enumerate(infnames):
        print(fname)
        ifs = oemolistream()
        ifs.SetConfTest(OEAbsCanonicalConfTest())
        if not ifs.open(fname):
            OEThrow.Fatal("Unable to open %s for reading" % fname)

        for imol in ifs.GetOEMols():
            for conf in imol.GetConfs():
                # skip selected molecules and write out others
                if exclude:
                    if OEGetSDData(conf, conf_id_tag) in nameset:
                        continue
                    else:
                        OEWriteConstMolecule(ofs, conf)
                # write selected molecules and skip the others
                else:
                    if OEGetSDData(conf, conf_id_tag) in nameset:
                        OEWriteConstMolecule(ofs, conf)
                    else:
                        continue

def natural_key(astr):
    """https://stackoverflow.com/questions/34518/natural-sorting-algorithm/34528"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', astr[0])]

def sort_by_title(ifs, ofs):
    moldb = OEMolDatabase(ifs)

    titles = [(t, i) for i, t in enumerate(moldb.GetTitles())]
    titles.sort(key=natural_key)

    indices = [i for t, i in titles]

    moldb.Order(indices)
    moldb.Save(ofs)


Interface = """
!BRIEF -i <infile1> [<infile2>...] -o <outfile>
!PARAMETER -i
  !ALIAS -in
  !TYPE string
  !LIST true
  !REQUIRED true
  !BRIEF input file name(s)
!END
!PARAMETER -o
  !ALIAS -out
  !TYPE string
  !REQUIRED true
  !BRIEF output file name
!END
!PARAMETER -s
  !ALIAS -sdtag
  !TYPE string
  !REQUIRED true
  !BRIEF SD tag which corresponds to the input data to get or exclude
!END
!PARAMETER -title
  !ALIAS -t
  !TYPE string
  !LIST true
  !BRIEF Single or space-separated list of titles to get or exclude
!END
!PARAMETER -list
  !ALIAS -l
  !TYPE string
  !BRIEF List file of mol titles to get or exclude
!END
!PARAMETER -e
  !ALIAS -exclude
  !TYPE bool
  !DEFAULT false
  !BRIEF True to EXCLUDE selected mols from file, False to GET selected mols
!END
"""


def main(argv=[__name__]):
    itf = OEInterface(Interface, argv)

    # collect names
    nameset = set()
    if itf.HasString("-list"):
        try:
            lfs = open(itf.GetString("-list"))
        except IOError:
            OEThrow.Fatal("Unable to open %s for reading" % itf.GetString("-list"))
        for name in lfs.readlines():
            name = name.strip()
            nameset.add(name)
    elif itf.HasString("-title"):
        for t in itf.GetStringList("-title"):
            nameset.add(t)

    # iterate over conformers
    print(itf.GetString("-s"))
    if itf.GetBool("-e"):
        get_mols(itf.GetStringList("-i"), '____temp____.sdf', itf.GetString("-s"), nameset, True)
    else:
        get_mols(itf.GetStringList("-i"), '____temp____.sdf', itf.GetString("-s"), nameset)

    # sort molecules by title
    sort_by_title('____temp____.sdf', itf.GetString("-o"))

    # remove temp files after sorting
    os.remove('____temp____.sdf')
    os.remove(itf.GetString("-o") + '.idx')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
