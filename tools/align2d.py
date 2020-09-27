#!/usr/bin/env python

"""
align2d.py

Objective: From two multi-molecule files with similar structures between
  both files, align the 2D representations and write out a PDF file.

Version:    Mar 6 2020

Note: This script can be easily modified to write out 2 separate PDF files
  by calling prep_pdf_writer() 2x to get report1 and report2, generating
  cell1 and cell2 from the separate report objects, and creating a
  second ofs to write out the second report object.

"""

from openeye import oechem
from openeye import oedepict

def prep_pdf_writer():

    itf = oechem.OEInterface()
    ropts = oedepict.OEReportOptions()
    oedepict.OESetupReportOptions(ropts, itf)
    ropts.SetFooterHeight(25.0)
    report = oedepict.OEReport(ropts)

    popts = oedepict.OEPrepareDepictionOptions()
    oedepict.OESetupPrepareDepictionOptions(popts, itf)

    dopts = oedepict.OE2DMolDisplayOptions()
    oedepict.OESetup2DMolDisplayOptions(dopts, itf)
    dopts.SetDimensions(report.GetCellWidth(), report.GetCellHeight(), oedepict.OEScale_AutoScale)

    return popts, dopts, report


def align2d(file1, file2):

    atomexpr = oechem.OEExprOpts_AtomicNumber | oechem.OEExprOpts_RingMember
    bondexpr = oechem.OEExprOpts_RingMember

    ifs1 = oechem.oemolistream(file1)
    ifs2 = oechem.oemolistream(file2)
    ifs1.SetConfTest(oechem.OEAbsCanonicalConfTest())
    ifs2.SetConfTest(oechem.OEAbsCanonicalConfTest())

    popts, dopts, report = prep_pdf_writer()

    for mol1, mol2 in zip(ifs1.GetOEMols(), ifs2.GetOEMols()):
        oechem.OESuppressHydrogens(mol1)
        oechem.OESuppressHydrogens(mol2)
        oechem.OEGenerate2DCoordinates(mol2)
        ss = oechem.OESubSearch(mol2, atomexpr, bondexpr)

        oechem.OEPrepareSearch(mol1, ss)
        alignres = oedepict.OEPrepareAlignedDepiction(mol1, ss)

        if not alignres.IsValid():
            oechem.OEThrow.Error("Substructure is not found in input molecule!")

        cell1 = report.NewCell()
        cell2 = report.NewCell()
        oedepict.OEPrepareDepiction(mol1, popts)
        oedepict.OEPrepareDepiction(mol2, popts)
        disp1 = oedepict.OE2DMolDisplay(mol1, dopts)
        disp2 = oedepict.OE2DMolDisplay(mol2, dopts)
        oedepict.OERenderMolecule(cell1, disp1)
        oedepict.OERenderMolecule(cell2, disp2)

    ofs = oechem.oeofstream()
    if not ofs.open('output.pdf'):
        oechem.OEThrow.Fatal("Cannot open output file!")
    oedepict.OEWriteReport(ofs, "pdf", report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--file1",
            help="First input molecule file")
    parser.add_argument("--file2",
            help="Second input molecule file")

    args = parser.parse_args()
    align2d(args.file1, args.file2)
