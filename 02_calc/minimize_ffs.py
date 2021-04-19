#!/usr/bin/env python

"""
minimize_ffs.py

Charge molecules using AM1-BCC ELF10 scheme, then perform energy minimization
with OESzybki (MMFF94(S)) or OpenMM.

Supported force fields: GAFF, GAFF2, MMFF94, MMFF94S, any OpenFF FFXML file

Version: Nov 16 2019
"""

import os

from openeye import oechem, oequacpac, oeszybki
import simtk.openmm as mm
from simtk.openmm import app

import openmoltools
import parmed
from parmed import unit as u

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Molecule, Topology

from openforcefield.utils import structure

# eric zhang feb12 2021
import qcengine
import qcelemental as qcel
from qcelemental.models import AtomicInput, OptimizationInput
from qcelemental.models.common_models import Model
from qcelemental.models.procedures import QCInputSpecification

# ez mar4 2021
from openeye import oeff

# ez mar31 2021
from forcebalance.molecule import Molecule as fb_molecule
from forcebalance.molecule import get_rotate_translate
import numpy as np

def run_openmm(topology, system, positions):
    """
    Minimize molecule with specified topology, system, and positions
       using OpenMM. Return the positions of the optimized moelcule.

    Parameters
    ----------
    Topology:  OpenMM topology
    System:    OpenMM system
    Positions: OpenMM positions

    Returns
    -------
    concat_coords: list of positions ready to add to an OEMol
    energy       : energy in kcal/mol

    """

    # need to create integrator but don't think it's used
    integrator = mm.LangevinIntegrator(
            300.0 * u.kelvin,
            1.0 / u.picosecond,
            2.0 * u.femtosecond)

    # create simulation object then minimize
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=5.0E-9, maxIterations=1500)

    # get minimized positions
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    positions = positions/u.angstroms
    coordlist = list()
    for atom_coords in positions:
        coordlist += [i for i in atom_coords]

    # get minimized energy
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    energy = energy.value_in_unit(u.kilocalories_per_mole)

    return coordlist, energy


def charge_mol(mol):
    """
    Charge molecule using AM1-BCC ELF10 scheme.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule

    Returns
    -------
    chg_mol : OpenEye single-conformer molecule with charges

    """

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


def charge_conf(chg_mol, conf):
    """
    Apply charges from chg_mol onto conf.

    Parameters
    ----------
    chg_mol : OpenEye conformer with charges
    conf    : OpenEye conformer without charges

    Returns
    -------
    chg_conf : OpenEye conformer with charges

    """

    # make copy of the input mol
    chg_conf = oechem.OEGraphMol(conf)

    # source: https://github.com/MobleyLab/SolvationToolkit/blob/master/solvationtoolkit/mol2tosdf.py
    for atom, atomCharged in zip(chg_conf.GetAtoms(), chg_mol.GetAtoms()):
        atom.SetPartialCharge( atomCharged.GetPartialCharge() )

    return chg_conf


def min_mmff94x(mol, ofs, mmff94s=False):
    """
    Minimize the mol with MMFF94 or MMFF94S force field.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream
    mmff94s : Boolean
        True to minimize with MMFF94S

    """

    # make copy of the input mol
    oe_mol = oechem.OEGraphMol(mol)

    # set general energy options along with the run type specification
    optSzybki = oeszybki.OESzybkiOptions()
    optSzybki.SetSolventModel(oeszybki.OESolventModel_NoSolv)
    optSzybki.SetOptimizerType(oeszybki.OEOptType_BFGS)

    # minimize with input charges not mmff94(s) charges
    # https://docs.eyesopen.com/toolkits/python/szybkitk/examples.html#optimization-of-all-conformers-of-a-ligand
    optSzybki.GetSolventOptions().SetChargeEngine(oequacpac.OEChargeEngineNoOp())

    # set the particular force field
    if mmff94s:
        sdlabel = "MMFF94S"
        optSzybki.SetForceFieldType(oeszybki.OEForceFieldType_MMFF94S)
    else:
        sdlabel = "MMFF94"
        optSzybki.SetForceFieldType(oeszybki.OEForceFieldType_MMFF94)

    # generate minimization engine
    szOpt = oeszybki.OESzybki(optSzybki)

    # make object to hold szybki results
    szResults = oeszybki.OESzybkiResults()

    # perform minimization
    if not szOpt(oe_mol, szResults):
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> MMFF94x minimization failed for %s\n' % smilabel )
    energy = szResults.GetTotalEnergy()

    # save geometry, save energy as tag, write mol to file
    oechem.OESetSDData(oe_mol, f"Energy {sdlabel}", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)


def min_gaffx(mol, ofs, gaff2=False):
    """
    Minimize the mol with GAFF or GAFF2 force field.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream
    gaff2 : Boolean
        True to minimize with GAFF2

    """

    # make copy of the input mol
    oe_mol = oechem.OEMol(mol)
    title = oe_mol.GetTitle()
    smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")

    # get unique tmp filename based on smiles string
    # truncate to 200 characters for linux limit, hopefully still unique
    int_from_text = int.from_bytes(smilabel.encode(), 'little')
    short_int = int(str(int_from_text)[:200])

    # assign ambertools filenames
    tmol2 = f'{short_int}_t.mol2'
    gmol2 = f'{short_int}_g.mol2'
    frc =   f'{short_int}.frcmod'
    prm =   f'{short_int}.prmtop'
    inp =   f'{short_int}.inpcrd'

    # generate tripos mol2 file
    openmoltools.openeye.molecule_to_mol2(oe_mol, tmol2)

    if gaff2:
        invar = 'gaff2'
        leaprc = 'leaprc.gaff2'
        sdlabel = 'GAFF2'
    else:
        invar = 'gaff'
        leaprc = 'leaprc.gaff'
        sdlabel = 'GAFF'

    try:
        # generate gaff mol2 file and frcmod files
        openmoltools.amber.run_antechamber(title, tmol2, charge_method=None,
            gaff_mol2_filename = gmol2, frcmod_filename = frc,
            gaff_version = invar)
    except Exception as e:
        # earlier smilabel seems to be missing
        smilabel = oechem.OEGetSDData(mol, "SMILES QCArchive")
        print( ' >>> Antechamber failed to produce GAFF mol2 file: '
               f'{title} {smilabel}: {e}')
        return

    # generate gaff inpcrd and prmtop files
    openmoltools.amber.run_tleap(title, gaff_mol2_filename = gmol2,
        frcmod_filename = frc, prmtop_filename = prm, inpcrd_filename = inp,
        leaprc = leaprc)

    # load input files and create parmed system
    parm = parmed.load_file(prm, inp)
    topology = parm.topology
    system = parm.createSystem(nonbondedMethod=app.NoCutoff)
    positions = parm.positions

    # minimize structure
    newpos, energy = run_openmm(topology, system, positions)

    # save geometry, save energy as tag, write mol to file
    oe_mol.SetCoords(oechem.OEFloatArray(newpos))

    oechem.OESetSDData(oe_mol, f"Energy {sdlabel}", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    # remove gaff-related files
    [os.remove(f) for f in [tmol2, gmol2, frc, prm, inp]]

    return


def min_ffxml(mol, ofs, ffxml):
    """
    Minimize the mol with force field input from FFXML file.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream
    ffxml : string
        name of FFXML file

    """

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
               f'{oe_mol.GetTitle()} {smilabel}: {e}')
        return

    positions = structure.extractPositionsFromOEMol(oe_mol)

    # minimize structure with ffxml
    newpos, energy = run_openmm(topology, system, positions)

    # save geometry, save energy as tag, write mol to file
    oe_mol.SetCoords(oechem.OEFloatArray(newpos))
    oechem.OESetSDData(oe_mol, "Energy FFXML", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    return

# eric zhang feb12 2021
def get_xtb(mol, ofs):
    """
    calculate xtb energy of a single-conformer molecule.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream

    """

    # make copy of the input mol
    # for some reason this has to be oemol instead of oegraphmol
    oe_mol = oechem.OEMol(mol)

    energy = None

    try:
        # create openforcefield molecule ==> prone to triggering Exception
        off_mol = Molecule.from_openeye(oe_mol, allow_undefined_stereo=True)

        # create molecule for use in qcengine
        qc_mol = off_mol.to_qcschema()

        # set up qcengine
        xtb_model = Model(method="gfn2-xtb", basis=None)
        qc_task = AtomicInput(molecule=qc_mol, driver="energy", model=xtb_model)

        result = qcengine.compute(input_data=qc_task, program="xtb")

        # xtb returns energy in hartree
        hartree_to_kcalpmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

        energy = result.return_result * hartree_to_kcalpmol

    except Exception as e:
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> xtb calculation: something went wrong: '
               f'{oe_mol.GetTitle()} {smilabel}: {e}')
        return

    # save energy as tag, write mol to file
    oechem.OESetSDData(oe_mol, "Energy xtb", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    return

# eric zhang feb12 2021
def get_am1(mol, ofs):
    """
    calculate am1 energy of a single-conformer molecule.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream

    """

    # make copy of the input mol
    oe_mol = oechem.OEGraphMol(mol)

    energy = None

    try:
        # set up oeAM1 calculator
        calc = oequacpac.OEAM1()

        # make space for the calculation result
        result = oequacpac.OEAM1Results()

        success = calc.CalcAM1(result, oe_mol)
        if success == False:
            print("calcAM1 failed to return a result")
            raise Exception

        # am1 returns energy in kcal/mol
        energy = result.GetEnergy()
        
    except Exception as e:
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> am1 calculation: something went wrong: '
               f'{oe_mol.GetTitle()} {smilabel}: {e}')
        return

    # save energy as tag, write mol to file
    oechem.OESetSDData(oe_mol, "Energy am1", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    return

# eric zhang feb22 2021
def min_xtb(mol, ofs):
    """
    calculate xtb energy of a single-conformer molecule.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream

    """

    # make copy of the input mol
    # for some reason this has to be oemol instead of oegraphmol
    oe_mol = oechem.OEMol(mol)

    energy = None
    geom = None

    try:
        # create openforcefield molecule ==> prone to triggering Exception
        off_mol = Molecule.from_openeye(oe_mol, allow_undefined_stereo=True)

        # create molecule for use in qcengine
        qc_mol = off_mol.to_qcschema()

        # set up qcengine
        xtb_model = Model(method="gfn2-xtb", basis=None)

        geometric_input = OptimizationInput(
                            initial_molecule=qc_mol,
                            input_specification=QCInputSpecification(model=xtb_model),
                            keywords={"coordsys": "tric", "maxiter": 300, "program": "xtb"}
                        )
        opt_result = qcengine.compute_procedure(input_data=geometric_input, procedure="geometric",
		local_options = {"ncores":1})
        opt_energy = opt_result.trajectory[-1].properties.return_energy

        # xtb returns energy in hartree
        hartree_to_kcalpmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
        b2a = qcel.constants.conversion_factor("bohr", "angstrom")

        energy = opt_energy * hartree_to_kcalpmol
        geom = opt_result.final_molecule.geometry * b2a

    except AttributeError as ae:
        print(f"xtb did not converge for: {mol.GetTitle()}")
        return
    except Exception as e:
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> xtb calculation: something went wrong: '
               f'{oe_mol.GetTitle()} {smilabel}: {e}')
        return

    flatgeom = []
    for a in geom:
        flatgeom.extend(a)
    # save geometry, save energy as tag, write mol to file
    oe_mol.SetCoords(flatgeom)

    # change this to True to align
    if False:
        alignedgeom = align(mol, oe_mol, "xtb")

        # save geometry, save energy as tag, write mol to file
        oe_mol.SetCoords(alignedgeom)

    oechem.OESetSDData(oe_mol, "Energy xtb", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    return

def align(mol, opt_mol, method):
    # alignment
    cc = mol.GetCoords()
    a = []
    for i in range(len(cc)):
        a.append(cc[i])
    ogcd = np.array(a)
    
    cc = opt_mol.GetCoords()
    a = []
    for i in range(len(cc)):
        a.append(cc[i])
    geom = np.array(a)

    xyzs = [ogcd.copy(), geom.copy()]
    _align(xyzs)

    alignedgeom = xyzs[1]
    flatageom = []
    for a in alignedgeom:
        flatageom.extend(a)

    return flatageom

def _align(xyzs):
    """ Align molecules.
    adapted from leeping/forcebalance/src/molecule.py
    """
    xyz1 = xyzs[0]

    for index2, xyz2 in enumerate(xyzs):
        if index2 == 0: continue
        xyz2 -= xyz2.mean(0)
        ref = 0

        tr, rt = get_rotate_translate(xyz2,xyzs[ref])

        xyz2 = np.dot(xyz2, rt) + tr
        xyzs[index2] = xyz2


# eric zhang mar2 2021
def min_am1(mol, ofs):
    """
    calculate am1 energy of a single-conformer molecule.

    Parameters
    ----------
    mol : OpenEye single-conformer molecule
    ofs : OpenEye output filestream

    """

    # make copy of the input mol
    oe_mol = oechem.OEMol(mol)

    energy = None

    try:
        am1 = oequacpac.OEAM1()  # this is a low-level api class that will not perform any optimization on its own
        results = oequacpac.OEAM1Results()  # this is where you can access the energy as well as the bond orders

        # get vector coords
        vecCoords = oechem.OEDoubleArray(3 * oe_mol.NumAtoms())
        fcoords = oechem.OEDoubleArray(3 * oe_mol.NumAtoms())
        cdict = oe_mol.GetCoords(vecCoords)

        # somehow this is needed to prevent segfault on optimizer( ... )
        am1.CalcAM1(results, oe_mol)

        # Optimize the geometry using AM1
        optimizer = oeff.OEBFGSOpt()
        
        optimizer(am1, vecCoords, fcoords)  # (MolFunc1, Input Coords, Output Coords)

        # Set the optimized coords on the conf
        oe_mol.SetCoords(fcoords)

        # Perform the AM1 calculation again on the optimized conf
        am1.CalcAM1(results, oe_mol)
        
        energy = results.GetEnergy()
       
    except Exception as e:
        smilabel = oechem.OEGetSDData(oe_mol, "SMILES QCArchive")
        print( ' >>> am1 calculation: something went wrong: '
               f'{oe_mol.GetTitle()} {smilabel}: {e}')
        return

    # change this to True to align
    if False:
        acoords = align(mol, oe_mol, "am1")
        oe_mol.SetCoords(acoords)
    # save energy as tag, write mol to file
    oechem.OESetSDData(oe_mol, "Energy am1", str(energy))
    oechem.OEWriteConstMolecule(ofs, oe_mol)

    return






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


def main(infile, outfile, ffxml, minimizer):
    """
    Process molecules and handle minimization.

    Parameters
    ----------
    infile : string
        name of file with input molecules
    outfile : string
        name of file for output molecules
    ffxml : string
        name of FFXML file, may be None if not provided
    minimizer : string
        name of the force field for which to do minimizations
    """

    # open multi-molecule, multi-conformer file
    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(infile):
        raise FileNotFoundError(f"Unable to open {infile} for reading")
    mols = ifs.GetOEMols()

    # open an outstream file
    ofs = oechem.oemolostream()
    if os.path.exists(outfile):
        raise FileExistsError("Output file {} already exists in {}".format(
            outfile, os.getcwd()))
    if not ofs.open(outfile):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % outfile)

    # minimize with openforcefield ffxml file
    for i, mol in enumerate(mols):

        # perceive stereochemistry for mol
        oechem.OEPerceiveChiral(mol)
        oechem.OEAssignAromaticFlags(mol, oechem.OEAroModel_MDL)

        # assign charges to copy of mol
        # note that chg_mol does NOT have conformers
        try:
            chg_mol = charge_mol(mol)

        except RuntimeError:
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
                title = mol.GetTitle()
                smilabel = oechem.OEGetSDData(mol, "SMILES QCArchive")
                print( ' >>> Charge assignment failed due to unspecified '
                      f'stereochemistry {title} {smilabel}')
                continue

        for j, conf in enumerate(mol.GetConfs()):

            # perceive sterochemistry for conf coordinates
            oechem.OE3DToInternalStereo(conf)

            # assign charges to the conf itself
            chg_conf = charge_conf(chg_mol, conf)

            if minimizer == 'ffxml':
                # minimize with parsley (charges set by ff not used from conf)
                min_ffxml(chg_conf, ofs, ffxml)

            elif minimizer == 'mmff94':
                # minimize with mmff94
                min_mmff94x(chg_conf, ofs, mmff94s=False)

            elif minimizer == 'mmff94s':
                # minimize with mmff94S
                min_mmff94x(chg_conf, ofs, mmff94s=True)

            elif minimizer == 'gaff':
                # minimize with gaff
                min_gaffx(chg_conf, ofs, gaff2=False)

            elif minimizer == 'gaff2':
                # minimize with gaff2
                min_gaffx(chg_conf, ofs, gaff2=True)

            elif minimizer == 'xtb':
                # calculate xtb
                min_xtb(chg_conf, ofs)

            elif minimizer == 'am1':
                # calculate am1
                min_am1(chg_conf, ofs)

    ifs.close()
    ofs.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile",
            help="Input molecule file")

    parser.add_argument("-o", "--outfile",
            help="Output molecule file")

    parser.add_argument("-m", "--minimizer",
            help="Force field minimization to undertake. "
                 "Options include: gaff gaff2 mmff94 mmff94s ffxml")

    parser.add_argument("-f", "--ffxml",
            help="Open force field ffxml file",
            default=None)

    args = parser.parse_args()

    if args.minimizer not in ['gaff', 'gaff2', 'mmff94', 'mmff94s', 'ffxml', 'xtb', 'am1']:
        raise ValueError('Please specify one of the following: '
                         'gaff gaff2 mmff94 mmff94s ffxml')
    if args.minimizer == 'ffxml' and not os.path.isfile(args.ffxml):
        raise ValueError('Please specify ffxml file for minimizer of \'ffxml\'')

    main(args.infile, args.outfile, args.ffxml, args.minimizer)

