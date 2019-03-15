from simtk import openmm, unit
from simtk.openmm import app
import metadynamics as mtd
import numpy as np

ffxml_filenames = ['amber99sbildn.xml', 'tip3p.xml']

# pressure = 1.0 * unit.atmospheres
temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds

# Read equilibrated pdb
pdb_filename = 'equilibrated.pdb'
print('Loading %s' % pdb_filename)
pdb = app.PDBFile(pdb_filename)

print("Loading forcefield: %s" % ffxml_filenames)
forcefield = app.ForceField(*ffxml_filenames)

# Create the system
print('Creating OpenMM System...')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds, removeCMMotion=True)

# Add forces
print("Adding CV force...")
rmsd = openmm.RMSDForce(pdb.positions, list(range(702)))
cv_force = openmm.CustomCVForce("rmsd")
cv_force.addCollectiveVariable('rmsd', rmsd)
bv = mtd.BiasVariable(cv_force, 0, 2.5, 0.1, False)

# Add a barostat
# print('Adding barostat...')
# barostat = openmm.MonteCarloBarostat(pressure, temperature)
# system.addForce(barostat)

# Create simulation
print("Creating Simulation...")
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
meta = mtd.Metadynamics(system, [bv], temperature, 3, 1.2*unit.kilojoules_per_mole, 100)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

print("Appending reporters...")
simulation.reporters.append(app.DCDReporter('mtd_0.dcd', 50000))
simulation.reporters.append(app.StateDataReporter('mtd_0.out', 1000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=75000000, separator='\t'))

print("Running 150 ns of metadynamics...")    
meta.step(simulation, 75000000)

print("Done!")
