from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
from openmmtools.integrators import LangevinIntegrator
import argparse
import time
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('model_index', type=int)
args = parser.parse_args()
model_index = args.model_index

ffxml_filenames = ['amber99sbildn.xml', 'tip3p.xml']

pressure = 1.0 * unit.atmospheres
temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picoseconds
timestep = 5.0 * unit.femtoseconds
hydrogen_mass = 3 * unit.amu
splitting = 'V R O R V'
nsteps = 500 # 2.5 ps
niterations = 4000 # 10 ns

equilibrated_pdb_filename = '%s/equilibrated_hmr.pdb' % model_index
system_xml_filename = '%s/system_hmr.xml' % model_index
integrator_xml_filename = '%s/integrator_hmr.xml' % model_index
state_xml_filename = '%s/state_hmr.xml' % model_index

# Read equilibrated pdb
pdb_filename = '%s/equilibrated.pdb' % model_index
print('Loading %s' % pdb_filename)
pdb = app.PDBFile(pdb_filename)

print("Loading forcefield: %s" % ffxml_filenames)
forcefield = app.ForceField(*ffxml_filenames)

# Create the system
print('Creating OpenMM System...')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds, removeCMMotion=False, hydrogenMass=hydrogen_mass)

# Add a barostat
print('Adding barostat...')
barostat = openmm.MonteCarloBarostat(pressure, temperature)
system.addForce(barostat)

# Serialize integrator
print('Serializing integrator to %s' % integrator_xml_filename)
integrator = LangevinIntegrator(temperature, collision_rate, timestep, splitting)
with open(integrator_xml_filename, 'w') as outfile:
    xml = openmm.XmlSerializer.serialize(integrator)
    outfile.write(xml)
    
# Prepare context
print('Preparing context...')
context = openmm.Context(system, integrator)
context.setPositions(pdb.positions)

# Equilibrate
print('Equilibrating...')
initial_time = time.time()
for iteration in progressbar.progressbar(range(niterations)):
    integrator.step(nsteps)
elapsed_time = (time.time() - initial_time) * unit.seconds
simulation_time = niterations * nsteps * timestep
print('    Equilibration took %.3f s for %.3f ns (%8.3f ns/day)' % (elapsed_time / unit.seconds, simulation_time / unit.nanoseconds, simulation_time / elapsed_time * unit.day / unit.nanoseconds))
with open(equilibrated_pdb_filename, 'w') as outfile:
    app.PDBFile.writeFile(pdb.topology, context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(), file=outfile, keepIds=True)
print('  final   : %8.3f kcal/mol' % (context.getState(getEnergy=True).getPotentialEnergy()/unit.kilocalories_per_mole))

# Serialize state
print('Serializing state to %s' % state_xml_filename)
state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
with open(state_xml_filename, 'w') as outfile:
    xml = openmm.XmlSerializer.serialize(state)
    outfile.write(xml)

# Serialize system
print('Serializing System to %s' % system_xml_filename)
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with open(system_xml_filename, 'w') as outfile:
    xml = openmm.XmlSerializer.serialize(system)
    outfile.write(xml)
