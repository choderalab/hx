from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

from pickle import dump
from openmmtools.integrators import LangevinIntegrator
from thresholds import utils, bisect, stability

ffxml_filenames = ['amber99sbildn.xml', 'tip3p.xml']

pressure = 1.0 * unit.atmospheres
temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
hydrogen_mass = 1 * unit.amu
splitting = 'V R O R V'

# Read equilibrated pdb
pdb_filename = 'equilibrated.pdb'
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

# Create simulation
print("Creating Simulation...")

##### THRESHOLDS PART

equilibrium_sim = app.Simulation(pdb.topology, system, LangevinIntegrator(temperature, collision_rate, timestep, splitting))
equilibrium_sim.context.setPositions(pdb.positions)
equilibrium_sim.context.setVelocitiesToTemperature(equilibrium_sim.integrator.getTemperature())
equilibrium_sim.step(1000)

# Test timesteps
print("Testing timesteps...")

def set_initial_conditions(sim):
    equilibrium_sim.step(100)
    utils.clone_state(equilibrium_sim, sim)
    
sim = app.Simulation(pdb.topology, system, LangevinIntegrator(temperature, collision_rate, timestep, splitting))
sim.context.setPositions(pdb.positions) 
sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature())

iterated_stability_oracle = stability.stability_oracle_factory(sim, set_initial_conditions, n_steps=100) # set steps of trial simulation

def noisy_oracle(dt):
    return iterated_stability_oracle(dt, n_iterations=3) # set how many times to repeat trial simulation (it's repeated until it fails)

x, zs, fs = bisect.probabilistic_bisection(noisy_oracle, search_interval=(2,6), resolution=100000, max_iterations=100, p=0.8, early_termination_width=0.001)

print('measured stability threshold: {:.3f}fs'.format(x[np.argmax(fs[-1])]))
