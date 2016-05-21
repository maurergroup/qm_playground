"""
test_new_workflow.py

This routine puts the new workflow in 
branch newdeal to a test.
"""
import numpy as np
import sys
from qmp import Model, Wave
from qmp.potential import Potential
from qmp.integrator.dyn_tools import create_gaussian 

domain = [-10.,10.]
mass = 1850.0

###DEFINE POTENTIAL
def create_harm(x0,k):
    def f_harm(x):
        return 0.5*k*k*(x-x0)**2

    return f_harm

pot = Potential(domain, f= create_harm(0.,0.1))
print(pot)
######define base model
m = Model(mass, pot)
print m
####define wave model
w = Wave(mass, pot, 
        dx=0.1,
        N=400,
        solver='scipy',
        integrator='SOFT',
        )

print w
#initialization sets the DVR grid, 
#we can tell what solver and what integrator to use
sigma = 1./2.
psi_0 = create_gaussian(w.data.basis.x, x0=0., p0=1.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))

#both works, manual wvfn
w.set_initial_conditions(psi_0)
#...and automatic gaussian with position, momentum and width
w.set_initial_conditions(x0=0.,p0=1.0, sigma=1./2.)

w.run(steps=200, dt=82.)

### VISUALIZATION ###

w.visualize_dynamics()
