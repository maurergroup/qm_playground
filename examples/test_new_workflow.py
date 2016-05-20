"""
test_new_workflow.py

This routine puts the new workflow in 
branch newdeal to a test.
"""

import numpy as np
import sys
from qmp import Model
from qmp import Wave
from qmp.potential import Potential

domain = [-10.,10.]
mass = 1.0

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

w = Wave(mass, pot)

print w
#initialization sets the DVR grid, 
#we can tell what solver and what integrator to use

