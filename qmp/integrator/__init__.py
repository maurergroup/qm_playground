"""
"""

from qmp.integrator.waveintegrators import *
from qmp.integrator.trajintegrators import *

integrator_type = {
    'primitive' : prim_propagator,
    'primprop' : prim_propagator,
    'eigen' : eigen_propagator,
    'eigenprop' :eigen_propagator,
    'SOFT' : split_operator_propagator,
    'splitopprop' : split_operator_propagator,
    'split_operator' : split_operator_propagator,
    'velocity_verlet' : velocity_verlet_integrator,
    'vel_verlet' : velocity_verlet_integrator,
    }

def integrator_init(data, potential):

    param = data.parameters
    integrator = integrator_type[param['integrator']](\
            data=data, potential=potential)

    return integrator

def integrator_help():
        string = 'Integrator types include: \n'
        for k in integrator_type.keys():
            string += key+'\n'
        return string

