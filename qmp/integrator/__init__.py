"""
"""

from qmp.integrator.waveintegrators import *
from qmp.integrator.trajintegrators import *
from qmp.integrator.rpmdintegrators import *

integrator_type = {
    'primitive' : prim_propagator,
    'primprop' : prim_propagator,
    'eigen' : eigen_propagator,
    'eigenprop' :eigen_propagator,
    'SOFT' : SOFT_propagation,
    'splitopprop' : SOFT_propagation,
    'split_operator' : SOFT_propagation,
    'SOFT_scatter': SOFT_scattering,
    'SOFT_averages' : SOFT_average_properties,
    'SOFT_avg_properties' : SOFT_average_properties,
    'SOFT_avg_props' : SOFT_average_properties,
    'velocity_verlet' : velocity_verlet_integrator,
    'vel_verlet' : velocity_verlet_integrator,
    'VelocityVerlet_RPMD' : RPMD_VelocityVerlet,
    'RPMD_VelocityVerlet' : RPMD_VelocityVerlet,
    'RPMD_averages' : RPMD_equilibrium_properties,
    'RPMD_equilibrium_properties' : RPMD_equilibrium_properties,
    'RPMD_eq_props' : RPMD_equilibrium_properties,
    'RPMD_scatter' : RPMD_scattering,
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

