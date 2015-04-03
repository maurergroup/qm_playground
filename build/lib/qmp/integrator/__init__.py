"""
"""

from qmp.integrator.waveintegrators import *
from qmp.integrator.trajintegrators import *

integrator_type = {
    'primprop' : prim_propagator,
    #''
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

