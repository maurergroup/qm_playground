"""
integrator.py
"""

import numpy as np
from wvfn_integrators import *
from traj_integrators import *


class integrator(object):
    """
    Base class for all integrators
    """

    def __init__(self):
        pass



class integrator_init():

    integrator_type = {
            'primprop' : prim_propagator,
            #''
            }

    def __init__(param):

        integrator = integrator_type[param['integrator']]()

        return integrator




