"""
integrator.py
"""

import numpy as np
#from qmp.logger import Logger

class Integrator(object):
    """
    Base class for all integrators. All integrators 
    need to be initialized with basis, potential and data 
    object.

    Every integrator needs the following:
     __init__,
     run, 
     get_forces,
     integrate
    
    """

    def __init__(self, data=None, potential=None, logger=None):
        
        self.data = data
        self.pot = potential
        
        #if logger is None:
            #self.data.logger = Logger()
        #else:
            #self.data.logger = logger

        self.counter = 0

    def run(self, steps, dt):
        """
        Placeholder for run function of subclasses.
        """

        raise NotImplementedError('run needs to be implemented by \
                the subclasses!')
        pass

    def get_forces(self):
        """
        Placeholder for get_forces function of subclasses.
        """

        raise NotImplementedError('get_forces needs to be implemented by \
                the subclasses!')
        pass

    def integrate(self):
        """
        Placeholder for integrate function of subclasses.
        """

        raise NotImplementedError('integrate needs to be implemented by \
                the subclasses!')
        pass


