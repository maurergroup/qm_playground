"""
waveintegrators.py
"""

from integrator import Integrator
import numpy as np

class prim_propagator(Integrator):
    """
    Primitive exp(-iEt) propagator
    """

    def __init__(self, **kwargs):
        """
        Primprop init
        """
        Integrator.__init__(self, **kwargs)

        #prepare data object for time integration
       
    def run(self, steps, dt):
        """
        Propagates the system for 'steps' steps and 
        a timestep of dt.
        """


