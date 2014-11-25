"""
model class
"""

from qmp.utilities import *

class model(object):
    """
    The model class contains all information and all 
    subroutines to run the calculations.

    Before it can run it has to be given a 
    dynamics object (wavefunction, particle, or necklace), 
    a potential and an integrator.

    """

    default_parameters = {
            'ndim': '1',
            'mode': 'wave', # wave, traj, rpmd
            'basis': '1dgrid', # 1dgrid , 2dgrid, pws, 
            'integrator': 'numerov', # numerov, ...
            '': '',



    def __init__(self, ndim=1, mode='qm'):
        """
        
        """

        self.parameters = default_parameters
        self.dyn = None 
        self.pot = None
        self.integrator = None


   def propagate(self, steps):
       """
       propagates the system for the given time step
       """
      
       raise NotImplementedError('no propagation')


   def solve(self):
       """

       """
      
       raise NotImplementedError('no solution')

