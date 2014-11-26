"""
model class
"""

from utilities import *

class Model(object):
    """
    The model class contains all information and all 
    subroutines to run the calculations.

    Before it can run it has to be given a 
    dynamics object (wavefunction, particle, or necklace), 
    a potential and an integrator.

    """

    def __init__(self, **kwargs):
        """
        
        """
    
        default_parameters = {
                'ndim': 1,
                'mass': 1.0,
                'mode': 'wave', # wave, traj, rpmd
                'basis': 'onedgrid', # 1dgrid , 2dgrid, pws,
                'solver': 'numpy', # numpy, scipy, Lanczos
                'integrator': 'numerov', # numerov, ...
        }
        
        self.parameters = default_parameters
        for key, value in kwargs.iteritems():
            self.parameters[key]=value

        self.dyn = None 
        self.pot = None
        self.integrator = None

    def __repr__(self):

        string = 'Model Summary:\n'
        for key in self.parameters.keys():
            string += key +' : '+ str(self.parameters[key])+'\n'

        return string

    def set_potential(self,pot):

        pot.model = self
        self.pot = pot
        self.cell = pot.cell

    def set_basis(self,basis):

        basis.set_parameters(self.parameters)
        self.b = basis

    def propagate(self, steps):
       """
       propagates the system for the given time step
       """
      
       raise NotImplementedError('no propagation')


    def solve(self):
        """

        """
     
        


