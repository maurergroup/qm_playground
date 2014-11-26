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
                'solver': 'scipy', #  scipy, alglib, Lanczos
                'integrator': 'numerov', # numerov, ...
        }
        
        self.parameters = default_parameters
        for key, value in kwargs.iteritems():
            self.parameters[key]=value

        #exclusions
        if self.parameters['mode'] = 'wave':
            self.parameters['solver'] = 'alglib'
            if self.parameters['basis'] = 'onedgrid':
                self.parameters['integrator'] = 'primprop'


        self.pot = None
        self.integrator = None
        self.solver = None
        self.basis = None

        from data_containers import *
        self.data = data_containers[self.parameters['mode']]() 

        self.data.ndim = self.parameters['ndim']
        self.data.mass = self.parameters['mass']

    def __repr__(self):

        string = 'Model Summary:\n'
        for key in self.parameters.keys():
            string += key +' : '+ str(self.parameters[key])+'\n'

        return string

    def set_potential(self,pot):

        pot.data = self.data
        self.pot = pot
        self.data.cell = pot.cell

    def set_basis(self,basis):

        basis.data = self.data
        self.basis = basis

    def run(self, steps, dt):
        """
        propagates the system for the given time step and time 
        interval
        """
        from integrator import *

        if (self.basis is None) or (self.pot is None):
            raise ValueError('Integrator can only run with \
                initialized basis and potential')
        
        #initialize integrator
        self.integrator = integrator_type[self.parameters['integrator']]()

        # Solve Hamiltonian and write data
        self.data = self.integrator.solve(self.data, self.pot, \
                self.basis) 
      


    def solve(self):
        """
        Solve the time-independent problem
        This is only relevant for rpmd and wave. 
        In the case of rpmd this amounts to PIMD
        """
     
        from solver import *

        if (self.basis is None) or (self.pot is None):
            raise ValueError('Solver can only run with \
                    initialized basis and potential')

        self.solver = solver_init(self.parameters)

        # Solve Hamiltonian and write data
        self.data = self.solver.solve(self.data, self.pot, \
                self.basis) 


