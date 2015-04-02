"""
model class
"""

from qmp.utilities import *
from qmp.integrator import integrator_init
from qmp.solver import solver_init
from qmp.data_containers import data_container 

class Model(object):
    """
    The model class contains all information and all 
    subroutines to run the calculations.

    Before it can run it has to be given a 
    potential, and a basis.

    There are two basic tasks associated with the 
    solver subclass (time-independent problems) and the 
    dyn subclass (time  propagation)

    """

    def __init__(self, **kwargs):
        """
        Initializes the calculation model using arbitrary keyword arguments 
        which are parsed later.
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
        #wavepacket dynamics needs a grid basis
        if self.parameters['mode'] == 'wave':
            #self.parameters['solver'] = 'alglib'
            if self.parameters['basis'] == 'onedgrid':
                self.parameters['integrator'] = 'primprop'

        self.pot = None
        self.dyn = None
        self.solver = None
        self.basis = None

        self.data = data_container()
        self.data.ndim = self.parameters['ndim']
        self.data.mass = self.parameters['mass']
        
        #data also keeps parameters
        self.data.parameters = self.parameters

    def __repr__(self):

        string = 'Model Summary:\n'
        for key in self.parameters.keys():
            string += key +' : '+ str(self.parameters[key])+'\n'

        string += '\n'
        for key in self.data.keys():
            string += 'contains data entries for: '+key+'\n'

        return string

    def set_potential(self,pot):

        pot.data = self.data
        self.pot = pot
        self.data.cell = pot.cell

    def set_basis(self,basis):

        basis.data = self.data
        self.basis = basis

        #now we can prepare the data object for the tasks ahead
        self.data.prep(self.parameters['mode'], basis) 

    def solve(self):
        """
        Wrapper for solver.solve
        """

        if (self.basis is None) or \
           (self.pot is None) or \
           (self.data is None):
            raise ValueError('Integrator can only run with \
                initialized basis and potential')
        
        try:
            self.solver.solve()
        except (AttributeError, TypeError):
            print 'Initializing Solver'
            self.solver = solver_init(self.data, self.pot)
            self.solver.solve()

    def run(self, steps, dt):
        """
        Wrapper for dyn.run
        """

        if (self.basis is None) or \
           (self.pot is None) or \
           (self.data is None):
            raise ValueError('Integrator can only run with \
                initialized basis and potential')
        
        try:
            self.dyn.run(steps,dt)
        except (AttributeError, TypeError):
            print 'Initializing Integrator'
            self.dyn = integrator_init(self.data, self.pot)
            #self.dyn.run(steps,dt)
 
        #self.dyn.run(steps, dt)
