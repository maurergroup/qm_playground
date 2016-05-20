#qmp.model
#
#    qm_playground - python package for dynamics simulations
#    Copyright (C) 2016  Reinhard J. Maurer 
#
#    This file is part of qm_playground.
#    
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#
"""
model class
"""

from qmp.model import Model
from qmp.tools.utilities import *
from qmp.integrator import integrator_init
from qmp.solver import solver_init
from qmp.tools.termcolors import *
from qmp.data_containers import data_container

class Wave(Model):
    """
    The Wave class derives from the model class and guides the workflow 
    for quantum dynamics with wavepackets.


    """

    def __init__(self, mass=1.0, potential=None,**kwargs):
        """
        Initializes the calculation model 
        
        mass:
          float
        potential:
          class Potential

        """
        Model.__init__(self,mass,potential)
        
        #TODO
        #wave specific stuff goes here

    def __repr__(self):
        string = Model.__repr__(self)
        
        string += str(self.data)
        string += '\n'

        return string

    def set_initial_conditions(self):
        """
        Sets initial position and momentum. The rest pretty 
        much depends if quantum or classical.
        """

        #TODO
        #wave specific stuff goes here
        raise NotImplementedError('This is not implemented at the base class level')

    #TODO this is still the Model duplicate
    def solve(self):
        """
        Wrapper for solver.solve
        """

        if (self.pot is None): 
            raise ValueError('Integrator can only run with \
                initialized potential')
        
        try:
            self.solver.solve()
        except (AttributeError, TypeError):
            print gray+'Initializing Solver'+endcolor
            self.solver = solver_init(self.data, self.pot)
            self.solver.solve()

    
    #TODO this is still the Model duplicate
    # adapt
    def run(self, steps, dt, **kwargs):
        """
        Wrapper for dyn.run
        """
        if (self.pot is None):
            raise ValueError('Integrator can only run with \
                             initialized potential')
       
        if (not hasattr(self.data,'p0') or hasattr(self.data,'x0')):
            print 'You should set initial positions and momenta'

        if ((self.parameters['integrator'] == 'eigenprop') or \
            (self.parameters['integrator'] == 'eigen') or \
            (add_info=='coefficients')) and \
           (self.data.solved == False):
            print gray+'Projection onto eigen requires solving eigenvalue problem...'+endcolor
            self.solve()
        
        try:
            self.dyn.run(int(steps),dt,**kwargs)
        except (AttributeError, TypeError):
            print gray+'Initializing Integrator'+endcolor
            self.dyn = integrator_init(self.data, self.pot)
            self.dyn.run(int(steps),dt,**kwargs)
 

#--EOF--#
