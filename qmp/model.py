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

from qmp.tools.utilities import *
from qmp.integrator import integrator_init
from qmp.solver import solver_init
from qmp.tools.termcolors import *
from qmp.data_containers import data_container

class Model(object):
    """
    The model class is an overaarching base class for 
    the model classes wave, particle, and necklace and 
    guides the workflow of dynamics calculations

    It guids the three workflow steps:
    - System Preparation
      We attach a potential to the model and make some general 
      definitions, this is done by __init__, set_potential, and 
      set_initial_conditions

    - Calculate things
      This either means solving the eigensystem with .solve or 
      running dynamics with .run

    - Analyze results
      This refers to visualization and data analysis. This is 
      taken care of by subroutines .analyze and .visualize

    Every model needs to be initialized with a potential and 
    with a mass.

    There are two basic tasks associated with the 
    solver subclass (time-independent problems) and the 
    dyn subclass (time  propagation)

    """

    def __init__(self, mass=1.0, potential=None):
        """
        Initializes the calculation model using mass and potential.
        """
    
        self.pot = potential 
        self.dyn = None
        self.solver = None
        self.model_type = 'Base Model'
        self.data = data_container()
        setattr(self.data, 'mass',mass) 
        self.set_potential(potential) 

    def __repr__(self):

        string = 'Model Summary:\n'
        string += '--------------\n'
        string += 'Potential : {0}\n'.format(str(self.pot))
        string += '\n'

        return string

    def set_potential(self,pot):
        """
        set or reset the potential if not set by __init__
        """
        self.pot = pot
        setattr(self.data,'domain',self.pot.domain)
        setattr(self.data,'ndim',self.pot.ndim)
        setattr(self.data,'nstates',self.pot.nstates)

    def set_initial_conditions(self):
        """
        Sets initial position and momentum. The rest pretty 
        much depends if quantum or classical.
        """

        raise NotImplementedError('This is not implemented at the base class level')

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
