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
from qmp.data_containers import data_container
from qmp.tools.termcolors import *

#TODO change model. integrators and solvers should be 
#initialized outside and model should only wrap run and solve routines


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
                'ndim': 1, # 1, 2
                'mass': 1.0,
                'mode': 'wave', # wave, traj, rpmd
                'basis': 'onedgrid', # onedgrid , twodgrid, pws,
                'solver': 'scipy', #  scipy, alglib, Lanczos
                'integrator': 'eigenprop', # primprop, eigenprop, splitopprop
                'states': 20,
        }
        
        self.parameters = default_parameters
        for key, value in kwargs.items():
            self.parameters[key]=value

        #exclusions ~> needed??
        #wavepacket dynamics needs a grid basis
        #if self.parameters['mode'] == 'wave':
            #self.parameters['solver'] = 'alglib'
            #if self.parameters['basis'] == 'onedgrid':
                #self.parameters['integrator'] = 'primprop'

        self.pot = None
        self.dyn = None
        self.solver = None
        self.basis = None
        
        self.data = data_container()
        self.data.ndim = self.parameters['ndim']
        self.data.mass = self.parameters['mass']
        self.data.solved = False
        
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
            print(gray+'Initializing Solver'+endcolor)
            self.solver = solver_init(self.data, self.pot)
            self.solver.solve()


    def run(self, steps, dt, **kwargs):
        """
        Wrapper for dyn.run
        """

        add_info = kwargs.get('additional', None)
        
        if (self.basis is None) or \
           (self.pot is None) or \
           (self.data is None):
            raise ValueError('Integrator can only run with \
                             initialized basis and potential')
        
        if ((self.parameters['integrator'] == 'eigenprop') or \
            (self.parameters['integrator'] == 'eigen') or \
            (add_info=='coefficients')) and \
           (self.data.solved == False):
            print(gray+'Projection onto eigen basis requires solving eigenvalue problem...'+endcolor)
            self.solve()
        
        try:
            self.dyn.run(steps=int(steps),dt=dt,**kwargs)
        except (AttributeError, TypeError):
            print(gray+'Initializing Integrator'+endcolor)
            self.dyn = integrator_init(self.data, self.pot)
            self.dyn.run(int(steps),dt,**kwargs)
 

#--EOF--#
