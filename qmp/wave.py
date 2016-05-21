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
from qmp.basis.gridbasis import onedgrid, twodgrid
from qmp.integrator.dyn_tools import create_gaussian

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
        
        #defaults
        default_keywords = {
                'integrator': 'SOFT',
                'solver': 'scipy',
                'dx' : 0.1, 
                'N' : 100,
                }
        self.data.parameters = default_keywords.copy()
        for keyword in kwargs:
            self.data.parameters[keyword] = kwargs[keyword]

        #Basis
        if self.pot is not None:
            self.set_basis()

        #wvfn
        self.data['psi'] = None

    def __repr__(self):
        string = Model.__repr__(self)
        
        string += str(self.data)
        string += '\n'

        return string

    #def set_potential(self,pot):
        #Model.set_potential(self,pot)
        #self.set_basis()

    def set_basis(self,start=None, end=None):
        """
        defines DVR grid over whole domain
        """
        
        domain = self.pot.domain
        if start is None:
            start = domain[0]
        if end is None:
            end = domain[1]
        domain = np.array([start,end])
        dist = domain[1]-domain[0]

        if self.data['ndim'] ==1:
            N = int(dist/self.data.parameters['dx'])
            if N<self.data.parameters['N']:
                N = self.data.parameters['N']
            print 'Initializing DVR grid with {0} points'.format(N)
            self.data.basis = onedgrid(domain[0],domain[1],N)


        if self.data['ndim'] == 2:
            N1 = int(dist[0]/self.data.parameters['dx'])
            N2 = int(dist[1]/self.data.parameters['dx'])
            if N1<self.data.parameters['N']:
                N1 = self.data.parameters['N']
            if N2<self.data.parameters['N']:
                N2 = self.data.parameters['N']
            N2 = N1
            print 'Initializing DVR grid with {0} and {1} points'.format(N1,N2)
            self.data.basis = twodgrid(domain[0],domain[1],N1,N2)

    def set_initial_conditions(self,psi=None,x0=None,p0=None,sigma=None):
        """
        Sets initial position and momentum. The rest pretty 
        much depends if quantum or classical.
        """

        if x0 is None:
            x0 = 0.
        if p0 is None:
            p0 = 1.0
        if sigma is None:
            sigma = 1./2.
        if psi is None:
            psi = create_gaussian(self.data.basis.x,x0=x0,p0=p0,sigma=sigma)
        self.data['psi'] = psi

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

        self.data.parameters['dt'] = dt

        if (self.pot is None):
            raise ValueError('Integrator can only run with \
                             initialized potential')
       
        #if (not hasattr(self.data,'p0') or hasattr(self.data,'x0')):
            #print 'You should set initial positions and momenta'

        if ((self.data.parameters['integrator'] == 'eigenprop') or \
            (self.data.parameters['integrator'] == 'eigen')) and \
           (self.data.solved == False):
            print gray+'Projection onto eigen requires solving eigenvalue problem...'+endcolor
            self.solve()
        
        try:
            self.dyn.run(int(steps),dt,**kwargs)
        except (AttributeError, TypeError):
            print gray+'Initializing Integrator'+endcolor
            self.dyn = integrator_init(self.data, self.pot)
            self.dyn.run(int(steps),dt,**kwargs)

    def visualize_dynamics(self):
        """
        TODO currently is just calling wave_movie1D, doesn't work for 2D
        """
        from qmp.tools.visualizations import *

        wave_movie1D(self.data.basis.x, self.data.psi_t*np.conjugate(self.data.psi_t),\
                self.pot(self.data.basis.x), dt=self.data.parameters['dt'], \
                E_arr=self.data.E_t, \
                rho_tot_arr=np.sum(self.data.psi_t*np.conjugate(self.data.psi_t),1), \
                E_kin_arr=self.data.E_kin_t, \
                E_pot_arr=self.data.E_pot_t,\
                )

#--EOF--#
