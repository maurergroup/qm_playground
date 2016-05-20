#qmp.integrator.trajintegrators
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
trajintegrators.py
"""

from qmp.integrator.integrator import Integrator
from qmp.tools.termcolors import *
import numpy as np


class velocity_verlet_integrator(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """
    
    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)
        
        
    def run(self, steps, dt, **kwargs):
        
        N = self.data.traj.basis.npar
        ndim = self.data.traj.basis.ndim
        m = self.data.traj.basis.masses
        r_t = np.zeros((N,steps+1,ndim))
        r_t[:,0,:] = np.array(self.data.traj.basis.r)
        v_t = np.zeros((N,steps+1,ndim))
        v_t[:,0,:] = np.array(self.data.traj.basis.v)
        E_pot = np.zeros((N,steps+1))
        E_kin = np.zeros((N,steps+1))
        E_tot = np.zeros((N,steps+1))
        
        print gray+'Integrating...'+endcolor
        for i_par in xrange(N):
            for i_step in xrange(steps):
                e_pot = self.data.traj.basis.get_potential_energy(r_t[i_par,i_step], self.pot)
                e_kin = self.data.traj.basis.get_kinetic_energy(m[i_par], v_t[i_par,i_step])
                E_pot[i_par,i_step] = e_pot
                E_kin[i_par,i_step] = e_kin
                E_tot[i_par,i_step] = (e_pot+e_kin)
                
                F = self.data.traj.basis.get_forces(r_t[i_par, i_step], self.pot)
                v1 = v_t[i_par,i_step] + F/m[i_par]*dt/2.
                r_t[i_par,i_step+1] = r_t[i_par,i_step]+v1*dt
                F = ( F + self.data.traj.basis.get_forces(r_t[i_par,i_step+1], self.pot) )/2.
                v_t[i_par,i_step+1] = v_t[i_par,i_step]+F/m[i_par]*dt
                
            e_pot = self.data.traj.basis.get_potential_energy(r_t[i_par,-1], self.pot)
            e_kin = self.data.traj.basis.get_kinetic_energy(m[i_par], v_t[i_par,-1])
            E_pot[i_par,steps] = e_pot
            E_kin[i_par,steps] = e_kin
            E_tot[i_par,steps] = (e_pot+e_kin)
                
        print gray+'INTEGRATED'+endcolor
        
        self.data.traj.r_t = r_t
        self.data.traj.v_t = v_t
        self.data.traj.E_kin_t = np.array(E_kin)
        self.data.traj.E_pot_t = np.array(E_pot)
        self.data.traj.E_t = np.array(E_tot)
        
