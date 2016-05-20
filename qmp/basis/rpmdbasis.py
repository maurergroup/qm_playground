#qmp.basis.rpmdbasis
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
initialization arrays for rpmd simulations
"""

from qmp.basis.basis import basis
from qmp.tools.utilities import *
from qmp.tools.termcolors import *
import numpy as np
from scipy.stats import maxwell
import scipy.linalg as la


class bead_basis(basis):
    """
    initialization array for rpmd in extented phase space
    """

    def __init__(self, coords, vels, masses, n_beads=4,T=None,pos_init=False,vel_init=True):
        """
        Initializes bead positions and velocities and stores parameters for propagation
        
        Parameters/Keywords:
        ====================
            coords:     list of list of coordinates of particle 1,2,... in au ([[x1,y1],[x2,y2], ...])
            vels:       list of list of velocities of particle 1,2,... in au ([[v1x,v1y],[v2x,v2y], ...])
            masses:     list of particle masses in au ([m1,m2, ...])
            n_beads:    number of beads to represent particles (integer, default 4)
            T:          list of particle temperatures in K ([T1,T2, ...])
            pos_init:   set bead positions on around particle pos (boolean, default False)
            vel_init:   2D - same absolute velocity from Maxwell-Boltzmann distribution equally spread in plane
                        1D - necklace divided into pairs, same absolute velocity with different sign for beads of pair
                        (boolean, default True, set to True if pos_init == False)
        
        **COMMENTS**    - pos_init approximative!
                        - Only pos_init or vel_init will be set to True, to prevent double counting
                          (default: vel_init = True)
        """

        basis.__init__(self)
        if vel_init == True:
            pos_init = False
        else:
            pos_init = True
        
        self.r = np.array(coords)
        self.npar = self.r.shape[0]
        if self.r.size == self.npar:
            self.ndim = 1
        else:
            self.ndim = self.r.shape[1]
        
        self.m = np.array(masses)
        if self.m.size != self.m.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D dynamics implemented yet')
        
        self.v = np.array(vels)
        assert (self.v.shape == self.r.shape)
        
        if (n_beads == 0) or (type(n_beads) != int):
            print grey+'0 and lists are not allowed for number of beads, using n_beads = 4 per default'+endcolor
            self.nb = 4
        else:
            self.nb = n_beads
        
        if (T is None) or (len(T)!=self.npar):
            print grey+"Dude, you gave me an inconsistent list of temperatures or none at all ~> using 293.15 K throughout"+endcolor
            T = [293.15]*self.npar
        
        self.Temp = np.array(T)
        self.om = self.Temp*self.nb*kB/hbar
        
        print 'RPMD simulation using'
        print 'Np = '+str(self.npar)+'   non-interacting particles in'
        print 'Ndim = '+str(self.ndim)+'  dimensions using'
        #print 'T [K] =   '+str(self.Temp)
        print 'Nb = '+str(self.nb)+'  beads per particle\n'
        
        if self.ndim == 2:
            xi = 2.*np.pi/self.nb
            rotMat = np.array([[np.cos(xi),np.sin(xi)],[-np.sin(xi),np.cos(xi)]])
            arand = np.random.random(1)*2.*np.pi
            rM = np.array([[np.cos(arand),np.sin(arand)],[-np.sin(arand), np.cos(arand)]])
            
        self.r_beads = np.zeros((self.npar,self.nb,self.ndim))
        self.v_beads = np.zeros((self.npar,self.nb,self.ndim))
        
        if pos_init == True:
            for i_par in xrange(self.npar):
                r_abs = (hbar/np.pi)*np.sqrt(self.nb/(2.*self.m[i_par]*kB*self.Temp[i_par]))/400.
                if self.ndim == 1:
                    self.r_beads[i_par,0] = self.r[i_par] - r_abs
                    for i_bead in xrange(1,self.nb):
                        self.r_beads[i_par,i_bead] = self.r_beads[i_par,i_bead-1]+(2.*r_abs/(self.nb-1))
                else:
                    r_vec = np.dot(np.array([0.,r_abs]),rM).T
                    self.r_beads[i_par,0] = self.r[i_par] + r_vec
                    for i_bead in xrange(1, self.nb):
                        r_vec = np.dot(r_vec,rotMat)
                        self.r_beads[i_par,i_bead] = self.r[i_par] + r_vec
        else:
            for i_par in xrange(self.npar):
                for i_bead in xrange(self.nb):
                    self.r_beads[i_par,i_bead] = self.r[i_par]
                    
        if vel_init == True:
            for i_par in xrange(self.npar):
                if self.ndim == 1:
                    i_start = self.nb%2    # for odd number of beads: (nb-1)/2 pairs, v=0 for last bead ~> start at i_start=1
                    for i_bead in xrange(i_start,self.nb,2): 
                        v_mb = self.get_v_maxwell(self.m[i_par], self.Temp[i_par])
                        # assign v_p + v_mb to bead1 of pair
                        self.v_beads[i_par,i_bead] = self.v[i_par] + v_mb
                        # assign v_p - v_mb to bead2 of pair
                        self.v_beads[i_par,i_bead+1] = self.v[i_par] - v_mb
                else:
                    v_abs = self.get_v_maxwell(self.m[i_par],self.Temp[i_par])
                    v_vec = np.dot(np.array([0.,v_abs]),rM).T
                    for i_bead in xrange(self.nb):
                         self.v_beads[i_par,i_bead] = self.v[i_par] + v_vec
                         v_vec = np.dot(v_vec,rotMat)
        else:
            for i_par in xrange(self.npar):
                for i_bead in xrange(self.nb):
                    self.v_beads[i_par,i_bead] = self.v[i_par]


    def get_v_maxwell(self, m, T):
        """
        draw velocity from Maxwell-Boltzmann distribution with mean 0.
        """
        s = np.sqrt(kB*T/m)
        x_rand = np.random.random(1)
        return maxwell.ppf(x_rand,loc=0.,scale=s)
    

    def __eval__(self):
        return self.r, self.v


    def get_kinetic_energy(self, m, v):
        return m*np.sum(v*v,1)/2.


    def get_potential_energy_beads(self, r, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        return pot(*(r.T)).T + (1./2.)*m*om*om*np.sum(M.dot(r)*M.dot(r),1)
    

    def get_potential_energy_atom(self, r, pot):
        return pot(*(r.T)).T
    
    
    def get_forces_beads(self, r, pot, m, om):
        M = 2.*np.eye(self.nb) - np.diag(np.ones(self.nb-1),1) - np.diag(np.ones(self.nb-1),-1)
        M[-1,0], M[0,-1] = -1., -1.
        if self.ndim == 1:
            return -1.*(np.array([num_deriv(pot, r)]).T + m*om*om*M.dot(r))
        elif self.ndim == 2:
            return -1.*(num_deriv_2D(pot, *(r.T)).T + m*om*om*M.dot(r))
        
        
    def get_forces_atom(self, r, pot):
        if self.ndim == 1:
            return -1.*num_deriv(pot, r)
        elif self.ndim == 2:
            return -1.*num_deriv_2D(pot, *(r.T)).T
        
    
    def get_hessian_atom(self, r, pot):
        if self.ndim == 1:
            return num_deriv2(pot, r)
        elif self.ndim == 2:
            return num_deriv2_2D(pot, *(r.T)).T
        
    
    def get_omega_Rugh(self, r, pot, v, m):
        """
        return omega(t) = (nb/hbar)*kB*T = (nb/hbar)*( (|V'(r)|^2 + |v|^2) / (V"(r) + ndim/m) )
        (definition of dynamical temperature according to Rugh)
        
        parameters:
        ===========
            r:  the PARTICLE's position
            v:  the PARTICLE's velocity
            m:  the PARTICLE's mass
        """
        A = la.norm(self.get_forces_atom(r, pot))
        a = self.get_hessian_atom(r, pot)
        v = la.norm(v)
        
        return (self.nb/hbar)*( (A**2 + v**2) / (a + (self.ndim/m)) )
