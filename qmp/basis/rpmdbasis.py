"""
initialization arrays for rpmd simulations
"""


from qmp.basis.basis import basis
from qmp.utilities import *
import numpy as np


class bead_basis(basis):
    """
    initialization array for rpmd in extented phase space
    """

    def __init__(self, coordinates, velocities, masses, n_beads=2, Temperature=None):
        """
        
        """

        basis.__init__(self)
        
        self.r = np.array(coordinates)
        self.npar = self.r.shape[0]
        if self.r.size == self.npar:
            self.ndim = 1
        else:
            self.ndim = self.r.shape[1]
        self.v = np.array(velocities)
        self.masses = np.array(masses)
        if (n_beads == 0) or (type(n_beads) != int):
            print '0 and lists are not allowed for number of beads, using n_beads = 2 per default'
            self.nb = 2
        else:
            self.nb = n_beads
            
        if (Temperature is None) or (len(Temperature)!=self.npar):
            print "Dude, you defined an inconsistent Temperature or none at all ~> using temperature of 293.15 K throughout"
            Temperature = [293.15]*self.npar
        
        self.Temp = np.array(Temperature)
        self.om = self.Temp*self.nb*kB/hbar
            
        self.r_beads = np.zeros((self.npar,self.nb,self.ndim))
        self.v_beads = np.zeros((self.npar,self.nb,self.ndim))
        
        if self.masses.size != self.masses.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif (self.masses.size != self.npar) or \
             (self.r.shape != self.v.shape):
            raise ValueError('Please provide consistent masses, coordinates, and velocities')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D dynamics implemented yet')
        
        for i_par in xrange(self.npar):
            for i_bead in xrange(self.nb-1):
                self.r_beads[i_par,i_bead] = self.r[i_par] #+ self.get_offset_bead(self.masses[i_par], self.Temp[i_par])
                self.v_beads[i_par,i_bead] = self.v[i_par] #self.get_v_bead(self.v[i_par])
            self.v_beads[i_par,self.nb-1] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,:], 0)
            self.r_beads[i_par,self.nb-1] = self.nb*self.r[i_par] - np.sum(self.r_beads[i_par,:], 0)



    def get_offset_bead(self, m, Temp):
        sigma = .1
        mu = np.sqrt(hbar/(2.*np.sqrt(m*kB*Temp)))
        l = np.random.normal(loc=mu, scale=sigma)
        v = np.random.normal(size=self.ndim)
        v /= np.sqrt(v.dot(v))
        return np.array(v*l)
    
    def get_v_bead(self, v):
        sigma = .1
        return np.random.normal(loc=v, scale=sigma, size=self.ndim)
    
    def __eval__(self):
        return self.r, self.v

    def get_kinetic_energy(self, m, v_beads):
        return m*np.sum(v*v)/2.

    def get_potential_energy(self, r_beads, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        return pot(*np.array(r_beads).T) + (1./2.)*m*om*om*np.sum(M.dot(r_beads)*M.dot(r_beads), 1)
            
    
    def get_forces(self, r1, r2, pot, m, om):
        #M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        #M[-1,0] = -1.
        if self.ndim == 1:
            return -1.*(num_deriv(pot, r1) + m*om*om*(r1-r2))
        elif self.ndim == 2:
            return -1.*(num_deriv_2D(pot, *r1) + m*om*om*(r1-r2))
    