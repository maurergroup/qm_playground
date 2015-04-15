"""
initialization arrays for rpmd simulations
"""


from qmp.basis.basis import basis
from qmp.utilities import *
import numpy as np


class rpmd_basis(basis):
    """
    initialization array for rpmd in extented phase space
    """

    def __init__(self, coordinates, velocities, masses, n_beads=2, T=293.15):
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
        self.nb = n_beads
        self.r_beads = np.zeros((self.npar,self.n_beads,self.ndim))
        
        if self.masses.size != self.masses.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif (self.masses.size != self.npar) or \
             (self.r.shape != self.v.shape):
            raise ValueError('Please provide consistent masses, coordinates, and velocities')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D implemented yet')
        
        for j in xrange(self.npar):
            for i in xrange(self.nb-1):
                self.r_beads[j,i] = [self.r[j] + get_offset_bead(self.m[j], T)]
            self.r_beads[j,self.nb] = [self.nb*self.r[j] - np.sum(self.r_beads[j,:], 0)]
        
        print r_beads
        
        for j in xrange(self.npar):
            for i in xrange(self.nb-1):
                self.v_beads[j,i] = get_v_bead(self.v[j])
            self.v_beads[j,self.nb] = self.nb*self.v[j] - np.sum(self.v_beads[j,:], 0)

        print v_beads

    def get_offset_bead(self, m, T):
        sigma = .2
        mu = np.sqrt(hbar/(2.*np.sqrt(m*kB*T)))
        l = np.random.normal(loc=mu, scale=sigma)
        v = np.random.normal(size=self.ndim)
        v /= np.sqrt(v.dot(v))
        return v*l
    
    def get_v_bead(self, v):
        sigma = .2
        return np.random.normal(loc=v, scale=sigma, size=self.ndim)
    
    def __eval__(self):
        return self.r, self.v

    def get_kinetic_energy(self, m, v_beads):
        return m*np.sum(v*v,1)/2.

    def get_potential_energy(self, r_beads, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        return pot(*np.array(r_beads).T) + (1./2.)*m*om*om*np.sum(M.dot(r_beads)*M.dot(r_beads), 1)
            
    
    def get_forces(self, r_beads, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        if self.ndim == 1:
            return -1.*(num_deriv(pot, *np.array(r_beads).T) + m*om*om*M.dot(r_beads))
        elif self.ndim == 2:
            return -1.*(num_deriv_2D(pot, *np.array(r_beads).T) + m*om*om*M.dot(r_beads))
    