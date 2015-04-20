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
            print "Dude, you gave me an inconsistent list of temperatures or none at all ~> using 293.15 K throughout"
            Temperature = [293.15]*self.npar
        
        self.Temp = np.array(Temperature)
        self.om = self.Temp*self.nb*kB/hbar
        
        if self.ndim == 2:
            xi = 2.*np.pi/self.nb
            rotMat = np.array([[np.cos(xi),np.sin(xi)],[-np.sin(xi),np.cos(xi)]])
            
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
            ## FIXME!!
            lvec = np.sqrt(hbar/(2.*np.sqrt(self.masses[i_par]*kB*self.Temp[i_par])))/800.
	    if self.ndim == 1:
	        self.r_beads[i_par,0] = self.r[i_par] - lvec
	        for i_bead in xrange(1,self.nb):
	            self.r_beads[i_par,i_bead] = self.r[i_par,i_bead-1]+(2.*lvec/self.nb)
	            self.v_beads[i_par,i_bead] = self.get_v_bead(self.v[i_par])
                self.v_beads[i_par,0] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,:], 0)
            else:
                rxi = np.random.rand(1)*2.*np.pi
                rM = np.array([[np.cos(rxi),np.sin(rxi)],[-np.sin(rxi), np.cos(rxi)]])
                rvec = np.dot(np.array([0.,lvec]),rM).T
                self.r_beads[i_par,0] = self.r[i_par] + rvec
                for i_bead in xrange(1, self.nb):
                    rvec = np.dot(rvec,rotMat)
                    self.r_beads[i_par,i_bead] = self.r[i_par] + rvec
                    self.v_beads[i_par,i_bead] = self.get_v_bead(self.v[i_par])
                self.v_beads[i_par,0] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,:], 0)



    def get_v_bead(self, v):
        sigma = .000001
        return np.random.normal(loc=v, scale=sigma, size=self.ndim)
    
    def __eval__(self):
        return self.r, self.v

    def get_kinetic_energy(self, m, v):
        return m*np.sum(v*v)/2.

    def get_potential_energy(self, r1, r2, pot, m, om):
        #M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        #M[-1,0] = -1.
        return pot(*r1) + (1./2.)*m*om*om*np.sum((r1-r2)*(r1-r2))
            
    
    def get_forces(self, r1, r2, pot, m, om):
        #M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        #M[-1,0] = -1.
        if self.ndim == 1:
            return -1.*(num_deriv(pot, r1) + m*om*om*(r1-r2))
        elif self.ndim == 2:
            return -1.*(num_deriv_2D(pot, *r1) + m*om*om*(r1-r2))
    
