"""
initialization arrays for rpmd simulations
"""


from qmp.basis.basis import basis
from qmp.utilities import num_deriv_2D, num_deriv, hbar, kB
import numpy as np
from scipy.stats import maxwell
import scipy.linalg as la


class bead_basis(basis):
    """
    initialization array for rpmd in extented phase space
    """

    def __init__(self, coordinates, velocities, masses, n_beads=2, Temperature=None, trial_init=False):
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
        
        if (trial_init == True) and (self.ndim == 2):
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
        
        if trial_init == True:
            for i_par in xrange(self.npar):
                lvec = (hbar/np.pi)*np.sqrt(self.nb/(2.*self.masses[i_par]*kB*self.Temp[i_par]))/800.
                #lvec = np.sqrt(self.nb*hbar/(2.*np.sqrt(self.masses[i_par]*kB*self.Temp[i_par])))
                if self.ndim == 1:
                    self.r_beads[i_par,0] = self.r[i_par] - lvec
                    for i_bead in xrange(1,self.nb):
                        self.r_beads[i_par,i_bead] = self.r_beads[i_par,i_bead-1]+(2.*lvec/self.nb)
                        self.v_beads[i_par,i_bead] = self.get_v_bead(self.v[i_par], self.masses[i_par], self.Temp[i_par])
                    self.v_beads[i_par,0] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,:], 0)
                else:
                    rxi = np.random.rand(1)*2.*np.pi
                    rM = np.array([[np.cos(rxi),np.sin(rxi)],[-np.sin(rxi), np.cos(rxi)]])
                    rvec = np.dot(np.array([0.,lvec]),rM).T
                    self.r_beads[i_par,0] = self.r[i_par] + rvec
                    for i_bead in xrange(1, self.nb):
                        rvec = np.dot(rvec,rotMat)
                        self.r_beads[i_par,i_bead] = self.r[i_par] + rvec
                        self.v_beads[i_par,i_bead] = self.get_v_bead(self.v[i_par], self.masses[i_par], self.Temp[i_par])
                    self.v_beads[i_par,0] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,:], 0)
        else:
            for i_par in xrange(self.npar):
                for i_bead in xrange(self.nb):
                    self.r_beads[i_par,i_bead] = self.r[i_par]
                    self.v_beads[i_par,i_bead] = self.get_v_bead(self.v[i_par], self.masses[i_par], self.Temp[i_par])
                self.v_beads[i_par,0] = self.nb*self.v[i_par] - np.sum(self.v_beads[i_par,1:], 0)
        

    def get_v_bead(self, v, m, T):
        s = np.sqrt(kB*T/m)
        l = -np.sqrt(8./np.pi)*s
        x = np.random.random(1)
        v_off = maxwell.ppf(x,loc=l,scale=s)
        if self.ndim ==1:
            if v == 0.:
                return v_off*np.random.choice([-1.,1.])
            else:
                return v + v_off*np.sign(v)
        else:
            if not (np.any(v) != 0.):
                ang = 2.*np.pi*np.random.random()
                v_off = v_off*np.array([0.,1.])
            else:
                ang = 2.*np.pi*np.random.normal(0., scale=0.02)
                v_off = v + v_off*v/la.norm(v)
            M = np.array( [ [np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)] ] )
            return v_off.dot(M)
        
    
    def __eval__(self):
        return self.r, self.v

    def get_kinetic_energy(self, m, v):
        return m*np.sum(v*v)/2.

    def get_potential_energy(self, r, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        return pot(*(r.T)).T + (1./2.)*m*om*om*np.sum(M.dot(r)*M.dot(r),1)
            
    
    def get_forces(self, r, pot, m, om):
        M = np.eye(self.nb) - np.diag(np.ones(self.nb-1),1)
        M[-1,0] = -1.
        if self.ndim == 1:
            return -1.*(np.array([num_deriv(pot, r)]).T + m*om*om*M.dot(r))
        elif self.ndim == 2:
            return -1.*(num_deriv_2D(pot, *(r.T)).T + m*om*om*M.dot(r))
    
