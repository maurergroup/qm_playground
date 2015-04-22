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

    def __init__(self, coords, vels, masses, n_beads=2, T=None,pos_init=False,vel_init=True):
        """
        Initializes bead positions and velocities and stores parameters for propagation
        
        Parameters/Keywords:
        ====================
            coords:     list of list of coordinates of particle 1,2,... in au ([[x1,y1],[x2,y2], ...])
            vels:       list of list of velocities of particle 1,2,... in au ([[v1x,v1y],[v2x,v2y], ...])
            masses:     list of particle masses in au ([m1,m2, ...])
            n_beads:    number of beads to represent particles (integer, default 2)
            T:          list of particle temperatures in K ([T1,T2, ...])
            pos_init:   set bead positions on around particle pos (boolean, default False)
            vel_init:   draw vel from Maxwell-Boltzmann distribution, \
                        set absolut bead velocities to vel, \
                        set bead velocities equally spread away from particle pos
                        (boolean, default True, set to True if pos_init == False)
                        
        **COMMENTS**    - pos_init approximative!
                        - Only pos_init or vel_init will be set to True, otherwise double counting
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
        self.v = np.array(vels)
        self.masses = np.array(masses)
        if (n_beads == 0) or (type(n_beads) != int):
            print '0 and lists are not allowed for number of beads, using n_beads = 2 per default'
            self.nb = 2
        else:
            self.nb = n_beads
            
        if (T is None) or (len(T)!=self.npar):
            print "Dude, you gave me an inconsistent list of temperatures or none at all ~> using 293.15 K throughout"
            T = [293.15]*self.npar
        
        self.Temp = np.array(T)
        self.om = self.Temp*self.nb*kB/hbar
        
        if self.ndim == 2:
            xi = 2.*np.pi/self.nb
            rotMat = np.array([[np.cos(xi),np.sin(xi)],[-np.sin(xi),np.cos(xi)]])
            arand = np.random.random(1)*2.*np.pi
            rM = np.array([[np.cos(arand),np.sin(arand)],[-np.sin(arand), np.cos(arand)]])
            
        self.r_beads = np.zeros((self.npar,self.nb,self.ndim))
        self.v_beads = np.zeros((self.npar,self.nb,self.ndim))
        
        if self.masses.size != self.masses.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif (self.masses.size != self.npar) or \
             (self.r.shape != self.v.shape):
            raise ValueError('Please provide consistent masses, coordinates, and velocities')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D dynamics implemented yet')
        
        if pos_init == True:
            for i_par in xrange(self.npar):
                r_abs = (hbar/np.pi)*np.sqrt(self.nb/(2.*self.masses[i_par]*kB*self.Temp[i_par]))/400.
                print r_abs
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
                v_abs = self.get_v_maxwell(self.masses[i_par],self.Temp[i_par])
                if self.ndim == 1:
                    i_iter = np.random.choice([-1.,1.])
                    for i_bead in xrange(self.nb):
                        self.v_beads[i_par,i_bead] = self.v[i_par] + v_abs*i_iter
                        i_iter *= -1
                    if self.nb%2 == 1:
                        self.v_beads[i_par,0] = 0.
                else:
                    v_vec = np.dot(np.array([0.,v_abs]),rM).T
                    for i_bead in xrange(self.nb):
                        self.v_beads[i_par,i_bead] = self.v[i_par] + v_vec
                        v_vec = np.dot(v_vec,rotMat)
        else:
            for i_par in xrange(self.npar):
                for i_bead in xrange(self.nb):
                    self.v_beads[i_par,i_bead] = self.v[i_par]
        

    def get_v_maxwell(self, m, T):
        s = np.sqrt(kB*T/m)
        l = -np.sqrt(8./np.pi)*s
        x_rand = np.random.random(1)
        return maxwell.ppf(x_rand,loc=l,scale=s)
    
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
    
