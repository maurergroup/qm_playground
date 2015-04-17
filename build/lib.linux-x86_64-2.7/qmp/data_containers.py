"""
data_containers.py
collection of different data containers for different 
jobs such as wavefunction, particle, RPMD necklace
"""

import numpy as np

#wave data object

class wave(object):
    """
    Abstract wavefunction object
    contains eigenfunctions, eigenvalues, 
    and all important subroutines that act 
    on the wavefunctions.
    """
    
    def __init__(self, basis):
   
        self.basis = basis
        N = basis.N
        self.psi = np.random.random([N,N])
        self.E = np.random.random(N)

    def normalize(self):
        """
        Normalizes the wavefunction vector 
        """

        norm = np.dot(self.psi,self.psi)
        self.psi = self.psi / np.sqrt(norm)


class traj(object):
    """
    initializes point(s) in phase space given in basis
    """
    
    def __init__(self, basis):
        
        self.basis = basis
        self.r = basis.r
        self.v = basis.v
        self.masses = basis.masses
        
        
class rpmd(object):
    """
    initializes beads in phase space according to basis
    """
    
    def __init__(self, basis):
        
        self.basis = basis
        self.r = basis.r
        self.v = basis.v
        self.masses = basis.masses
        self.n_beads = basis.nb



class data_container(dict):
    """
    Base class for all data containers
    """

    def __init__(self):

        #dimensions

        self.ndim = None
        self.mass = None
        self.cell = None

    def prep(self, mode, basis):

        #individual preparation
        if mode is 'wave':
            self.wvfn_preparation(basis)

        elif mode is 'traj':
            self.traj_preparation(basis)

        elif mode is 'rpmd':
            self.rpmd_preparation(basis)
        else:
            pass

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


    def wvfn_preparation(self,basis):
        """
        Initializes wvfn object for wvfn calculations
        """
    
        #stationary eigenvalues and eigenfunctions
        self.wvfn = wave(basis)
        #self.wvfn.normalize()
        
        #


    def traj_preparation(self, basis):
        """
        Initializes arrays for traj dynamics
        """
   
        ##error handling?
        self.traj = traj(basis)
    
    def rpmd_preparation(self, basis):
        """
        Initializes special arrays for wvfn calculations
        """

        self.rpmd = rpmd(basis)

