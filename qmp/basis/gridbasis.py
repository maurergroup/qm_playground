"""
gridbasis.py

all grid-based basis functions
"""

from qmp.utilities import *
from qmp.basis.basis import basis
import numpy as np

class onedgrid(basis):
    """
    wavefunction representation as equidistant grid
    """

    def __init__(self, start=0.0, end=1.0, N=100):
        """
        Equidistant 1D grid of points as wvfn representation
        """

        basis.__init__(self)

        self.x = np.linspace(start, end, N)
        self.dx = self.x[1] - self.x[0] 
        self.N = N
        
        #wavefunction vector initialised with zeros
        self.psi = np.zeros_like(self.x)

        #define derivative matrix with symm. finite difference
        D = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1) )/(2.*self.dx)
        #boundary conditions f(0)=0 f(N)=0
        D[0,0] = 0
        D[0,1] = 0
        D[1,0] = 0
        D[-1,-2] = 0
        D[-1,-1] = 0
        D[-2,-1] = 0
        self.D = D

        #define Laplace operator
        L = (-2*np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1) \
                + np.diag(np.ones(N-1),-1)) / (self.dx*self.dx)
        L[0,0] = 0
        L[0,1] = 0
        L[1,0] = 0
        L[-1,-2] = 0
        L[-1,-1] = 0
        L[-2,-1] = 0
        self.L = L
        

    def __eval__(self):

        return self.psi

    def deriv_psi(self):

        return np.dot(self.D,self.psi)

    def Lap_psi(self):

        return np.dot(self.L,self.psi)

    def construct_Tmatrix(self):

        m = self.data.mass
        return -(1./2.)*((hbar**2)/m)*self.L

    def construct_Vmatrix(self,pot):

        return np.diag(pot(self.x))
        



