"""
gridbasis.py

all grid-based basis functions
"""

from qmp.utilities import *
from qmp.basis.basis import basis
import numpy as np
import scipy.sparse as sparse

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
        

class  twodgrid(basis):
    """
    wavefunction on equidistant 2D grid
    """

    def __init__(self, start=[0.,0.], end=[1.,1.], N=100.):
        """
        Equidistant 2D grid
        """

        basis.__init__(self)

        self.x = np.linspace(start[0], end[0], N)
        self.dx = self.x[1]-self.x[0]
        self.N = N

        self.y = np.linspace(start[1], end[1], N)
        self.dy = self.y[1] - self.y[0]

        self.xgrid, self.ygrid = np.meshgrid(self.x, self.y)

        self.psi = np.zeros_like(self.xgrid)

        #1st Derivatives (central difference approx)
        Dx = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))/(self.dx*2.)
        Dy = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))/(self.dy*2.)

        Dx[0,:] = [0.]*Dx.shape[1]
        Dx[:,0] = [0.]*Dx.shape[0]
        Dx[-1,:] = [0.]*Dx.shape[1]
        Dx[:,-1] = [0.]*Dx.shape[0]
        self.Dx = Dx

        Dy[0,:] = [0.]*Dy.shape[1]
        Dy[:,0] = [0.]*Dy.shape[0]
        Dy[-1,:] = [0.]*Dy.shape[1]
        Dy[:,-1] = [0.]*Dy.shape[0]
        self.Dy = Dy

        #Laplacian
        a, b = -2.*np.ones(N), np.ones(N)
        D = sparse.spdiags([b,a,b], [-1,0,1], N,N)
        Lx = D/(self.dx*self.dx)
        Ly = D/(self.dy*self.dy)

        L = sparse.kron(Lx, np.eye(N)) + sparse.kron(np.eye(N), Ly)

        #L[0,:] = [0.]*L.shape[1]
        #L[:,0] = [0.]*L.shape[0]
        #L[-1,:] = [0.]*L.shape[1]
        #L[:,-1] = [0.]*L.shape[0]
        self.L = L


    def __eval__(self):
        return self.psi

    def deriv_psi(self):
        """
        Returns array of derivatives (result[0]: d/dx, result[1]: d/dy)
        """
        return np.array([np.dot(self.psi, self.Dx), np.dot(self.Dy, self.psi)])

    def Lap_psi(self):
        return self.L.dot(self.psi.flatten())#+reshape?

    def construct_Tmatrix(self):
        m = self.data.mass
        return -(hbar*hbar/2./m)*self.L

    def construct_Vmatrix(self,pot):
        Vflat = pot(self.xgrid,self.ygrid).flatten()
        return sparse.spdiags([Vflat], 0, self.N*self.N,self.N*self.N)



class threedgrid(object):
    """
    wavefunction on 3D grid
    """

    def __init__(self, start=[0.,0.,0.], end=[1.,1.,1.], N=100.):
        """
        Equidistant 3D grid
        """

        basis.__init__(self)

        self.x = np.linspace(start[0], end[0], N)
        self.dx = self.x[1]-self.x[0]
        self.N = N

        self.y = np.linspace(start[1], end[1], N)
        self.dy = self.y[1] - self.y[0]

        self.z = np.linspace(start[2], end[2], N)
        self.dz = self.z[1] - self.z[0]

        self.xgrid, self.ygrid, self.zgrid = np.meshgrid(self.x, self.y, self.z)

        self.psi = np.zeros_like(self.xgrid)

        #1st Derivatives (central difference approx)
        Dx = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))/(self.dx*2.)
        Dy = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))/(self.dy*2.)
        Dz = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))/(self.dz*2.)

        #Laplacian
        Lx = (-2.*np.diag(np.ones(N),0) + np.diag(np.ones(N-1),-1) \
              + np.diag(np.ones(N-1),1))/(self.dx*self.dx)

        Ly = (-2.*np.diag(np.ones(N),0) + np.diag(np.ones(N-1),-1) \
              + np.diag(np.ones(N-1),1))/(self.dy*self.dy)

        Lz = (-2.*np.diag(np.ones(N),0) + np.diag(np.ones(N-1),-1) \
              + np.diag(np.ones(N-1),1))/(self.dz*self.dz)

        L = np.kron(np.kron(Lx, np.eye(N)),np.eye(N)) + np.kron(np.kron(np.eye(N), Ly),np.eye(N)) \
              + np.kron(np.kron(np.eye(N), np.eye(N)), Lz)

        L[0,:] = [0.]*L.shape[1]
        L[:,0] = [0.]*L.shape[0]
        L[-1,:] = [0.]*L.shape[1]
        L[:,-1] = [0.]*L.shape[0]
        self.L = L


    def __eval__(self):
        return self.psi

#    def deriv_psi(self):
#        """
#        Returns array of derivatives (result[0]: d/dx, result[1]: d/dy, result[2]: d/dz)
#        """
#        return np.array([np.dot(self.psi, self.Dx), np.dot(self.Dy, self.psi)])

    def Lap_psi(self):
        return np.dot(self.L, self.psi.flatten())#+reshape?

    def construct_Tmatrix(self):
        m = self.data.mass
        return -(hbar*hbar/2./m)*self.L

    def construct_Vmatrix(self,pot):
        return np.diag(pot(self.xgrid,self.ygrid,self.zgrid).flatten())


