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

        self.x, self.dx = np.linspace(start, end, N, retstep=True)
        self.N = N
        
        #wavefunction vector initialised with zeros
        self.psi = np.zeros_like(self.x)

        #define derivative matrix with symm. finite difference
        D = sparse.lil_matrix(np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1) )
        #pbc
        #D[0,:] =  sparse.lil_matrix([0.]*D.shape[1])
        #D[:,0] =  sparse.lil_matrix([0.]*D.shape[0]).T
        #D[-1,:] = sparse.lil_matrix([0.]*D.shape[1])
        #D[:,-1] = sparse.lil_matrix([0.]*D.shape[0]).T
        D[0,-1] = 1.
        D[-1,0] = 1.
        self.D = sparse.lil_matrix(D/(2.*self.dx))

        #define Laplace operator
        L = sparse.lil_matrix(-2*np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1) \
                + np.diag(np.ones(N-1),-1))
        #L[0,:] =  sparse.lil_matrix([0.]*L.shape[1])
        #L[:,0] =  sparse.lil_matrix([0.]*L.shape[0]).T
        #L[-1,:] = sparse.lil_matrix([0.]*L.shape[1])
        #L[:,-1] = sparse.lil_matrix([0.]*L.shape[0]).T
        L[0,-1] = 1.
        L[-1,0] = 1.

        self.L = sparse.lil_matrix(L/(self.dx*self.dx))
        

    def __eval__(self):
        return self.psi

    def deriv_psi(self):
        return np.dot(self.D,self.psi)
    
    def Nabla_psi(self):
        return np.dot(self.D,self.psi)

    def Lap_psi(self):
        return np.dot(self.L,self.psi)

    def construct_Tmatrix(self):
        m = self.data.mass
        return -(1./2.)*((hbar**2)/m)*self.L

    def construct_Vmatrix(self,pot):
        return np.diag(pot(self.x))
    
    def get_potential_flat(self,pot):
        return pot(self.x)
        

class  twodgrid(basis):
    """
    wavefunction on equidistant 2D grid
    """

    def __init__(self, start=[0.,0.], end=[1.,1.], N=100.):
        """
        Equidistant 2D grid
        """

        basis.__init__(self)

        self.x, self.dx = np.linspace(start[0], end[0], N, retstep=True)
        self.N = N
        self.y, self.dy = np.linspace(start[1], end[1], N, retstep=True)

        self.xgrid, self.ygrid = np.meshgrid(self.x, self.y)

        self.psi = np.zeros_like(self.xgrid)

        #1st Derivatives (central difference approx)
        Dx = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))
        Dy = (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))

        Dx[0,:] = [0.]*Dx.shape[1]
        Dx[:,0] = [0.]*Dx.shape[0]
        Dx[-1,:] = [0.]*Dx.shape[1]
        Dx[:,-1] = [0.]*Dx.shape[0]
        Dx[0,-1] = 1.
        Dx[-1,0] = 1.
        self.Dx = sparse.lil_matrix(Dx/(self.dx*2.))

        Dy[0,:] = [0.]*Dy.shape[1]
        Dy[:,0] = [0.]*Dy.shape[0]
        Dy[-1,:] = [0.]*Dy.shape[1]
        Dy[:,-1] = [0.]*Dy.shape[0]
        Dy[0,-1] = 1.
        Dy[-1,0] = 1.
        self.Dy = sparse.lil_matrix(Dy/(self.dy*2.))

        #Laplacian
        A, b = -2.*np.eye(N), np.ones(N-1)
        L1 = sparse.lil_matrix(A+np.diagflat(b,-1)+np.diagflat(b,1))

        L = (sparse.kron(L1, np.eye(N)) + sparse.kron(np.eye(N), L1)).tolil()

        #L[0,:] =  sparse.lil_matrix([0.]*L.shape[1])
        #L[:,0] =  sparse.lil_matrix([0.]*L.shape[0]).T
        #L[-1,:] = sparse.lil_matrix([0.]*L.shape[1])
        #L[:,-1] = sparse.lil_matrix([0.]*L.shape[0]).T
        L[0,-1] = 1.
        L[-1,0] = 1.
        self.L = sparse.lil_matrix(L/(self.dx**2*self.dy**2))


    def __eval__(self):
        return self.psi

    def deriv_psi(self):
        """
        Returns array of derivatives (result[0]: d/dx, result[1]: d/dy)
        """
        return np.array([np.dot(self.psi, self.Dx), np.dot(self.Dy, self.psi)])
    
    def Nabla_psi(self):
        return np.dot(self.psi, self.Dx) + np.dot(self.Dy, self.psi)

    def Lap_psi(self):
        return (self.L.dot(self.psi.flatten())).reshape(self.N, self.N)

    def construct_Tmatrix(self):
        m = self.data.mass
        return -(hbar*hbar/2./m)*self.L

    def construct_Vmatrix(self,pot):
        Vflat = pot(self.xgrid,self.ygrid).flatten()
        return sparse.diags(Vflat, 0, (self.N*self.N,self.N*self.N))
    
    def get_potential_flat(self,pot):
        return pot(self.xgrid,self.ygrid).flatten()


