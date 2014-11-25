"""

"""

from qmp.utilities import *

class basis(object):
    """
    The Basis class defines the quantum mechanical basis functions and 
    the operators that act on it, such as the Laplace operator, 
    differentiation etc.
    """

    def __init__(self, ndim=1):
        """
        """

        self.ndim = 1

    #def __eval__


    #def deriv_psi


    #def Lap_psi

class 1dgrid(basis):
    """
    wavefunction representation as equidistant grid
    """

    def __init__(self, ndim=1, start=0.0, end=1.0, N=100):
        """
        Equidistant 1D grid of points as wvfn representation
        """

        self.x = np.linspace(start, end, N)
        self.dx = x[1] - x[0] 
        self.N = N
        
        #wavefunction vector initialised with zeros
        self.psi = np.zeros_like(self.x)

    def __eval__():

        return self.psi

    def deriv_psi():

        #define derivative matrix with symm. finite difference
        D = (np.diag(np.ones(N),1) - np.diag(np.ones(N),-1) )/(2.*dx)


    def Lap_psi():

        L = 



