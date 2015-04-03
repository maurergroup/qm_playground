"""
eigensolvers.py
"""

from qmp.utilities import *
from qmp.solver.solver import solver
import numpy as np

class scipy_solver(solver):
    """
    standard scipy eigensolver
    """

    def __init__(self, **kwargs):

        solver.__init__(self, **kwargs)


    def solve(self):

        from scipy.sparse.linalg import eigsh

        basis = self.data.wvfn.basis
        T = basis.construct_Tmatrix()
        V = basis.construct_Vmatrix(self.pot)
        H = T + V

        #states = len(H)
        evals, evecs = eigsh(H, 20, sigma=0., which='LM')
        #evals, evecs = np.sort(np.linalg.eig(H))

        self.data.wvfn.E = evals
        self.data.wvfn.psi = evecs


class alglib_solver(solver):
    """
    alglib based eigensolver
    """
    def __init__(self, **kwargs):

        solver.__init__(self, **kwargs)

    def solve(self):

        try: 
            import xalglib as xa
        except:
            print 'Cannot import alglib'
            pass

        from scipy.sparse import issparse
        
        basis = self.data.wvfn.basis
        T = basis.construct_Tmatrix()
        V = basis.construct_Vmatrix(self.pot)
        H = (T + V)
        if issparse(H):
            H = np.array(H.todense())
        
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)

        self.data.wvfn.E = E
        N = self.data.wvfn.basis.N
        self.data.wvfn.psi = psi
        #if self.data.ndim == 1:
        #    self.data.wvfn.psi = np.array(psi).reshape(N,N)
        #elif self.data.ndim == 2:
        #    self.data.wvfn.psi = np.array(psi).reshape(N,N,N**2)

