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
        H = T + V     #(100,100)

        states = self.data.parameters['states']

        print 'Solving...'
        evals, evecs = eigsh(H, states, sigma=0., which='LM')
        #evals, evecs = np.sort(np.linalg.eig(H))

        self.data.wvfn.E = np.array(evals)
        self.data.wvfn.psi = np.array(evecs)     #(100,k)
        self.data.solved = True


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
            H = H.todense()
        
        print 'Solving...'
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)

        self.data.wvfn.E = np.array(E)
        self.data.wvfn.psi = np.array(psi)
        self.data.solved = True
