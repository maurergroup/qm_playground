"""
solver.py
Different solvers for time-independent 
problems
"""

from utilities import *
import numpy as np
import scipy as sp

class solver(object):
    """
    Base class for all solver
    """

    def __init__(self):
        pass


class scipy_solver(solver):
    """
    standard scipy eigensolver
    """

    def __init__(self):

        solver.__init__(self)



    def solve(self, data, pot, basis):

        from scipy.sparse.linalg import eigsh

        T = basis.construct_Tmatrix()
        V = basis.construct_Vmatrix(pot)
        H = T + V

        #evals, evecs = eigsh(H, states, sigma=0, which='LM')
        evals, evecs = np.sort(np.linalg.eig(H))

        data.E = evals
        data.psi = evecs

        return data

class alglib_solver(solver):
    """
    alglib based eigensolver
    """
    def __init__(self):

        solver.__init__(self)

    def solve(self, data, pot, basis):

        try: 
            import xalglib as xa
        except:
            print 'Cannot import alglib'
            pass

        T = basis.construct_Tmatrix()
        V = basis.construct_Vmatrix(pot)
        H = T + V
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)

        data.E = E
        data.psi = np.array(psi).reshape([basis.N,basis.N])

        return data

class solver_init():

    solver_type = {
            'scipy': scipy_solver,
            'alglib': alglib_solver,
            #'pimd': pimd_solver,
            }
    
    def __init__(param):

        solver = solver_type[param['solver']]()


        return solver
