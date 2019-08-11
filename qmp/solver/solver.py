#    qmp.solver.solver
#
#    qm_playground - python package for dynamics simulations
#    Copyright (C) 2016  Reinhard J. Maurer
#
#    This file is part of qm_playground.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#
"""
solver.py
Different solvers for time-independent
problems
"""

import numpy as np


class Solver:
    """
    Base class for all solver
    """

    def __init__(self, system, potential):
        self.system = system
        self.potential = potential


class ScipySolver(Solver):
    """
    standard scipy eigensolver
    """

    def __init__(self, system, potential, states):

        Solver.__init__(self, system, potential)
        self.states = states

    def solve(self):

        from scipy.sparse.linalg import eigsh

        T = self.system.construct_T_matrix()
        V = self.system.construct_V_matrix(self.potential)
        H = T + V     # (100,100)

        if self.states >= H.shape[1]:
            print('Scipy solver only capable of solving for '
                  + '(grid points - 1) eigenvectors.')
            print('Adjusting number of states to '+str(H.shape[1]-1))
            self.states = H.shape[1]-1

        print('Solving...')
        evals, evecs = eigsh(H, self.states, sigma=0., which='LM')
        print('SOLVED\n')

        self.system.E = np.array(evals)
        self.system.basis = np.array(evecs)
        self.system.solved = True


class AlglibSolver(Solver):
    """
    alglib based eigensolver
    """
    def __init__(self, system, potential):

        Solver.__init__(self, system, potential)

    def solve(self):

        try:
            import xalglib as xa
        except ImportError:
            print('Cannot import alglib')
            pass

        from scipy.sparse import issparse

        T = self.system.construct_Tmatrix()
        V = self.system.construct_Vmatrix(self.potential)
        H = (T + V)
        if issparse(H):
            H = H.todense()

        print('Solving...')
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)
        print('SOLVED\n')

        self.system.E = np.array(E)
        self.system.basis = np.array(psi)
        self.system.solved = True
