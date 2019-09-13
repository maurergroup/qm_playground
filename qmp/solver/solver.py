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
Different solvers for time-independent problems.
"""

import numpy as np
from abc import ABC, abstractmethod


class Solver(ABC):
    """
    Abstract base class for all solvers.

    Attributes
    ----------
    system : qmp.systems.grid object
        The system to be operated on by the solver.
    potential : qmp.potential object
        The potential that the Hamiltonian is constructed from.
    """

    def __init__(self, system, potential):
        self.system = system
        self.potential = potential

    @abstractmethod
    def solve(self):
        """Solve the eigenvalue problem.
        Must be implemented by child classes.
        """
        pass


class ScipySolver(Solver):
    """
    Standard scipy eigensolver.

    Attributes
    ----------
    states : int
        The desired number of eigenvectors.
    """

    def __init__(self, system, potential, states):

        Solver.__init__(self, system, potential)
        self.states = states

    def solve(self):
        """Solve the eigenvalue problem.
        The results are written into system.E and system.basis.
        """

        from scipy.sparse.linalg import eigsh

        self.system.construct_hamiltonian(self.potential)
        H = self.system.H

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
    Alglib based eigensolver.
    """
    def __init__(self, system, potential):

        Solver.__init__(self, system, potential)

    def solve(self):
        """Solve the eigenvalue problem.
        The results are written into system.E and system.basis.
        """

        try:
            import xalglib as xa
        except ImportError:
            print('Cannot import alglib')
            pass

        from scipy.sparse import issparse

        self.system.construct_hamiltonian(self.potential)
        H = self.system.H
        if issparse(H):
            H = H.todense()

        print('Solving...')
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)
        print('SOLVED\n')

        self.system.E = np.array(E)
        self.system.basis = np.array(psi)
        self.system.solved = True
