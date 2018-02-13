#qmp.solver.eigensolver
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
eigensolvers.py
"""

from qmp.tools.utilities import *
from qmp.solver.solver import solver
from qmp.tools.termcolors import *
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
        if states >= H.shape[1]:
        	print( gray+'Scipy solver only capable of solving for (grid points - 1) eigen vectors.')
        	print( 'Adjusting number of states to '+str(H.shape[1]-1)+endcolor)
        	states = H.shape[1]-1

        print (gray+'Solving...'+endcolor)
        evals, evecs = eigsh(H, states, sigma=0., which='LM')
        print (gray+'SOLVED\n'+endcolor)

        self.data.wvfn.E = np.array(evals)
        self.data.wvfn.psi = np.array(evecs)
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
            print( red+'Cannot import alglib'+endcolor)
            pass

        from scipy.sparse import issparse
        
        basis = self.data.wvfn.basis
        T = basis.construct_Tmatrix()
        V = basis.construct_Vmatrix(self.pot)
        H = (T + V)
        if issparse(H):
            H = H.todense()
        
        print( gray+'Solving...'+endcolor)
        result, E, psi = xa.smatrixevd(H.tolist(), H.shape[0], 1, 1)
        print( gray+'SOLVED\n'+endcolor)

        self.data.wvfn.E = np.array(E)
        self.data.wvfn.psi = np.array(psi)
        self.data.solved = True


#--EOF--#
