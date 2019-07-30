#    qmp.model
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
model class
"""

from qmp.solver.solver import ScipySolver
from qmp.data_containers import Data


class Model:
    """
    The model class contains all information and all
    subroutines to run the calculations.

    Before it can run it has to be given a
    potential, and a basis.

    There are two basic tasks associated with the
    solver subclass (time-independent problems) and the
    dyn subclass (time  propagation)

    """

    def __init__(self, system, potential, integrator, mode, solver=None,
                 states=20):
        """
        Initializes the calculation model using arbitrary keyword arguments
        which are parsed later.
        """

        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.mode = mode
        self.states = states

        if solver is None:
            self.solver = ScipySolver(self.system, self.potential, self.states)

        self.data = self.prepare_data()

    def __repr__(self):

        string = 'Model Summary\n'
        string += 'System: ' + type(self.system).__name__ + '\n'
        string += 'Integrator: ' + type(self.integrator).__name__ + '\n'
        string += 'Mode: ' + self.mode + '\n'

        return string

    def prepare_data(self):
        mode = self.mode
        return Data()
        if mode == 'wave':
            return WaveData()
        elif mode == 'traj':
            return TrajData()
        elif mode == 'rpmd':
            return RpmdData()
        elif mode == 'hop':
            return HopData()

    def solve(self):
        """
        Wrapper for solver.solve
        """
        self.solver.solve()

    def run(self, steps, **kwargs):
        """
        Wrapper for dyn.run
        """

        add_info = kwargs.get('additional', None)

        integ = type(self.integrator).__name__
        if (integ == 'EigenPropagator' or add_info == 'coefficients') and \
                self.system.solved is False:
            print('Projection onto eigenbasis '
                  + 'requires solving eigenvalue problem...')
            self.solve()

        self.integrator.run(self.system, steps=int(steps),
                            potential=self.potential, data=self.data, **kwargs)
