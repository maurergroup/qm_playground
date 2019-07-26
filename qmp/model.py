#qmp.model
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

from qmp.solver import solver_init
from qmp.data_containers import WaveData, TrajData, RpmdData, HopData


class Model(object):
    """
    The model class contains all information and all
    subroutines to run the calculations.

    Before it can run it has to be given a
    potential, and a basis.

    There are two basic tasks associated with the
    solver subclass (time-independent problems) and the
    dyn subclass (time  propagation)

    """

    def __init__(self, system, potential, integrator, mode='traj'):
        """
        Initializes the calculation model using arbitrary keyword arguments
        which are parsed later.
        """

        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.mode = mode
        self.solved = False

        self.data = self.prepare_data()

    # def __repr__(self):

    #     string = 'Model Summary:\n'
    #     for key in self.parameters.keys():
    #         string += key + ' : ' + str(self.parameters[key])+'\n'

    #     string += '\n'
    #     for key in self.data.keys():
    #         string += 'contains data entries for: '+key+'\n'

    #     return string

    def prepare_data(self):
        mode = self.mode
        if mode == 'wave':
            return WaveData()
        elif mode == 'traj':
            return TrajData()
        elif mode == 'rpmd':
            return RpmdData()
        elif mode == 'hop':
            return HopData()

    def set_solver(self, solver):
        self.solver = solver

    def solve(self):
        """
        Wrapper for solver.solve
        """

        try:
            self.solver.solve()
        except (AttributeError, TypeError):
            print('Initializing Solver')
            self.solver = solver_init(self.data, self.pot)
            self.solver.solve()

    def run(self, steps, **kwargs):
        """
        Wrapper for dyn.run
        """

        # if ((self.parameters['integrator'] == 'eigenprop') or
        #     (self.parameters['integrator'] == 'eigen') or
        #     (add_info == 'coefficients')) and \
        #    (self.data.solved is False):
        #     print(gray+'Projection onto eigen basis requires solving eigenvalue problem...'+endcolor)
        #     self.solve()

        self.integrator.run(self.system, steps=int(steps),
                            potential=self.potential, data=self.data)
