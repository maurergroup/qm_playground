#qmp.solver.__init__
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
Solver
"""

from qmp.solver.solver import ScipySolver, AlglibSolver

solver_type = {
        'scipy': ScipySolver,
        'alglib': AlglibSolver,
        # 'pimd': pimd_solver,
        }


def solver_init(data, potential):
    param = data.parameters
    solver = solver_type[param['solver']](data=data, potential=potential)

    return solver


def solver_help():
    string = 'Solver types include: \n'
    for key in solver_type.keys():
        string += key+'\n'
    return string
