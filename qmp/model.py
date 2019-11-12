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
from qmp.solver.solver import ScipySolver
from qmp.data_containers import Data
import pickle


class Model:
    """Instances of this class are used to run calculations in qmp.

    The two functions of the class are the run and solve methods. These
    correspond to time-dependent and time-independent problems respectively.

    Typical usage of qmp entails instantiating this class with the
    necessary settings and calling either run or solve.
    """

    def __init__(self, system, potential, mode, integrator=None, solver=None,
                 states=20):
        """
        Initialise the calculation model.

        Parameters
        ----------
        system : qmp.systems object
            This system to be simulated.
        potential : qmp.potential.potential.Potential
            The potential of the simulation.
        mode : {'wave', 'rpmd', 'hop', 'traj'}
            A label that helps track the type of simulation being carried out.
        integrator : qmp.integrator object, optional
            The integrator used to propagate the system.
        solver : qmp.solver.solver, optional
            The eigensolver used to solve the eigenvalue problem.
        states : int, optional
            The number of eigenvectors found by the eigensolver.
        name : str, optional
            The name of the output file.
        """

        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.mode = mode
        self.states = states

        if solver is None:
            self.solver = ScipySolver(self.system, self.potential, self.states)

        self.prepare_data()

    def __repr__(self):

        string = 'Model Summary\n'
        string += 'System: ' + type(self.system).__name__ + '\n'
        string += 'Integrator: ' + type(self.integrator).__name__ + '\n'
        string += 'Mode: ' + self.mode + '\n'

        return string

    def prepare_data(self):
        """Create the data object.

        The data object is basically just a dictionary right now and is not
        really necessary but could be developed later."""
        self.data = Data()
        self.data.mode = self.mode
        self.data.integrator = type(self.integrator).__name__
        self.data.cell = self.potential.cell

    def set_integrator(self, integrator):
        self.integrator = integrator

    def solve(self):
        """
        Wrapper for solver.solve
        """
        self.solver.solve()

    def run(self, steps, **kwargs):
        """
        Wrapper for integrator.run
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

        print('Simulation complete.\nData contains the following entries:')
        for key in self.data:
            print(key)

    def write_output(self, name='simulation'):

        print(f'Writing results to \'{name}\'.')
        out = open(name, 'wb')
        pickle.dump(self.data, out)
