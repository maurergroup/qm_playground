#    qmp.integrator.rpmdintegrators
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
"""Contains integrators for propagating ring polymers."""
import numpy as np

from qmp.tools.dyn_tools import create_thermostat

from .trajintegrators import VelocityVerlet


class RPMD_VelocityVerlet(VelocityVerlet):
    """Velocity verlet integrator for RPMD."""

    def initialise_start(self):
        super().initialise_start()
        self.system.initialise_propagators(self.dt)

    def propagate_system(self):
        """Propagate the system by a single timestep.

        This function carries out the shortened form of the velocity verlet
        algorithm. It requires that the systems taking advantage of this
        integrator implement the three functions used within it.
        """
        self.system.propagate_velocities(self.dt*0.5)
        self.system.propagate_positions()
        self.system.compute_acceleration(self.potential)
        self.system.propagate_velocities(self.dt*0.5)

    def assign_data(self, data):
        data.rb_t = np.array(self.r_t)
        data.vb_t = np.array(self.v_t)
        data.r_t = np.mean(self.r_t, 2)
        data.v_t = np.mean(self.v_t, 2)

        data.Eb_kin_t = np.array(self.E_kin)
        data.Eb_pot_t = np.array(self.E_pot)
        data.Eb_t = data.Eb_kin_t + data.Eb_pot_t
        data.E_kin_t = np.mean(data.Eb_kin_t, 2)
        data.E_pot_t = np.mean(data.Eb_pot_t, 2)
        data.E_t = np.mean(data.Eb_t, 2)

        self.save_potential(data)

    def save_potential(self, data):
        data.potential = self.potential.compute_cell_potential(density=1000)


class PIMD_LangevinThermostat(RPMD_VelocityVerlet):

    def __init__(self, dt, tau0=1):
        super().__init__(dt)
        self.tau0 = tau0
        self.ignore_centroid = False

    def initialise_start(self):
        super().initialise_start()
        self.system.initialise_thermostat(self.dt, tau0=self.tau0,
                                          ignore_centroid=self.ignore_centroid)

    def propagate_system(self):

        self.system.apply_thermostat()
        super().propagate_system()
        self.system.apply_thermostat()


class TRPMD(PIMD_LangevinThermostat):

    def __init__(self, dt, tau0=1):
        super().__init__(dt, tau0=tau0)
        self.ignore_centroid = True


class NRPMD(RPMD_VelocityVerlet):

    def initialise_start(self):
        super().initialise_start()
        self.system.initialise_propagators(self.dt/2)

        self.system.calculate_state_probability()
        self.state_occupation = [self.system.state_prob]

    def propagate_system(self):
        """Propagate the system by a single timestep.
        """
        self.system.propagate_positions()

        self.system.compute_V_matrix(self.potential)
        self.system.compute_V_prime_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_propagators(self.dt)
        self.system.propagate_mapping_variables()
        self.system.compute_adiabatic_derivative()
        self.system.compute_gamma_and_epsilon(self.dt)
        self.system.rotate_matrices()
        self.system.propagate_bead_velocities(self.dt)

        self.system.propagate_positions()

    def store_result(self):
        super().store_result()

        self.system.calculate_state_probability()
        self.state_occupation.append(self.system.state_prob)

    def save_potential(self, data):
        data.v11, data.v12, data.v22 = self.potential.compute_cell_potential(density=1000)

    def assign_data(self, data):
        super().assign_data(data)

        data.state_occupation_t = np.array(self.state_occupation)
