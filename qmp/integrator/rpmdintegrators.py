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
from qmp.integrator.trajintegrators import VelocityVerlet
from qmp.tools.dyn_tools import create_thermostat
import numpy as np


class RPMD_VelocityVerlet(VelocityVerlet):
    """Velocity verlet integrator for RPMD."""

    def initialise_start(self):
        super().initialise_start()
        self.system.initialise_propagators(self.dt)

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
        super().__init__(dt)
        self.tau0 = tau0
        self.ignore_centroid = False

    def initialise_start(self):
        super().initialise_start()
        self.system.initialise_thermostat(self.dt, tau0=self.tau0,
                                          ignore_centroid=self.ignore_centroid)
