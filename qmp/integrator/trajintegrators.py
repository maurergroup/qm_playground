#    qmp.integrator.trajintegrators
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
trajintegrators.py
"""

from qmp.integrator.integrator import Integrator
from numpy.random import standard_normal
import numpy as np


class Langevin(Integrator):
    """
    Langevin integrator for classical dynamics
    This has not been rigorously tested. Quick update to work with new layout.
    """

    def updatevars(self, dt=0.1, temp=300, friction=0.1):
        self.temp = temp
        self.friction = friction
        self.dt = dt
        self.lt = self.dt*self.friction
        # If the friction is an array some other constants must be arrays too.
        lt = self.lt
        masses = self.m
        sdpos = dt * np.sqrt(self.temp / masses.reshape(-1) *
                             (2.0 / 3.0 - 0.5 * lt) * lt)
        sdpos.shape = (-1, 1)
        sdmom = np.sqrt(self.temp * masses.reshape(-1) * 2.0 * (1.0 - lt) * lt)
        sdmom.shape = (-1, 1)
        pmcor = np.sqrt(3.0) / 2.0 * (1.0 - 0.125 * lt)
        cnst = np.sqrt((1.0 - pmcor) * (1.0 + pmcor))

        act0 = 1.0 - lt + 0.5 * lt * lt
        act1 = (1.0 - 0.5 * lt + (1.0 / 6.0) * lt * lt)
        act2 = 0.5 - (1.0 / 6.0) * lt + (1.0 / 24.0) * lt * lt
        c1 = act1 * dt / masses.reshape(-1)
        c1.shape = (-1, 1)
        c2 = act2 * dt * dt / masses.reshape(-1)
        c2.shape = (-1, 1)
        c3 = (act1 - act2) * dt
        c4 = act2 * dt
        del act1, act2
        #if self._localfrict:
            ## If the friction is an array, so are these
            #act0.shape = (-1, 1)
            #c3.shape = (-1, 1)
            #c4.shape = (-1, 1)
            #pmcor.shape = (-1, 1)
            #cnst.shape = (-1, 1)
        self.sdpos = sdpos
        self.sdmom = sdmom
        self.c1 = c1
        self.c2 = c2
        self.act0 = act0
        self.c3 = c3
        self.c4 = c4
        self.pmcor = pmcor
        self.cnst = cnst

    def run(self, system, steps, potential, data, temp, friction):

        self.system = system
        self.N = self.system.n_particles
        self.ndim = self.system.ndim
        self.m = self.system.masses
        self.updatevars(self.dt, temp, friction)

        N = self.N
        ndim = self.ndim
        m = self.m

        r_t = np.zeros((steps, N, ndim))
        v_t = np.zeros((steps, N, ndim))
        r_t[0] = np.array(self.system.r)
        v_t[0] = np.array(self.system.v)

        E_pot = np.zeros((steps, N))
        E_kin = np.zeros((steps, N))
        E_pot[0] = self.system.compute_potential_energy(potential)
        E_kin[0] = self.system.compute_kinetic_energy()

        print('Integrating...')
        f = self.system.compute_force(potential)
        for i_step in range(1, steps):

            random1 = standard_normal(size=(N, ndim))
            random2 = standard_normal(size=(N, ndim))
            rrnd = self.sdpos * random1
            prnd = (self.sdmom * self.pmcor * random1 +
                    self.sdmom * self.cnst * random2)

            p = self.system.v * m[:, np.newaxis]
            self.system.r += self.c1 * p + self.c2 * f + rrnd
            p *= self.act0
            p += self.c3 * f + prnd

            f = self.system.compute_force(potential)

            self.system.v = p / m[:, np.newaxis] + self.c4 * f

            r_t[i_step] = self.system.r
            v_t[i_step] = self.system.v
            E_pot[i_step] = self.system.compute_potential_energy(potential)
            E_kin[i_step] = self.system.compute_kinetic_energy()

        E_tot = E_pot + E_kin

        print('INTEGRATED')

        data.r_t = r_t
        data.v_t = v_t
        data.E_kin_t = E_kin
        data.E_pot_t = E_pot
        data.E_t = E_tot


class VelocityVerlet(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)

        self.system = system
        N = self.system.n_particles
        ndim = self.system.ndim
        m = self.system.masses

        r_t = np.zeros((steps, N, ndim))
        v_t = np.zeros((steps, N, ndim))
        r_t[0] = np.array(self.system.r)
        v_t[0] = np.array(self.system.v)

        E_pot = np.zeros((steps, N))
        E_kin = np.zeros((steps, N))
        E_pot[0] = self.system.compute_potential_energy(potential)
        E_kin[0] = self.system.compute_kinetic_energy()

        print('Integrating...')
        for i_step in range(1, steps):

            F = self.system.compute_force(potential)
            v1 = self.system.v + F / m[:, np.newaxis] * dt / 2
            self.system.r += v1 * dt
            F = (F + self.system.compute_force(potential)) / 2
            self.system.v += F / m[:, np.newaxis] * dt

            r_t[i_step] = self.system.r
            v_t[i_step] = self.system.v
            E_pot[i_step] = self.system.compute_potential_energy(potential)
            E_kin[i_step] = self.system.compute_kinetic_energy()

        E_tot = E_pot + E_kin

        print('INTEGRATED')

        data.r_t = r_t
        data.v_t = v_t
        data.E_kin_t = E_kin
        data.E_pot_t = E_pot
        data.E_t = E_tot
        data.potential = potential.compute_cell_potential(density=1000)
