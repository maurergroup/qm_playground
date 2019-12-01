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
"""Integrators that propagate the PhaseSpace system."""

from abc import ABC, abstractmethod
import progressbar

import numpy as np
from numpy.random import standard_normal


class AbstractVelocityVerlet(ABC):
    """Abstract base class for other classical integrators to implement.

    Only the functions decorated by '@abstractmethod' should be extended,
    all others should remain the same for all integrators.

    Attributes
    ----------
    dt : float or int
    system : qmp.systems.phasespace.PhaseSpace
    potential : qmp.potential.potential.Potential
    current_acc : array_like
        Current acceleration of each particle.
    """
    def __init__(self, dt=1):
        """Class is initialised with a timestep as a single argument."""
        self.dt = dt

    def run(self, system, steps, potential, data, **kwargs):
        """This function is called by the model to run the integration process.

        This function handles the operation of the integrator. The integrator
        is first set up and then the system is integrated for the specified
        number of steps. Finally, the data object is assigned.

        Parameters
        ----------
        system : qmp.systems.phasespace.PhaseSpace
        steps : int
        potential : qmp.potential.potential.Potential
        data : qmp.data_containers.Data
        kwargs : {'dt', 'output_freq'}
        """

        self.system = system
        self.potential = potential

        self.read_kwargs(kwargs)

        self.initialise_start()

        self.integrate(steps)

        self.assign_data(data)

    def read_kwargs(self, kwargs):
        """Allowed keyword arguments are read here.

        Parameters
        ----------
        kwargs : {'dt', 'output_freq'}
        """
        self.dt = kwargs.get('dt', self.dt)
        self.output_freq = kwargs.get('output_freq', 2)

    @abstractmethod
    def initialise_start(self):
        """Prepare any logging variables and calculate intial values."""
        pass

    def integrate(self, steps):
        """Carry out main integration loop.

        Parameters
        ----------
        steps : int
            The number of steps.
        """
        print('Integrating...')

        self.system.compute_acceleration(self.potential)
        for i in progressbar.progressbar(range(steps)):

            self.propagate_system()

            if (i+1) % self.output_freq == 0:
                self.store_result()

        print('INTEGRATED')

    def propagate_system(self):
        """Propagate the system by a single timestep.

        This function carries out the shortened form of the velocity verlet
        algorithm. It requires that the systems taking advantage of this
        integrator implement the three functions used within it.
        """
        self.system.propagate_velocities(self.dt*0.5)
        self.system.propagate_positions(self.dt)
        self.system.compute_acceleration(self.potential)
        self.system.propagate_velocities(self.dt*0.5)

    @abstractmethod
    def store_result(self):
        """Store the results of the current step."""
        pass

    @abstractmethod
    def assign_data(self, data):
        """Assign the data at the end of the simulation."""
        pass


class VelocityVerlet(AbstractVelocityVerlet):
    """Velocity verlet integrator for classical dynamics."""

    def initialise_start(self):
        self.r_t = [self.system.r]
        self.v_t = [self.system.v]

        self.E_pot = [self.system.compute_potential_energy(self.potential)]
        self.E_kin = [self.system.compute_kinetic_energy()]

    def store_result(self):
        self.r_t.append(self.system.r)
        self.v_t.append(self.system.v)

        self.E_pot.append(self.system.compute_potential_energy(self.potential))
        self.E_kin.append(self.system.compute_kinetic_energy())

    def assign_data(self, data):
        data.r_t = np.array(self.r_t)
        data.v_t = np.array(self.v_t)
        data.E_kin_t = np.array(self.E_kin)
        data.E_pot_t = np.array(self.E_pot)
        data.E_t = data.E_kin_t + data.E_pot_t
        data.potential = self.potential.compute_cell_potential(density=1000)


class Langevin:
    """
    TODO tidy this up.
    Langevin integrator for classical dynamics
    This has not been rigorously tested. Quick update to work with new layout.
    """

    def __init__(self, dt=1):
        self.dt = dt

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
