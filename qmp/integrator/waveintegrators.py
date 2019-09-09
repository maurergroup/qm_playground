#    qmp.integrator.waveintegrators
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
waveintegrators.py
"""

from qmp.tools.dyn_tools import project_wvfn
import numpy as np
from abc import ABC, abstractmethod
import scipy.sparse as sp
from numpy.fft import fftn
from numpy.fft import ifftn


class AbstractWavePropagator(ABC):
    """
    Abstract base class for other wave integrators to implement. Only the
    functions decorated by '@abstractmethod' should be extended, all others
    shoudl remain the same for all integrators.
    Had to leave this little picture in, someone obviously put of lot of work
    into it.
                     _               ..
       :            / \ ->          ;  ;                        :
       :           /   \ ->         ;  ;                        :
       :          /     \ ->        ;  ;                        :
       :_________/       \__________;  ;________________________:
      r_l                            rb                        r_r
    (border)       (wave)         (barrier)                  (border)
    """
    def __init__(self, dt=1):
        """
        Class is initialised with a timestep as a single argument.
        """
        self.dt = dt

    def run(self, system, steps, potential, data, **kwargs):
        """
        This function is called by the model to run the integration process.
        """
        self.system = system

        self.read_kwargs(kwargs)

        self.prepare_electronics(potential)

        self.initialise_start()

        self.integrate(steps)

        self.assign_data(data)

    def read_kwargs(self, kwargs):
        """
        Allowed keyword arguments are read here, no other keyword arguments
        should be specified.
        """
        self.dt = kwargs.get('dt', self.dt)
        self.output_freq = kwargs.get('output_freq', 200)

    @abstractmethod
    def prepare_electronics(self, potential):
        pass

    def initialise_start(self):
        """
        This function is for any initialisation that must occur before the
        integration begins, extending classes should prepare any variables
        constant throughout the integration by extending this function.
        """
        if (not self.system.psi.any() != 0.):
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

        self.status = ("Wave has not reached the cell boundary.")
        self.system.construct_imaginary_potential()
        self.propI = np.exp(-1j * self.system.imag * self.dt)
        self.psi_t = [self.system.psi]
        self.E_t = [self.compute_current_energy()]

    def integrate(self, steps):
        """
        This function carries out the main integration loop.
        """
        print('Integrating...')
        for i in range(steps):

            self.propagate_psi()

            self.absorb_boundary()

            if (i+1) % self.output_freq == 0:
                self.store_result()

            if self.is_finished():
                break

        print('INTEGRATED\n')
        print(self.status)

        self.system.detect_all()
        self.system.normalise_probabilities()
        self.psi_t = np.array(self.psi_t)

    @abstractmethod
    def propagate_psi(self):
        """
        Child classes should use this function to carry out a single timestep
        propagation.
        """
        pass

    def absorb_boundary(self):
        """
        The wavefunction is propagated by a whole timestep in the negative
        imaginary potential. This is present in the boundary regions to absorb
        the wavefunction and detect the scattering outcome.
        """
        density_before = self.system.compute_adiabatic_density()
        self.system.psi = self.propI * self.system.psi
        density_after = self.system.compute_adiabatic_density()

        absorbed_density = density_before - density_after
        self.system.detect_flux(absorbed_density)

    def store_result(self):
        """
        Lists of energies and wavefunctions are populated at each timestep.
        """
        self.E_t.append(self.compute_current_energy())

        psi = self.system.psi
        if (self.system.nstates == 2) and self.system.ndim == 1:
            psi = self.system.get_adiabatic_wavefunction()
        self.psi_t.append(psi)

    def is_finished(self):
        """
        The total amount absorbed so far is compared to the intial probability
        density, once the absorbed density reaches 0.9 of the initial density,
        the simulation is terminated. This is to avoid simulations running for
        long periods of time when density is struggling to reach the boundary.
        """
        absorbed = np.sum(self.system.exit_flux)
        absorbed_fraction = absorbed / self.system.total_initial_density
        exit = False
        if absorbed_fraction > 0.9:
            self.status = "Success, all is well."
            exit = True
        return exit

    @abstractmethod
    def compute_current_energy(self):
        """
        This function must be implemented in each child class to describe the
        energy calculation for the current wavefunction.
        """
        pass

    def assign_data(self, data):
        """
        Class variables are copied over to the data attribute of the model
        class.
        """
        data.psi_t = np.real(self.psi_t)
        data.N = self.system.N
        data.rho_t = np.real(np.conjugate(self.psi_t)*self.psi_t)
        data.outcome = self.system.exit_flux
        data.E_t = np.array(self.E_t)
        data.V = np.real(self.system.V.A)


class PrimitivePropagator(AbstractWavePropagator):
    """
    Primitive exp(-iHt) propagator for psi in arbitrary
    basis in spatial representation.
    """

    def prepare_electronics(self, potential):
        self.system.construct_hamiltonian(potential)
        self.prop = -1j * self.system.H * self.dt

    def propagate_psi(self):
        self.system.psi = sp.linalg.expm_multiply(self.prop, self.system.psi)

    def compute_current_energy(self):
        psi = self.system.psi
        return np.real(psi.conj().dot(self.system.H.dot(psi)))


class SOFT_Propagator(AbstractWavePropagator):
    """
    Split operator Fourier transform propagator
    Follows approach detailed in section 11.7 of David J. Tannor's
    "Introduction to Quantum Mechanics".
    """
    def prepare_electronics(self, potential):
        self.system.construct_V_matrix(potential)
        self.system.compute_k()
        self.propT = self.expT(self.dt/2)
        self.propV = self.expV(self.dt)

    def expV(self, dt):

        if self.system.nstates == 2:
            col1, col2 = np.array_split(self.system.V.A, 2)
            v1, v21 = np.array_split(col1, 2, axis=1)
            v12, v2 = np.array_split(col2, 2, axis=1)
            v1 = np.diag(v1)
            v2 = np.diag(v2)
            v12 = np.diag(v12)
            v21 = np.diag(v21)

            D = (v1 - v2)**2 + 4*v21**2

            diagonal = np.exp(-1.0j * (v1+v2) * dt/2)

            cos = np.cos(np.sqrt(D)*dt/2)
            x = np.tile(diagonal * cos, self.system.nstates)
            left = sp.diags(x)

            sin = 1.0j * np.sin(np.sqrt(D)*dt/2)/np.sqrt(D) * diagonal
            v_matrix = sp.bmat([[sp.diags(sin*(v2-v1)), -2*sp.diags(sin*v12)],
                                [-2*sp.diags(sin*v21), sp.diags(sin*(v1-v2))]])

            return left + v_matrix

        elif self.system.nstates == 1:
            return sp.diags(np.exp(-1j * self.system.V.diagonal() * dt))

    def expT(self, dt):
        return np.exp(-1j * self.system.momentum_T * dt)

    def propagate_psi(self):
        self.propagate_momentum()
        self.system.psi = self.propV.dot(self.system.psi)
        self.propagate_momentum()

    def propagate_momentum(self):
        self.system.psi = self.system.transform(self.system.psi, fftn)
        self.system.psi = self.propT * self.system.psi
        self.system.psi = self.system.transform(self.system.psi, ifftn)

    def compute_current_energy(self):
        E_kin = self.system.compute_kinetic_energy()
        E_pot = self.system.compute_potential_energy()
        return E_pot + E_kin


# class EigenPropagator(AbstractWavePropagator):
#     """
#     Projects initial wavefunction onto eigenbasis,
#     propagates expansion coefficients.
#     Currently limited to a single energy level.
#     Also, seems to be pretty dodgy, not sure what I did to break it.
#     """
#     def initialise_start(self, system, potential):

#         super().initialise_start(system, potential)
#         self.system.c = self.prepare_coefficients()

#         self.prop = np.diag(np.exp(-1j*self.system.E*self.dt))
#         self.psi_t = [self.system.basis.dot(self.system.c)]
#         self.c_t = [self.system.c]
#         self.E_t = [self.compute_current_energy()]

#         if np.all(self.system.psi) == 0.:
#             raise ValueError('Please provide initial wave function'
#                              + ' on appropriate grid')

#     def prepare_coefficients(self):
#         states = self.system.basis.shape[1]
#         print('Projecting wavefunction onto basis of '
#               + str(states) + ' eigenstates')
#         if self.system.basis.shape[0] != states:
#             print('** WARNING: This basis is incomplete,'
#                   + ' coefficients might contain errors **')

#         return project_wvfn(self.system.psi, self.system.basis)

#     def propagate_psi(self):
#         self.system.c = self.prop.dot(self.system.c)
#         self.system.psi = self.system.basis.dot(self.system.c)

#     def store_result(self):
#         self.psi_t.append(self.system.psi)
#         self.c_t.append(self.system.c)

#     def compute_current_energy(self):
#         return self.system.c.conj() * self.system.E * self.system.c

#     def assign_data(self, data):
#         super().assign_data(data)
#         data.rho_mean = np.mean(data.rho_t, 0)
#         data.c_t = np.array(self.c_t)
#         data.E_t = np.array(self.E)
