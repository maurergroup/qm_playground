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
"""Contains integrators to propagate grid systems through time."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from .integrator import Integrator, SimulationTerminated


class WavePropagator(Integrator, ABC):
    """Base class for other wave integrators to implement."""

    def __init__(self, dt=1, output_interval=1, absorb=True, output_adiabatic=True):
        """Class is initialised with a timestep as a single argument."""
        super().__init__(dt, output_interval)
        self.absorb = absorb
        self.output_adiabatic = output_adiabatic

    def _read_kwargs(self, kwargs):
        """Allowed keyword arguments are read here.

        Parameters
        ----------
        kwargs : {'dt', 'output_freq'}
        """
        self.dt = kwargs.get('dt', self.dt)
        self.output_freq = kwargs.get('output_freq', 200)

    def _initialise_start(self):
        """Initialise logging variables and imaginary potential.

        This function should likely be extended to add extra functionality, but
        not completely overrided.
        """
        if (not self.system.psi.any() != 0.):
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

        self._prepare_electronics()
        self.status = ("Wave has not reached the cell boundary.")
        self.system.construct_imaginary_potential()
        self.propI = np.exp(-1j * self.system.imag * self.dt)
        self.psi_t = [self.system.psi]
        self.E_t = [self._compute_current_energy()]

    @abstractmethod
    def _prepare_electronics(self):
        """Prepare electronic quantities required for the propagation.
        """

    def _perform_timestep(self, iteration):
        self._propagate_system()

        if self.absorb:
            self._absorb_boundary()

        if (iteration+1) % self.output_freq == 0:
            self._store_result()

        if self._is_finished():
            raise SimulationTerminated

    @abstractmethod
    def _propagate_system(self):
        """Propagate the wavefunction by a single timestep.
        """

    def _absorb_boundary(self):
        """Absorb the wavefunction in the boundary region.

        The wavefunction is propagated by a whole timestep in the negative
        imaginary potential. This is present in the boundary regions to absorb
        the wavefunction and detect the scattering outcome. I have not seen
        this done in this way elsewhere but it allows a convenient way to
        detect the density and absorb the wavefunction at the boundary. The
        methodology is based upon the Strang splitting idea of the potential
        and the imaginary potential.
        """
        density_before = self.system.compute_adiabatic_density()
        self.system.psi = self.propI * self.system.psi
        density_after = self.system.compute_adiabatic_density()

        absorbed_density = density_before - density_after
        self.system.detect_flux(absorbed_density)

    def _store_result(self):
        """Lists of energies and wavefunctions are populated at each timestep.
        """
        self.E_t.append(self._compute_current_energy())

        psi = self.system.psi
        if self.output_adiabatic and (self.system.nstates == 2) and self.system.ndim == 1:
            psi = self.system.get_adiabatic_wavefunction()
        self.psi_t.append(psi)

    def _is_finished(self):
        """Check if sufficient probability density has been absorbed.

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
    def _compute_current_energy(self):
        """Compute the current energy of the system."""
        pass

    def _assign_data(self, data):
        """Assign the relevant quantities to the data object.

        Class variables are copied over to the data attribute of the model
        class.
        """
        print(self.status)

        self.system.detect_all()
        self.system.normalise_probabilities()
        self.psi_t = np.array(self.psi_t)

        data.psi_t = np.real(self.psi_t)
        data.N = self.system.N
        data.rho_t = np.real(np.conjugate(self.psi_t)*self.psi_t)
        data.outcome = self.system.exit_flux
        data.E_t = np.array(self.E_t)
        data.V = np.real(self.system.V.A)


class PrimitivePropagator(WavePropagator):
    """Standard exp(-iHt) propagation for the wavefunction.

    Primitive exp(-iHt) propagator for psi in arbitrary
    basis in spatial representation.
    """

    def _prepare_electronics(self):
        """Construct the hamiltonian and the propagator.

        Parameters
        ----------
        potential : qmp.potential.potential.Potential
            The potential that the hamiltonian is constructed from.
        """
        self.system.construct_hamiltonian(self.potential)
        self.prop = -1j * self.system.H * self.dt

    def _propagate_system(self):
        """Propagate the wavefunction by one timestep."""
        self.system.psi = sp.linalg.expm_multiply(self.prop, self.system.psi)

    def _compute_current_energy(self):
        """Compute the expectation value of the Hamiltonian."""
        psi = self.system.psi
        return np.real(psi.conj().dot(self.system.H.dot(psi)))


class SOFT_Propagator(WavePropagator):
    """Split Operator Fourier Transform propagator.

    Follows approach detailed in section 11.7 of David J. Tannor's
    "Introduction to Quantum Mechanics".
    """
    def _prepare_electronics(self):
        """Construct the potential and kinetic energy propagators.

        Parameters
        ----------
        potential : qmp.potential.potential.Potential
            The potential that the propagators are constructed from.
        """
        self.system.construct_V_matrix(self.potential)
        self.system.compute_k()
        self.propT = self._expT(self.dt/2)
        self.propV = self._expV(self.dt)

    def _expV(self, dt):
        """Construct the potential energy propagator.

        Check out section 11.7 in Tannor's book for information on the
        transformations going on in the two state case.

        Parameters
        ----------
        dt : int or float
            The timestep of the propagation. For the symmetric split operator
            the central operator (this one in this case), should be double that
            of the other operator.

        Returns
        -------
        array_like
            A matrix representing the propagator, for a single state this will
            be diagonal, for 2 states, as seen below, things are a little more
            complicated.
        """

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

    def _expT(self, dt):
        """Construct the kinetic energy propagator.

        Parameters
        ----------
        dt : int or float
            The timestep for the propagator, in the current setup this should
            be half that of the timestep for the potential propagator.

        Returns
        -------
        array_like
            Vector representing the propagator.
        """
        return np.exp(-1j * self.system.momentum_T * dt)

    def _propagate_system(self):
        """Propagate the wavefunction by a single timestep."""
        self._propagate_momentum()
        self.system.psi = self.propV.dot(self.system.psi)
        self._propagate_momentum()

    def _propagate_momentum(self):
        """Propagate the momentum by a single timestep.

        This involves a unitary transformation into momentum space to allow for
        convenient operation of the kinetic energy operator, which is diagonal
        in this representation.
        """
        self.system.psi = self.system.transform(self.system.psi, np.fft.fftn)
        self.system.psi = self.propT * self.system.psi
        self.system.psi = self.system.transform(self.system.psi, np.fft.ifftn)

    def _compute_current_energy(self):
        """Compute the energy of the current wavefunction.

        Here we calculate the kinetic and potential components separately and
        add them together, innovative.
        """
        E_kin = self.system.compute_kinetic_energy()
        E_pot = self.system.compute_potential_energy()
        return E_pot + E_kin
