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

from qmp.tools.utilities import hbar
from qmp.integrator.integrator import Integrator
from qmp.integrator.dyn_tools import project_wvfn
import numpy as np
from abc import ABC, abstractmethod


class AbstractWavePropagator(ABC):
    """
    Abstract base class for other wave integrators to implement.
    """
    def __init__(self, dt=1):
        self.dt = dt

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """

        self.read_kwargs(kwargs)
        self.initialise_start(system, potential)

        self.integrate(steps)

        self.compute_energies()
        self.assign_data(data)

    def read_kwargs(self, kwargs):
        self.dt = kwargs.get('dt', self.dt)
        self.output_freq = kwargs.get('output_freq', 200)

    @abstractmethod
    def initialise_start(self, system, potential):
        pass

    def integrate(self, steps):
        output_index = 0
        print('Integrating...')
        for i in range(steps):

            self.propagate_psi()

            if (i+1) % self.output_freq == 0:
                output_index += 1
                self.store_result()

        print('INTEGRATED\n')
        self.psi_t = np.array(self.psi_t)

    @abstractmethod
    def propagate_psi(self):
        pass

    def store_result(self):
        self.psi_t.append(self.system.psi)

    @abstractmethod
    def compute_energies(self):
        pass

    def assign_data(self, data):
        data.x = np.array(self.system.x)
        data.psi_t = np.array(self.psi_t)
        data.rho_t = np.conjugate(data.psi_t)*data.psi_t


class EigenPropagator(AbstractWavePropagator):
    """
    Projects initial wavefunction onto eigenbasis,
    propagates expansion coefficients.
    Currently limited to a single energy level.
    """
    def initialise_start(self, system, potential):

        self.system = system
        self.system.c = self.prepare_coefficients()

        self.prop = np.diag(np.exp(-1j*self.system.E*self.dt))
        self.psi_t = [self.system.basis.dot(self.system.c)]
        self.c_t = [self.system.c]

        if np.all(self.system.psi) == 0.:
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

    def prepare_coefficients(self):
        states = self.system.basis.shape[1]
        print('Projecting wavefunction onto basis of '
              + str(states) + ' eigenstates')
        if self.system.basis.shape[0] != states:
            print('** WARNING: This basis is incomplete,'
                  + ' coefficients might contain errors **')

        return project_wvfn(self.system.psi, self.system.basis)

    def propagate_psi(self):
        self.system.c = self.prop.dot(self.system.c)
        self.system.psi = self.system.basis.dot(self.system.c)

    def store_result(self):
        self.psi_t.append(self.system.psi)
        self.c_t.append(self.system.c)

    def compute_energies(self):
        self.c_t = np.array(self.c_t)
        E_times_c = self.system.E * self.c_t
        self.E = np.einsum('ik,ik->i', self.c_t.conj(), E_times_c)

    def assign_data(self, data):
        data.psi_t = self.psi_t
        data.rho_t = np.conj(data.psi_t)*data.psi_t
        data.rho_mean = np.mean(data.rho_t, 0)
        data.c_t = np.array(self.c_t)
        data.E_t = np.array(self.E)


class PrimitivePropagator(AbstractWavePropagator):
    """
    Primitive exp(-iHt) propagator for psi in arbitrary
    basis in spatial representation.
    Could look at J. Chem. Phys. 127, 044109 (2007)
    """
    def initialise_start(self, system, potential):

        self.system = system
        self.psi_t = [self.system.psi]
        if (not self.system.psi.any() != 0.):
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')
        self.prepare_electronics(potential)

    def prepare_electronics(self, potential):
        import scipy.sparse.linalg as la

        V = self.system.construct_V_matrix(potential)
        T = self.system.construct_T_matrix()

        self.H = T + V

        self.prop = la.expm(-1j * self.H * self.dt)

    def propagate_psi(self):

        self.system.psi = self.prop.dot(self.system.psi)

    def compute_energies(self):

        E_t = np.zeros((len(self.psi_t)))
        for t, time in enumerate(self.psi_t):
            E_t[t] = np.real(time.conj().dot(self.H.dot(time)))
        self.E = E_t

    def assign_data(self, data):
        data.x = np.array(self.system.x)
        data.psi_t = np.array(self.psi_t)
        data.rho_t = np.conjugate(data.psi_t)*data.psi_t
        data.E_t = self.E


class SOFT_Propagator(AbstractWavePropagator):
    """
    Split operator Fourier transform propagator
    Follows approach detailed in section 11.7 of David J. Tannor's
    "Introduction to Quantum Mechanics".
    """

    def initialise_start(self, system, potential):
        self.system = system
        self.V = self.system.construct_V_matrix(potential)
        self.k = self.compute_k()

        self.psi_t = [self.system.psi]
        self.propT = self.expT(self.dt/2)
        self.propV = self.expV(self.dt)

    def compute_k(self):
        from numpy.fft import fftfreq as FTp

        dx = self.system.dx
        nx = self.system.N
        ndim = self.system.ndim

        if ndim == 1:
            k = 2 * np.pi * FTp(nx, dx)
        elif ndim == 2:
            k = FTp(nx, dx).conj()*FTp(nx, dx)
            k = np.pi*np.pi*(np.kron(np.ones(nx), k)
                             + np.kron(k, np.ones(nx)))
        else:
            raise NotImplementedError('Only 1D and 2D systems implemented')

        return k

    def expV(self, dt):
        import scipy.linalg as la
        try:
            N = self.system.N
            v1 = np.diag(self.V[:N, :N])
            v2 = np.diag(self.V[N:, N:])
            v12 = np.diag(self.V[:N, N:])
            v21 = np.diag(self.V[N:, :N])

            D = (v1 - v2)**2 + 4*v21**2

            diagonal = np.exp(-1.0j * (v1+v2) * dt/2)

            cos = np.cos(np.sqrt(D)*dt/2)
            x = diagonal * cos
            left = la.block_diag(np.diag(x), np.diag(x))

            sin = 1.0j * np.sin(np.sqrt(D)*dt/2)/np.sqrt(D) * diagonal
            v_matrix = np.block([[np.diag(sin*(v2-v1)), -2*np.diag(sin*v12)],
                                 [-2*np.diag(sin*v21), np.diag(sin*(v1-v2))]])

            return left + v_matrix

        except ValueError:
            print('Single energy surface')

            exp = np.zeros_like(self.V, dtype=complex)
            for i in range(self.system.N):
                exp[i, i] = np.exp(-1.0j * self.V[i, i] * dt)
            return exp

    def expT(self, dt):
        T = self.get_T()
        exp = np.zeros_like(T, dtype=complex)
        for i in range(self.system.N * self.system.nstates):
            exp[i, i] = np.exp(-1.0j * T[i, i] * dt)
        return exp

    def get_T(self):
        import scipy.linalg as la
        m = self.system.mass
        T = self.k**2 / (2.0 * m)
        T = np.diag(T)
        if self.system.nstates == 2:
            T = la.block_diag(T, T)
        return T

    def propagate_psi(self):
        self.propagate_momentum()
        self.system.psi = self.propV.dot(self.system.psi)
        self.propagate_momentum()
        pass

    def propagate_momentum(self):
        from numpy.fft import fft
        from numpy.fft import ifft

        self.system.psi = self.transform(self.system.psi, fft)
        self.system.psi = self.propT.dot(self.system.psi)
        self.system.psi = self.transform(self.system.psi, ifft)

    def transform(self, psi, transform):
        split = np.array(np.split(psi, self.system.nstates))
        psi_transformed = transform(split).reshape(self.system.N
                                                   * self.system.nstates)
        return psi_transformed

    def compute_energies(self):
        from numpy.fft import fft
        from numpy.fft import ifft

        T = self.get_T()

        E_pot_t = np.zeros(len(self.psi_t))
        E_kin_t = np.zeros(len(self.psi_t))

        for t, time in enumerate(self.psi_t):
            conj = time.conj()
            T_dot_psi = T.dot(self.transform(time, fft))
            E_kin_t[t] = np.real(conj.dot(self.transform(T_dot_psi, ifft)))
            E_pot_t[t] = np.real(conj.dot(self.V.dot(time)))

        self.E_pot_t = E_pot_t
        self.E_kin_t = E_kin_t

    def assign_data(self, data):
        data.x = np.array(self.system.x)
        data.psi_t = np.array(self.psi_t)
        data.rho_t = np.conjugate(data.psi_t)*data.psi_t
        data.E_kin_t = self.E_kin_t
        data.E_pot_t = self.E_pot_t
        data.E_t = self.E_kin_t + self.E_pot_t


# class SOFT_AverageProperties(SOFT_Propagator):
#     """
#     SOFT propagator for psi(r,0) to determine expectation values

#     output:
#     =======
#         rho:   average density at r, \sum_{steps} |psi(r,step)|^2/steps
#         E_tot: average total energy, \sum_{steps} E_tot/steps (should be constant)
#         E_kin: average kinetic energy, \sum_{steps} E_kin/steps
#         E_pot: average potential energy, \sum+{steps} E_pot/steps
#     """

#     def store_result(self, add_info):
#         from numpy.fft import fft as FT
#         from numpy.fft import ifft as iFT

#         psi = self.system.psi
#         m = self.system.mass

#         e_kin = np.conj(psi).dot(iFT(self.k**2/(2*m) * FT(psi)))
#         e_pot = np.conj(psi).dot(self.V*psi)

#         try:
#             self.rho += np.conj(psi)*psi
#             self.E_kin += e_kin
#             self.E_pot += e_pot
#         except AttributeError:
#             self.rho = np.conj(psi)*psi
#             self.e_kin = e_kin
#             self.e_pot = e_pot

#         if add_info == 'coefficients':
#             try:
#                 self.c_mean += project_wvfn(self.psi, self.psi_basis)
#             except AttributeError:
#                 self.c_mean = project_wvfn(self.psi, self.psi_basis)

#     def assign_data(self, data, i, add_info):
#         if add_info == 'coefficients':
#             data.c_mean = np.array(self.c_mean/i)

#         data.E_kin = np.real(self.e_kin / i)
#         data.E_pot = np.real(self.e_pot / i)
#         data.rho = np.real(self.rho/i)

#         data.E = data.E_kin + data.E_pot


# TODO I think there is a better, more general way of doing this.
# class SOFT_Scattering(SOFT_Propagator):
#     """
#     SOFT propagator for scattering processes
#                      _               ..
#        :            / \ ->          ;  ;                        :
#        :           /   \ ->         ;  ;                        :
#        :          /     \ ->        ;  ;                        :
#        :_________/       \__________;  ;________________________:
#       r_l                            rb                        r_r
#     (border)       (wave)         (barrier)                  (border)

#     Stops if reflected or transmitted part of wave package hits
#     border at r_l or r_r, respectively (=> t_stop).

#     input:
#     ======
#         wave:     defined by psi_0 (incl. momentum!)
#         barrier:  defined by potential class
#         div_surf: dividing surface (1D: rb), given as keyword argument

#     output:
#     =======
#         psi_t:    final wave package at t_stop
#         p_refl:   integrated reflected part of wave, \int rho(r,t_stop) for r<rb
#         p_trans:  integrated transmitted part of wave, \int rho(r,t_stop) for r>rb
#         energy:   E_kin, E_pot, E_tot as functions of simulation time
#         status:   information whether scattering process should be complete
#     """

#     # get borders, r_l and r_b, and dividing surface (1D: rb)
#     # TODO: 2D dividing surface?

#     def store_result(self, psi, e_kin, e_pot):

#         self.rho += np.conjugate(psi)*psi
#         self.psi.append(psi)
#         self.E_kin.append(e_kin)
#         self.E_pot.append(e_pot)
#         self.E.append(e_kin+e_pot)
#         self.E_mean = np.mean(self.E)

#     def assign_data(self, data, i, add_info):
#         data.psi_t = np.array(self.psi)
#         data.E_t = np.array(self.E)
#         data.E_mean = np.array(self.E_mean)
#         data.dErel_max = np.array(
#                 abs(max(abs(self.E-self.E_mean))/self.E_mean))
#         data.E_kin_t = np.array(self.E_kin)
#         data.E_pot_t = np.array(self.E_pot)
#         data.p_refl = np.array(self.p_refl)
#         data.p_trans = np.array(self.p_trans)
#         data.status = self.status
#         if add_info == 'coefficients':
#             data.c_t = np.array(self.c_t)

#     def run(self, system, steps, potential, data, **kwargs):
#         """
#         Propagates the system for 'steps' timesteps of length 'dt'.
#         Stops at (rho(r_l,t_stop) + rho(r_r,t_stop)) > 1E-8.
#         """
#         from numpy.fft import fft as FT
#         from numpy.fft import ifft as iFT

#         dt = kwargs.get('dt', self.dt)
#         add_info = kwargs.get('additional', None)
#         self.system = system
#         self.prepare(add_info)

#         grid = self.system.x
#         x_Vmax = grid[np.argmax(potential(grid))]
#         rb = kwargs.get('div_surf', x_Vmax)
#         self.rb_idx = np.argmin(abs(grid-rb))

#         self.status = 'Wave did not hit border(s). Scattering process might be incomplete.'
#         m = system.mass

#         self.V = self.system.compute_potential_flat(potential)
#         self.k = self.compute_k(self.V.size)

#         self.counter = 0

#         print('Integrating...')
#         for i in range(self.i_start, steps):
#             self.counter += 1

#             psi = self.propagate_psi(i, dt)
#             e_kin = (np.conjugate(psi).dot(iFT(2.*self.k/m * FT(psi))))
#             e_pot = np.conjugate(psi).dot(self.V*psi)

#             self.store_result(psi, e_kin, e_pot)

#             if add_info == 'coefficients':
#                 self.c_t.append(project_wvfn(psi, self.psi_basis))

#             rho_current = np.real(np.conjugate(psi)*psi)
#             if (rho_current[0]+rho_current[-1] > 2E-3):
#                 self.status = 'Wave hit border(s). Simulation terminated.'
#                 break

#         print('INTEGRATED')
#         print(self.status+'\n')

#         self.p_refl = np.sum(rho_current[:self.rb_idx])
#         self.p_trans = np.sum(rho_current[(self.rb_idx+1):])
#         self.assign_data(data, i, add_info)
