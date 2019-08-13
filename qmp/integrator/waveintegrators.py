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


class EigenPropagator(Integrator):
    """
    Projects initial wavefunction onto eigenbasis,
    propagates expansion coefficients.
    """
    def prepare_coefficients(self):
        states = self.system.basis.shape[1]
        print('Projecting wavefunction onto basis of '
              + str(states) + ' eigenstates')
        if self.system.basis.shape[0] != states:
            print('** WARNING: This basis is incomplete,'
                  + ' coefficients might contain errors **')

        return project_wvfn(self.system.psi, self.system.basis)

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates psi_0 for 'steps' timesteps of length 'dt'.
        """

        self.system = system
        dt = kwargs.get('dt', self.dt)
        output_freq = kwargs.get('output_freq', 200)
        psi_0 = self.system.psi

        if np.all(psi_0) == 0.:
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

        self.system.c = self.prepare_coefficients()

        prop = np.diag(np.exp(-1j*self.system.E*dt/hbar))  # (states,states)
        self.psi_t = [self.system.basis.dot(self.system.c)]
        self.c_t = [self.system.c]

        output_index = 0

        print('Integrating...')
        for i in range(1, steps+1):
            self.system.c = prop.dot(self.system.c)
            self.system.psi = self.system.basis.dot(self.system.c)
            if (i+1) % output_freq == 0:
                output_index += 1
                self.psi_t.append(self.system.psi.flatten())
                self.c_t.append(self.system.c)

        print('INTEGRATED\n')

        self.psi_t = np.array(self.psi_t)
        self.c_t = np.array(self.c_t)
        E_times_c = self.system.E * self.c_t
        self.E = np.einsum('ik,ik->i', self.c_t.conj(), E_times_c)

        data.psi_t = self.psi_t
        data.rho_t = np.conj(data.psi_t)*data.psi_t
        data.rho_mean = np.mean(data.rho_t, 0)
        data.c_t = np.array(self.c_t)
        data.E_t = np.array(self.E)


class PrimitivePropagator(Integrator):
    """
    Primitive exp(-iHt) propagator for psi in arbitrary
    basis in spatial representation
    """

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        import scipy.linalg as la

        self.system = system
        dt = kwargs.get('dt', self.dt)
        output_freq = kwargs.get('output_freq', 200)
        psi_0 = self.system.psi

        # construct H
        T = self.system.construct_T_matrix()
        V = self.system.construct_V_matrix(potential)
        H = np.array(T+V)

        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != T.shape[1]):
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

        prop = la.expm(-1j*H*dt/hbar)
        self.psi_t = [self.system.psi.flatten()]

        output_index = 0

        print('Integrating...')
        for i in range(steps):

            self.system.psi = np.einsum('ij,...j', prop, self.system.psi)

            if (i+1) % output_freq == 0:
                output_index += 1
                self.psi_t.append(self.system.psi.flatten())

        print('INTEGRATED\n')

        self.psi_t = np.array(self.psi_t)
        H_dot_psi = np.einsum('ij,xj->xi', H, self.psi_t)
        self.E = np.einsum('ik,ik->i', self.psi_t.conj(), H_dot_psi)

        data.psi_t = self.psi_t
        data.rho_t = np.conj(data.psi_t)*data.psi_t
        data.rho_mean = np.mean(data.rho_t, 0)
        data.E_t = self.E


class SOFT_Propagator(Integrator):
    """
    Split operator propagator for psi(x,0)
        Trotter series: exp(iHt) ~= exp(iVt/2)*exp(iTt)*exp(iVt/2)
        => exp(iHt)*psi(x) ~= exp(iTt/2)*exp(iVt)*exp(iTt/2)*psi(x)
        => use spatial representation for exp(iVt) and momentum representation for exp(iTt/2)
        => psi(x,t) = iFT(exp(t*p**2/4m) * FT(exp(iVt) * iFT(exp(t*p**2/4m) * FT(psi(x,0)))))
    """

    def prepare_coefficients(self):
        self.psi_basis = self.system.psi
        states = self.psi_basis.shape[1]
        print('Projecting wavefunction onto basis of '
              + str(states) + ' eigenstates')
        if self.psi_basis.shape[0] != states:
            print('** WARNING: This basis is incomplete, \
                  coefficients might contain errors **')

        return [project_wvfn(self.psi, self.psi_basis)]

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
        return np.exp(-1.0j * self.V * dt)

    def expT(self, dt):
        m = self.system.mass
        return np.exp(-1.0j * (self.k**2) * dt / (2.0 * m))

    def propagate_psi(self, dt):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        # Can use this as an alternative, should give the same result.
        # psi = iFT(self.expT(dt/2)*FT(psi))
        # psi = self.expV(dt)*psi
        # psi = iFT(self.expT(dt/2)*FT(psi))

        self.system.psi = self.expV(dt/2)*self.system.psi
        self.system.psi = iFT(self.expT(dt)*FT(self.system.psi))
        self.system.psi = self.expV(dt/2)*self.system.psi

    def compute_energies(self):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        m = self.system.mass
        conj = np.conj(self.psi_t)
        K_dot_psi = iFT(np.multiply(self.k**2/(2*m), FT(self.psi_t)))
        self.E_kin_t = np.einsum('ik,ik->i', conj, K_dot_psi)
        self.E_pot_t = np.einsum('ik,ik->i', conj, self.V*self.psi_t)

    def store_result(self, add_info):

        self.psi_t.append(self.system.psi.flatten())

        if add_info == 'coefficients':
            self.c_t.append(project_wvfn(self.system.psi, self.psi_basis))

    def assign_data(self, data, i, add_info):
        # TODO RETHINK c_t -- Why? (MS) -- what's happening? (JG)
        if add_info == 'coefficients':
            data.c_t = np.array(self.c_t)

        data.psi_t = self.psi_t
        data.rho_t = np.conjugate(data.psi_t)*data.psi_t
        data.E_kin_t = self.E_kin_t
        data.E_pot_t = self.E_pot_t
        data.E_t = self.E_kin_t + self.E_pot_t
        data.E_mean = np.mean(data.E_t)
        data.E_k_mean = np.mean(data.E_kin_t)
        data.E_p_mean = np.mean(data.E_pot_t)
        data.rho_mean = np.mean(data.rho_t, 0)

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """

        dt = kwargs.get('dt', self.dt)
        output_freq = kwargs.get('output_freq', 200)
        add_info = kwargs.get('additional', None)

        self.system = system
        self.V = self.system.compute_potential_flat(potential)
        self.k = self.compute_k()

        self.psi_t = [self.system.psi.flatten()]

        if (add_info == 'coefficients'):
            self.c_t = self.prepare_coefficients()

        output_index = 0

        print('Integrating...')
        for i in range(steps):

            self.propagate_psi(dt)

            if (i+1) % output_freq == 0:
                output_index += 1
                self.store_result(add_info)

        print('INTEGRATED\n')

        self.psi_t = np.array(self.psi_t)
        self.compute_energies()
        self.assign_data(data, i, add_info)


class SOFT_NonAdiabatic(SOFT_Propagator):

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

    def initialise_start(self, system, potential):
        self.system = system
        self.V = self.system.construct_V_matrix(potential)
        self.k = self.compute_k()

        self.psi_t = [self.system.psi]
        self.propT = self.expT(self.dt/2)
        self.propV = self.expV(self.dt)

    def expV(self, dt):
        try:
            v1 = self.V[0, 0]
            v2 = self.V[1, 1]
            v12 = self.V[0, 1]
            v21 = self.V[1, 0]

            D = (v1 - v2)**2 + 4*v21**2
            diagonal = np.exp(-1.0j * (v1+v2) * dt/2)

            cos = np.einsum('i,jk->jki', np.cos(np.sqrt(D)*dt/2), np.eye(2))

            sin = 1.0j * np.sin(np.sqrt(D)*dt/2)/np.sqrt(D)

            v_matrix = np.array([[v2-v1, -2*v12],
                                 [-2*v21, v1-v2]])

            return diagonal * (cos + sin * v_matrix)

        except IndexError:
            print('Single energy surface')

            exp = np.exp(-1.0j * self.V[0, 0] * dt)
            return np.array([[exp]])

    def expT(self, dt):
        T = self.get_T()
        return np.exp(-1.0j * T * dt)

    def get_T(self):
        m = self.system.mass
        T = self.k**2 / (2.0 * m)
        return T

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

    def propagate_psi(self):
        self.propagate_momentum()
        self.system.psi = np.einsum('ij...,j...->j...',
                                    self.propV, self.system.psi)
        self.propagate_momentum()

    def propagate_momentum(self):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        psi_p = FT(self.system.psi)
        psi_p = self.propT * psi_p
        self.system.psi = iFT(psi_p)

    def store_result(self):
        self.psi_t.append(self.system.psi)

    def compute_energies(self):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        conj = self.psi_t.conj()
        T = self.get_T()

        T_dot_psi = iFT(T * FT(self.psi_t))
        self.E_kin_t = np.real(np.einsum('tjx,tjx->tj', conj, T_dot_psi))

        V_dot_psi = np.einsum('ijx,tjx->tj', self.V, self.psi_t)
        self.E_pot_t = np.real(np.einsum('tjx,tj->tj', conj, V_dot_psi))

    def assign_data(self, data):
        data.psi_t = np.array(self.psi_t)
        data.rho_t = np.conjugate(data.psi_t)*data.psi_t
        data.E_kin_t = self.E_kin_t
        data.E_pot_t = self.E_pot_t
        data.E_t = self.E_kin_t + self.E_pot_t
        # data.E_mean = np.mean(data.E_t)
        # data.E_k_mean = np.mean(data.E_kin_t)
        # data.E_p_mean = np.mean(data.E_pot_t)
        # data.rho_mean = np.mean(data.rho_t, 0)


class SOFT_AverageProperties(SOFT_Propagator):
    """
    SOFT propagator for psi(r,0) to determine expectation values

    output:
    =======
        rho:   average density at r, \sum_{steps} |psi(r,step)|^2/steps
        E_tot: average total energy, \sum_{steps} E_tot/steps (should be constant)
        E_kin: average kinetic energy, \sum_{steps} E_kin/steps
        E_pot: average potential energy, \sum+{steps} E_pot/steps
    """

    def store_result(self, add_info):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        psi = self.system.psi
        m = self.system.mass

        e_kin = np.conj(psi).dot(iFT(self.k**2/(2*m) * FT(psi)))
        e_pot = np.conj(psi).dot(self.V*psi)

        try:
            self.rho += np.conj(psi)*psi
            self.E_kin += e_kin
            self.E_pot += e_pot
        except AttributeError:
            self.rho = np.conj(psi)*psi
            self.e_kin = e_kin
            self.e_pot = e_pot

        if add_info == 'coefficients':
            try:
                self.c_mean += project_wvfn(self.psi, self.psi_basis)
            except AttributeError:
                self.c_mean = project_wvfn(self.psi, self.psi_basis)

    def assign_data(self, data, i, add_info):
        if add_info == 'coefficients':
            data.c_mean = np.array(self.c_mean/i)

        data.E_kin = np.real(self.e_kin / i)
        data.E_pot = np.real(self.e_pot / i)
        data.rho = np.real(self.rho/i)

        data.E = data.E_kin + data.E_pot


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
