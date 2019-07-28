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


def remove_restart():
    """
    Removes filename from current directory, if existing
    """
    filename = '*.rst'
    try:
        from os import remove
        remove(filename)
        del remove
    except OSError:
        pass


class EigenPropagator(Integrator):
    """
    Projects initial wavefunction onto eigenbasis,
    propagates expansion coefficients.
    """

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates psi_0 for 'steps' timesteps of length 'dt'.
        """

        self.system = system
        dt = kwargs.get('dt', self.dt)
        psi_0 = self.system.psi
        psi_basis = self.system.basis
        self.system.c = np.zeros_like(self.system.E)

        if steps == 0:
            self.system.c[0] = 1
            c = self.system.c
        elif not (np.any(self.system.c) != 0.) and not (np.any(psi_0) != 0.):
            raise ValueError('Integrator needs either expansion coefficients '
                             + 'or initial wave function to propagate system!')
        elif not (np.any(psi_0) != 0.):
            c = self.system.c
            norm = np.sqrt(np.conjugate(c).dot(c))
            c /= norm
        elif (len(psi_0.flatten()) != psi_basis.shape[0]):
            raise ValueError('Initial wave function needs to be defined on '
                             + 'same grid as system was solved on!')
        else:
            states = psi_basis.shape[1]
            print('Projecting wavefunction onto basis of '
                  + str(states) + ' eigenstates')
            if psi_basis.shape[0] != states:
                print('**WARNING: This basis is incomplete, coefficients and'
                      + ' wavefunction might contain errors**')
            c = np.array([project_wvfn(psi_0, psi_basis)])

        prop = np.diag(np.exp(-1j*self.system.E*dt/hbar))  # (states,states)
        psi = [psi_basis.dot(c[0])]  # (x,1)
        E = [np.dot(np.conjugate(c[0]), (c[0]*self.system.E))]

        self.counter = 0
        print('Integrating...')
        for i in range(1, steps+1):
            self.counter += 1
            c = np.append(c, np.dot(prop, c[i-1])).reshape(i+1, states)
            psi = np.append(psi, np.dot(psi_basis, c[i])
                            ).reshape(i+1, psi_basis.shape[0])
            E.append(np.dot(np.conjugate(c[i]), (c[i]*self.system.E)))

        print('INTEGRATED\n')

        data.psi_t = np.array(psi)
        data.c_t = np.array(c)
        data.E_t = np.array(E)


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
        psi_0 = self.system.psi

        # construct H
        T = self.system.construct_T_matrix()
        V = self.system.construct_V_matrix(potential)

        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != T.shape[1]):
            raise ValueError('Please provide initial wave function'
                             + ' on appropriate grid')

        H = np.array(T+V)
        prop = la.expm(-1j*H*dt/hbar)  # (x,x)
        psi = np.array([psi_0.flatten()])  # (1,x)
        E = [np.dot(np.conjugate(psi_0.flatten()), np.dot(H, psi_0.flatten()))]

        self.counter = 0
        print('Integrating...')
        for i in range(steps):
            self.counter += 1
            psi = np.append(psi, np.dot(prop, psi[i]))
            psi = np.reshape(psi, (i+2, T.shape[0]))
            E.append(np.dot(psi[i+1].conjugate(), np.dot(H, psi[i+1])))

        print('INTEGRATED\n')

        data.psi_t = np.array(psi)
        data.E_t = np.array(E)


class SOFT_Propagator(Integrator):
    """
    Split operator propagator for psi(x,0)
        Trotter series: exp(iHt) ~= exp(iVt/2)*exp(iTt)*exp(iVt/2)
        => exp(iHt)*psi(x) ~= exp(iTt/2)*exp(iVt)*exp(iTt/2)*psi(x)
        => use spatial representation for exp(iVt) and momentum representation for exp(iTt/2)
        => psi(x,t) = iFT(exp(t*p**2/4m) * FT(exp(iVt) * iFT(exp(t*p**2/4m) * FT(psi(x,0)))))
    """

    def load_restart(self):
        import pickle as pick
        try:
            restart_file = open(self.system.psi, 'rb')
            current_data = pick.load(restart_file)
            self.psi = current_data['psi']
            self.rho = current_data['rho']
            self.E = current_data['E_tot']
            self.E_kin = current_data['E_kin']
            self.E_pot = current_data['E_pot']
            self.i_start = current_data['i']+1
        except FileNotFoundError:
            raise ValueError('Input does not refer to a wave function nor \
                             to a restart file.')

    def prepare_coefficients(self):
        self.psi_basis = self.system.psi
        states = self.psi_basis.shape[1]
        print('Projecting wavefunction onto basis of '
              + str(states) + ' eigenstates')
        if self.psi_basis.shape[0] != states:
            print('** WARNING: This basis is incomplete, \
                  coefficients might contain errors **')

        return [project_wvfn(self.psi, self.psi_basis)]

    def prepare(self, add_info):
        if type(self.system.psi) == str:
            self.load_restart(self.system)
        else:
            self.psi = [self.system.psi.flatten()]
            self.rho = np.conjugate(self.psi)*self.psi
            self.E, self.E_kin, self.E_pot = [], [], []
            self.i_start = 0

        if (add_info == 'coefficients'):
            self.c_t = self.prepare_coefficients()

    def compute_p(self, N):
        from numpy.fft import fftfreq as FTp

        dx = self.system.dx
        nx = self.system.N
        ndim = self.system.ndim

        if ndim == 1:
            p = np.pi*FTp(N, dx)
            p = p*p
        elif ndim == 2:
            p = FTp(nx, dx).conj()*FTp(nx, dx)
            p = np.pi*np.pi*(np.kron(np.ones(nx), p)
                             + np.kron(p, np.ones(nx)))
        else:
            raise NotImplementedError('Only 1D and 2D systems implemented')

        return p

    def compute_operators(self, potential, dt):
        m = self.system.mass
        self.V = self.system.compute_potential_flat(potential)
        N = self.V.size

        self.expV = np.exp(-1j*self.V*dt / hbar)

        self.p = self.compute_p(N)
        self.expT = np.exp(-1j*(dt/hbar)*self.p/m)

    def propagate_psi(self, i):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        psi1 = iFT(self.expT*FT(self.psi[i]))
        psi2 = FT(self.expV*psi1)
        psi3 = iFT(self.expT*psi2)
        return psi3

    def store_result(self, psi, e_kin, e_pot):

        self.rho += np.conjugate(psi)*psi
        self.psi.append(psi)
        self.E_kin.append(e_kin)
        self.E_pot.append(e_pot)
        self.E.append(e_kin+e_pot)

    def write_restart(self, i):
        import pickle as pick
        out = open('wave_dyn.rst', 'wb')
        wave_data = {'psi': self.psi, 'rho': self.rho,
                     'E_kin': self.E_kin,
                     'E_pot': self.E_pot,
                     'E_tot': self.E, 'i': i}
        pick.dump(wave_data, out)

    def assign_data(self, data, i, add_info):
        # TODO RETHINK c_t -- Why? (MS) -- what's happening? (JG)
        if add_info == 'coefficients':
            data.c_t = np.array(self.c_t)

        data.psi_t = np.array(self.psi)
        data.E_t = np.array(self.E)
        data.E_kin_t = np.array(self.E_kin)
        data.E_pot_t = np.array(self.E_pot)
        data.E_mean = np.sum(self.E)/i
        data.E_k_mean = np.sum(self.E_kin)/i
        data.E_p_mean = np.sum(self.E_pot)/i
        data.rho_mean = self.rho/i

    def write_output(self, data):
        import pickle as pick
        out = open('wave_dyn.end', 'wb')
        wave_data = {'psi': data.psi_t, 'rho': data.rho_mean,
                     'E_kin': data.E_kin_t,
                     'E_pot': data.E_pot_t, 'E_tot': data.E_t}
        pick.dump(wave_data, out)

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        dt = kwargs.get('dt', self.dt)
        add_info = kwargs.get('additional', None)
        self.system = system
        self.prepare(add_info)

        m = system.mass

        self.compute_operators(potential, dt)

        self.counter = 0

        print('Integrating...')
        for i in range(self.i_start, steps):
            self.counter += 1

            psi = self.propagate_psi(i)
            e_kin = (np.conjugate(psi).dot(iFT(2.*self.p/m * FT(psi))))
            e_pot = np.conjugate(psi).dot(self.V*psi)

            self.store_result(psi, e_kin, e_pot)

            if add_info == 'coefficients':
                self.c_t.append(project_wvfn(psi, self.psi_basis))

            if (np.mod(i+1, 1000000) == 0):
                self.write_restart(i)

        print('INTEGRATED\n')

        self.assign_data(data, i, add_info)
        self.write_output(data)
        remove_restart()


class SOFT_AverageProperties(SOFT_Propagator):
    """
    SOFT propagator for psi(r,0) to determine expectation values from long simulations

    output:
    =======
        rho:    average density at r, \sum_{steps} |psi(r,step)|^2/steps
        E_tot:  average total energy, \sum_{steps} E_tot/steps (should be constant)
        E_kin:  average kinetic energy, \sum_{steps} E_kin/steps
        E_pot:  average potential energy, \sum+{steps} E_pot/steps
    """
    def load_restart(self):
        import pickle as pick
        try:
            restart_file = open(self.system.psi, 'rb')
            current_data = pick.load(restart_file)
            self.i_start = current_data['i']+1
            self.psi = current_data['psi']
            self.rho = current_data['rho']*self.i_start
            self.E = current_data['E_tot']*self.i_start
            self.E_kin = current_data['E_kin']*self.i_start
            self.E_pot = current_data['E_pot']*self.i_start
        except FileNotFoundError:
            raise ValueError('Input does not refer to a wave function nor \
                             to a restart file.')

    def prepare(self, add_info):
        if type(self.system.psi) == str:
            self.load_restart(self.system)
        else:
            self.psi = np.array(self.system.psi.flatten())
            self.rho = np.conjugate(self.psi)*self.psi
            self.E, self.E_kin, self.E_pot = 0, 0, 0
            self.i_start = 0

        if (add_info == 'coefficients'):
            self.c_mean = self.prepare_coefficients()[0]

    def propagate_psi(self, i):
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        psi1 = iFT(self.expT*FT(self.psi))
        psi2 = FT(self.expV*psi1)
        psi3 = iFT(self.expT*psi2)
        return psi3

    def store_result(self, psi, e_kin, e_pot):
        self.rho += np.conjugate(psi)*psi
        self.psi = psi
        self.E_kin += e_kin
        self.E_pot += e_pot
        self.E += e_kin + e_pot

    def write_restart(self, i):
        import pickle as pick
        out = open('wave_avgs.rst', 'wb')
        wave_data = {'psi': self.psi,
                     'rho': self.rho/i,
                     'E_kin': self.E_kin/i,
                     'E_pot': self.E_pot/i,
                     'E_tot': self.E/i,
                     'i': i}
        pick.dump(wave_data, out)

    def assign_data(self, data, i, add_info):
        if add_info == 'coefficients':
            data.c_mean = np.array(self.c_mean/i)

        data.E_tot = self.E / i
        data.E_kin = self.E_kin / i
        data.E_pot = self.E_pot / i
        data.rho = self.rho/i

    def write_output(self, data):
        import pickle as pick
        out = open('wave_avgs.end', 'wb')
        wave_data = {'rho': data.rho, 'E_kin': data.E_kin,
                     'E_pot': data.E_pot, 'E_tot': data.E_tot}
        pick.dump(wave_data, out)


class SOFT_Scattering(SOFT_Propagator):
    """
    SOFT propagator for scattering processes
                     _               ..
       :            / \ ->          ;  ;                        :
       :           /   \ ->         ;  ;                        :
       :          /     \ ->        ;  ;                        :
       :_________/       \__________;  ;________________________:
      r_l                            rb                        r_r
    (border)       (wave)         (barrier)                  (border)

    Stops if reflected or transmitted part of wave package hits
    border at r_l or r_r, respectively (=> t_stop).

    input:
    ======
        wave:     defined by psi_0 (incl. momentum!)
        barrier:  defined by potential class
        div_surf: dividing surface (1D: rb), given as keyword argument

    output:
    =======
        psi_t:    final wave package at t_stop
        p_refl:   integrated reflected part of wave, \int rho(r,t_stop) for r<rb
        p_trans:  integrated transmitted part of wave, \int rho(r,t_stop) for r>rb
        energy:   E_kin, E_pot, E_tot as functions of simulation time
        status:   information whether scattering process should be complete
    """

    # get borders, r_l and r_b, and dividing surface (1D: rb)
    # TODO: 2D dividing surface?

    def store_result(self, psi, e_kin, e_pot):

        self.rho += np.conjugate(psi)*psi
        self.psi.append(psi)
        self.E_kin.append(e_kin)
        self.E_pot.append(e_pot)
        self.E.append(e_kin+e_pot)
        self.E_mean = np.mean(self.E)

    def assign_data(self, data, i, add_info):
        data.psi_t = np.array(self.psi)
        data.E_t = np.array(self.E)
        data.E_mean = np.array(self.E_mean)
        data.dErel_max = np.array(
                abs(max(abs(self.E-self.E_mean))/self.E_mean))
        data.E_kin_t = np.array(self.E_kin)
        data.E_pot_t = np.array(self.E_pot)
        data.p_refl = np.array(self.p_refl)
        data.p_trans = np.array(self.p_trans)
        data.status = self.status
        if add_info == 'coefficients':
            data.c_t = np.array(self.c_t)

    def write_output(self, data, rho_current):
        import pickle as pick
        out = open('wave_scatter.end', 'wb')
        wave_data = {'psi_t': data.psi_t, 'p_refl': data.p_refl,
                     'p_trans': data.p_trans, 'E_kin': data.E_kin_t,
                     'E_pot': data.E_pot_t, 'E_tot': data.E_t,
                     'rho_end': rho_current}
        pick.dump(wave_data, out)

    def run(self, system, steps, potential, data, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        Stops at (rho(r_l,t_stop) + rho(r_r,t_stop)) > 1E-8.
        """
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT

        dt = kwargs.get('dt', self.dt)
        add_info = kwargs.get('additional', None)
        self.system = system
        self.prepare(add_info)

        grid = self.system.x
        x_Vmax = grid[np.argmax(potential(grid))]
        rb = kwargs.get('div_surf', x_Vmax)
        self.rb_idx = np.argmin(abs(grid-rb))

        self.status = 'Wave did not hit border(s). Scattering process might be incomplete.'
        m = system.mass

        self.compute_operators(potential, dt)

        self.counter = 0

        print('Integrating...')
        for i in range(self.i_start, steps):
            self.counter += 1

            psi = self.propagate_psi(i)
            e_kin = (np.conjugate(psi).dot(iFT(2.*self.p/m * FT(psi))))
            e_pot = np.conjugate(psi).dot(self.V*psi)

            self.store_result(psi, e_kin, e_pot)

            if add_info == 'coefficients':
                self.c_t.append(project_wvfn(psi, self.psi_basis))

            rho_current = np.real(np.conjugate(psi)*psi)
            if (rho_current[0]+rho_current[-1] > 2E-3):
                self.status = 'Wave hit border(s). Simulation terminated.'
                break

            if (np.mod(i+1, 1000000) == 0):
                self.write_restart(i)

        print('INTEGRATED')
        print(self.status+'\n')

        self.p_refl = np.sum(rho_current[:self.rb_idx])
        self.p_trans = np.sum(rho_current[(self.rb_idx+1):])
        self.assign_data(data, i, add_info)

        # write psi, rho, energies to binary output file
        if (self.status == 'Wave hit border(s). Simulation terminated.'):
            self.write_output(data, rho_current)

        remove_restart()
