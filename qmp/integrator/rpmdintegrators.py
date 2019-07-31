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
"""
rpmd integrators.py
"""

from qmp.integrator.integrator import Integrator
from qmp.integrator.dyn_tools import create_thermostat
import numpy as np
import pickle as pick


def remove_restart():
    """
    Removes filename from current directory, if existing
    """
    try:
        from os import remove
        remove('*.rst')
        del remove
    except OSError:
        pass


class RPMD_VelocityVerlet(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """

    def __init__(self, dt, **kwargs):
        Integrator.__init__(self, dt)

        # create thermostat
        no_ts = {'name': 'no_thermostat', 'cfreq': 0., 'T_set': 0.}
        ts_dict = kwargs.get('thermostat', no_ts)
        ts_name = ts_dict['name']
        ts_cfreq = ts_dict['cfreq']
        ts_Tset = ts_dict['T_set']
        self.thermo = create_thermostat(name=ts_name,
                                        cfreq=ts_cfreq,
                                        T_set=ts_Tset)

        self.restart = kwargs.get('restart_file', False)

    def initialise_zero_arrays(self, steps, N, Nb, ndim, potential):
        self.rb_t = np.zeros((steps, N, Nb, ndim))
        self.vb_t = np.zeros((steps, N, Nb, ndim))
        self.r_t = np.zeros((steps, N, ndim))
        self.v_t = np.zeros((steps, N, ndim))

        self.rb_t[0] = np.array(self.system.r_beads)
        self.vb_t[0] = np.array(self.system.v_beads)
        self.r_t[0] = np.mean(self.system.r_beads, 1)
        self.v_t[0] = np.mean(self.system.v_beads, 1)

        self.e_pot = np.zeros((steps, N, Nb))
        self.e_kin = np.zeros((steps, N, Nb))

        self.e_pot[0] = self.system.compute_bead_potential_energy(potential)
        self.e_kin[0] = self.system.compute_kinetic_energy()

        self.dt_ts = np.zeros(Nb)

    def load_restart(self):
        try:
            restart_file = open(self.restart, 'rb')
            current_data = pick.load(restart_file)
            self.rb_t = current_data['rb_t']
            self.vb_t = current_data['vb_t']
            self.e_kin = current_data['Eb_kin']
            self.e_pot = current_data['Eb_pot']
            self.vals = current_data['prob_vals']
            self.rbins = current_data['bins']
            self.omega_t = current_data['omega_t']
            self.s_start = current_data['i_step']+1
        except FileNotFoundError:
            raise FileNotFoundError("Could not load restart file '"
                                    + str(self.restart)+"'.")

    def prepare(self, steps, potential):
        if not self.restart:

            ndim = self.system.ndim
            N = self.system.n_particles
            Nb = self.system.n_beads
            self.initialise_zero_arrays(steps, N, Nb, ndim, potential)

            bins = [np.arange(potential.cell[0][0], potential.cell[0][1], 0.1)]
            if ndim == 2:
                bins.append(np.arange(potential.cell[1][0],
                                      potential.cell[1][1], 0.1))
            bins = np.array(bins)

            H, self.rbins = np.histogramdd(self.system.r, bins=bins)
            self.vals = np.zeros(np.shape([H, ]*N))

            self.omega_t = np.zeros((steps+1, N))

            self.p_start = 0
            self.s_start = 0
        else:
            self.load_restart()

    def update_omega(self, dyn_T, potential, i_step):
        if dyn_T == 'Rugh':
            self.system.compute_omega_Rugh(potential)
        elif dyn_T is not False:
            raise ValueError("Scheme for dynamical Temperature '"
                             + dyn_T + "' not known. Use False or 'Rugh'.")
        self.omega_t[i_step] = self.system.omega

    def propagate_system(self, potential, dt):
        m = self.system.masses
        F = self.system.compute_bead_force(potential)
        v1 = self.system.v_beads + F / m[:, np.newaxis, np.newaxis] * dt / 2.
        self.system.r_beads += v1 * dt
        F = self.system.compute_bead_force(potential)
        self.system.v_beads = v1 + F / m[:, np.newaxis, np.newaxis] * dt / 2

    def apply_thermostat(self, dt):
        self.dt_ts += dt
        m = self.system.masses
        ndim = self.system.ndim
        self.system.v_beads, self.dt_ts = self.thermo(self.system.v_beads,
                                                      m, self.dt_ts, ndim)

    def store_result(self, e_pot, e_kin, i_step):

        self.rb_t[i_step] = self.system.r_beads
        self.vb_t[i_step] = self.system.v_beads
        self.r_t[i_step] = self.system.r
        self.v_t[i_step] = self.system.v
        self.e_pot[i_step] = e_pot
        self.e_kin[i_step] = e_kin

    def write_restart(self, i_step):
        out = open('rpmd_dyn.rst', 'wb')
        rpmd_data = {'rb_t': self.rb_t, 'vb_t': self.vb_t,
                     'Eb_pot': self.e_pot, 'Eb_kin': self.e_kin,
                     'bins': self.rbins, 'prob_vals': self.vals,
                     'omega_t': self.omega_t, 'i_step': i_step}
        pick.dump(rpmd_data, out)

    def create_histogram(self):
        for i in range(self.system.n_particles):
            self.vals[i] = np.histogramdd(self.r_t[:, i],
                                          bins=self.rbins, normed=True)[0]
        self.vals_tot = np.mean(self.vals, 0)

    def assign_data(self, data, i_step):
        data.rb_t = self.rb_t
        data.vb_t = self.vb_t
        data.r_t = np.mean(self.rb_t, 2)
        data.v_t = np.mean(self.vb_t, 2)

        data.Eb_kin_t = self.e_kin
        data.Eb_pot_t = self.e_pot
        data.Eb_t = self.e_kin + self.e_pot
        data.E_kin_t = np.mean(self.e_kin, 2)
        data.E_pot_t = np.mean(self.e_pot, 2)
        data.E_t = np.mean(data.Eb_t, 2)

        data.prob_vals = self.vals
        data.prob_tot = self.vals_tot
        data.prob_bins = self.rbins
        data.omega_t = self.omega_t

    def write_output(self, data):
        out = open('rpmd_dyn.end', 'wb')
        rpmd_data = {'rb_t': data.rb_t, 'r_t': data.r_t, 'vb_t': data.vb_t,
                     'v_t': data.v_t, 'E_kin': data.E_kin_t,
                     'E_pot': data.E_pot_t,
                     'E_tot': data.E_t, 'Eb_kin': data.Eb_kin_t,
                     'Eb_pot': data.Eb_pot_t,
                     'Eb_tot': data.Eb_t, 'bins': data.prob_bins,
                     'prob_vals': data.prob_vals,
                     'prob_tot': data.prob_tot, 'omega_t': data.omega_t}
        pick.dump(rpmd_data, out)

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)
        dyn_T = kwargs.get('dyn_T', False)

        self.system = system

        self.prepare(steps, potential)

        print('Integrating...')
        for i_step in range(1, steps):
            self.update_omega(dyn_T, potential, i_step)

            self.propagate_system(potential, dt)

            self.apply_thermostat(dt)

            self.system.r = np.mean(self.system.r_beads, 1)
            self.system.v = np.mean(self.system.v_beads, 1)

            e_pot = self.system.compute_bead_potential_energy(potential)
            e_kin = self.system.compute_kinetic_energy()

            self.store_result(e_pot, e_kin, i_step)

            # # write binary restart file
            if np.mod(i_step+1, 1000000) == 0:
                self.write_restart(i_step)

        self.create_histogram()
        print('INTEGRATED')

        self.assign_data(data, i_step)
        self.write_output(data)
        remove_restart()


class RPMD_EquilibriumProperties(RPMD_VelocityVerlet):
    """
    Velocity verlet integrator to obtain equilibrium
    properties from RPMD Trajectories
    """
    # Stuff commented out is probability distribution stuff,
    # not sure how relevant that is for this equilibrium calculation.

    def initialise_zero_arrays(self, steps, N, Nb, ndim, potential):
        self.r_mean = np.zeros((N, ndim))
        self.E_pot = np.zeros(N)
        self.E_kin = np.zeros(N)

        self.dt_ts = np.zeros(Nb)

    def store_result(self, e_pot, e_kin, i_step):
        self.E_pot += np.mean(e_pot, 1)
        self.E_kin += np.mean(e_kin, 1)
        self.r_mean += np.mean(self.system.r_beads, 1)

    def write_restart(self, i_step):
        out = open('rpmd_avgs.rst', 'wb')
        rpmd_data = {'r_mean': self.r_mean,
                     # 'bins': self.rbins, 'prob_vals': self.vals,
                     'E_kin': self.E_kin, 'E_pot': self.E_pot,
                     'i_step': i_step}
        pick.dump(rpmd_data, out)

    def create_histogram(self):
        pass
        # for i in range(self.system.n_particles):
        #     self.vals[i] += np.histogram(self.r_mean[1000:, i],
        #                                  bins=self.rbins, density=True)[0]
        # self.vals_tot = np.mean(self.vals, 0)

    def assign_data(self, data, i_step):
        data.E_kin = self.E_kin / i_step
        data.E_pot = self.E_pot / i_step
        data.E_t = data.E_kin + data.E_pot

        data.r_mean = self.r_mean / i_step
        # data.prob_vals = self.vals
        # data.prob_tot = self.vals_tot
        # data.prob_bins = self.rbins

    def write_output(self, data):
        out = open('rpmd_avgs.end', 'wb')
        rpmd_data = {
                     'E_pot': data.E_pot,
                     'E_tot': data.E_t, 'E_kin': data.E_kin,
                     # 'bins': data.prob_bins,
                     # 'prob_vals': data.prob_vals,
                     # 'prob_tot': data.prob_tot,
                     'r_mean': data.r_mean}
        pick.dump(rpmd_data, out)


class RPMD_Scattering(RPMD_VelocityVerlet):
    """
    Velocity verlet integrator for RPMD scatter trajectories
    """
    def update_omega(self, dyn_T, potential, i_step):
        if dyn_T == 'Rugh':
            self.system.compute_omega_Rugh(potential)
        elif dyn_T is not False:
            raise ValueError("Scheme for dynamical Temperature '"
                             + dyn_T + "' not known. Use False or 'Rugh'.")
        self.omega_t[i_step] = self.system.omega

    def propagate_system(self, potential, dt):
        m = self.system.masses
        F = self.system.compute_bead_force(potential)
        v1 = self.system.v_beads + F / m[:, np.newaxis, np.newaxis] * dt / 2.
        self.system.r_beads[self.active] += v1[self.active] * dt
        F = self.system.compute_bead_force(potential)
        v2 = (v1[self.active] + F[self.active]
              / m[self.active, np.newaxis, np.newaxis] * dt / 2)
        self.system.v_beads[self.active] = v2

    def assign_data(self, data):
        data.rb_t = self.rb_t
        data.vb_t = self.vb_t
        data.r_t = np.mean(self.rb_t, 2)
        data.v_t = np.mean(self.vb_t, 2)

        data.Eb_kin_t = self.e_kin
        data.Eb_pot_t = self.e_pot
        data.Eb_t = self.e_kin + self.e_pot
        data.E_kin_t = np.mean(self.e_kin, 2)
        data.E_pot_t = np.mean(self.e_pot, 2)
        data.E_t = np.mean(data.Eb_t, 2)
        data.E_mean = np.mean(data.E_t, 0)
        data.dErel_max = np.amax(abs(data.E_t - data.E_mean / data.E_mean), 0)

        data.p_refl = self.p_refl
        data.p_trans = self.p_trans
        data.omega_t = self.omega_t

    def write_output(self, data):
        out = open('rpmd_scatter.end', 'wb')
        rpmd_data = {'rb_t': data.rb_t, 'r_t': data.r_t, 'vb_t': data.vb_t,
                     'v_t': data.v_t, 'E_kin': data.E_kin_t,
                     'E_pot': data.E_pot_t,
                     'E_tot': data.E_t, 'Eb_kin': data.Eb_kin_t,
                     'Eb_pot': data.Eb_pot_t,
                     'Eb_tot': data.Eb_t,
                     'E_mean': data.E_mean,
                     'dErel_max': data.dErel_max,
                     'omega_t': data.omega_t}
        pick.dump(rpmd_data, out)

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)
        dyn_T = kwargs.get('dyn_T', False)

        self.system = system

        self.prepare(steps, potential)

        # get boundary freezing vectors
        grid_bounds = potential.cell[0]  # TODO: 2D?
        self.lower_bound = kwargs.get('lower_bound', grid_bounds[0]+3.)
        self.upper_bound = kwargs.get('upper_bound', grid_bounds[1]-3.)

        # get dividing surface, TODO: 2D?
        grid = np.linspace(grid_bounds[0], grid_bounds[1], 2000)
        r_Vmax = grid[np.argmax(potential(grid))]
        data.barrier = np.max(potential(grid))
        data.r_barrier = r_Vmax
        self.rb = kwargs.get('div_surf', r_Vmax)
        self.active = np.full(self.system.n_particles, True, dtype=bool)

        print('Integrating...')
        for i_step in range(steps):
            self.update_omega(dyn_T, potential, i_step)

            self.propagate_system(potential, dt)

            self.apply_thermostat(dt)

            self.system.r = np.mean(self.system.r_beads, 1)
            self.system.v = np.mean(self.system.v_beads, 1)

            # Update active particles
            self.active = np.logical_and(self.system.r >= self.lower_bound,
                                         self.system.r <= self.upper_bound)
            self.active = self.active.flatten()

            e_pot = self.system.compute_bead_potential_energy(potential)
            e_kin = self.system.compute_kinetic_energy()

            self.store_result(e_pot, e_kin, i_step)

        print('INTEGRATED')
        print(str(np.sum(self.active))+' particles did not reach a border\n')

        # count reflected/transmitted particles, in general:
        # count particles on either side of the barrier
        N = self.system.n_particles
        self.p_refl = float(np.count_nonzero(self.system.r < self.rb))/N
        self.p_trans = float(np.count_nonzero(self.system.r > self.rb))/N

        if (self.p_refl+self.p_trans-1.) > 1E-4:
            print('Congratulations, my friend!')
            print('You have just been elected the Most Outstanding Lucky Loser Of the Week (MOLLOW)')
            print('It seems like one or more particles are located exactly at the dividing surface.')

        self.assign_data(data)
        self.write_output(data)
