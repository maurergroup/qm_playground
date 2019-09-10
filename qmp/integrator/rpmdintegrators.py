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
from qmp.integrator.trajintegrators import AbstractVelocityVerlet
from qmp.tools.dyn_tools import create_thermostat
import numpy as np


class RPMD_VelocityVerlet(AbstractVelocityVerlet):
    """
    Velocity verlet integrator for RPMD.
    """
    def __init__(self, dt, **kwargs):
        super().__init__(dt)

        no_ts = {'name': 'no_thermostat', 'cfreq': 0., 'T_set': 0.}
        ts_dict = kwargs.get('thermostat', no_ts)
        ts_name = ts_dict['name']
        ts_cfreq = ts_dict['cfreq']
        ts_Tset = ts_dict['T_set']
        self.thermo = create_thermostat(name=ts_name,
                                        cfreq=ts_cfreq,
                                        T_set=ts_Tset)

    def read_kwargs(self, kwargs):
        super().read_kwargs(kwargs)
        self.dyn_T = kwargs.get('dyn_T', False)

    def initialise_start(self):

        potential = self.potential
        self.rb_t = [self.system.r_beads]
        self.vb_t = [self.system.v_beads]
        self.E_pot = [self.system.compute_bead_potential_energy(potential)]
        self.E_kin = [self.system.compute_kinetic_energy()]

        bins = [np.arange(potential.cell[0][0], potential.cell[0][1], 0.1)]
        if self.system.ndim == 2:
            bins.append(np.arange(potential.cell[1][0],
                                  potential.cell[1][1], 0.1))
        bins = np.array(bins)

        H, self.rbins = np.histogramdd(self.system.r, bins=bins)
        self.vals = np.zeros(np.shape([H, ]*self.system.n_particles))

        self.omega_t = []
        self.dt_ts = np.zeros(self.system.n_beads)

    def integrate(self, steps):
        print('Integrating...')

        self.current_acc = self.system.compute_acceleration(self.potential)
        for i in range(steps):
            self.update_omega()

            self.propagate_system()

            self.apply_thermostat()

            if (i+1) % self.output_freq == 0:
                self.store_result()

        print('INTEGRATED')

    def update_omega(self):
        if self.dyn_T == 'Rugh':
            self.system.compute_omega_Rugh(self.potential)
        elif self.dyn_T is not False:
            raise ValueError(f"Scheme for dynamical Temperature '{self.dyn_T}'"
                             + "' not known. Use False or 'Rugh'.")
        self.omega_t.append(self.system.omega)

    def apply_thermostat(self):
        self.dt_ts += self.dt
        m = self.system.masses
        ndim = self.system.ndim
        self.system.v_beads, self.dt_ts = self.thermo(self.system.v_beads,
                                                      m, self.dt_ts, ndim)

    def store_result(self):

        self.rb_t.append(self.system.r_beads)
        self.vb_t.append(self.system.v_beads)

        e_pot = self.system.compute_bead_potential_energy(self.potential)
        e_kin = self.system.compute_kinetic_energy()
        self.E_pot.append(e_pot)
        self.E_kin.append(e_kin)

    def assign_data(self, data):
        data.rb_t = self.rb_t
        data.vb_t = self.vb_t
        data.r_t = np.mean(self.rb_t, 2)
        data.v_t = np.mean(self.vb_t, 2)
        self.create_histogram(data.r_t)

        data.Eb_kin_t = np.array(self.E_kin)
        data.Eb_pot_t = np.array(self.E_pot)
        data.Eb_t = self.E_kin + self.E_pot
        data.E_kin_t = np.mean(data.Eb_kin_t, 2)
        data.E_pot_t = np.mean(data.Eb_pot_t, 2)
        data.E_t = np.mean(data.Eb_t, 2)

        data.prob_vals = self.vals
        data.prob_tot = self.vals_tot
        data.prob_bins = self.rbins
        data.omega_t = self.omega_t

    def create_histogram(self, positions):
        for i in range(self.system.n_particles):
            self.vals[i] = np.histogramdd(positions[:, i],
                                          bins=self.rbins, normed=True)[0]
        self.vals_tot = np.mean(self.vals, 0)


# TODO: Update this.
# class RPMD_Scattering(RPMD_VelocityVerlet):
#     """
#     Velocity verlet integrator for RPMD scatter trajectories
#     """
#     def update_omega(self, dyn_T, potential, i_step):
#         if dyn_T == 'Rugh':
#             self.system.compute_omega_Rugh(potential)
#         elif dyn_T is not False:
#             raise ValueError("Scheme for dynamical Temperature '"
#                              + dyn_T + "' not known. Use False or 'Rugh'.")
#         self.omega_t[i_step] = self.system.omega

#     def propagate_system(self, potential, dt):
#         m = self.system.masses
#         F = self.system.compute_bead_force(potential)
#         v1 = self.system.v_beads + F / m[:, np.newaxis, np.newaxis] * dt / 2.
#         self.system.r_beads[self.active] += v1[self.active] * dt
#         F = self.system.compute_bead_force(potential)
#         v2 = (v1[self.active] + F[self.active]
#               / m[self.active, np.newaxis, np.newaxis] * dt / 2)
#         self.system.v_beads[self.active] = v2

#     def assign_data(self, data):
#         data.rb_t = self.rb_t
#         data.vb_t = self.vb_t
#         data.r_t = np.mean(self.rb_t, 2)
#         data.v_t = np.mean(self.vb_t, 2)

#         data.Eb_kin_t = self.e_kin
#         data.Eb_pot_t = self.e_pot
#         data.Eb_t = self.e_kin + self.e_pot
#         data.E_kin_t = np.mean(self.e_kin, 2)
#         data.E_pot_t = np.mean(self.e_pot, 2)
#         data.E_t = np.mean(data.Eb_t, 2)
#         data.E_mean = np.mean(data.E_t, 0)
#         data.dErel_max = np.amax(abs(data.E_t - data.E_mean / data.E_mean), 0)

#         data.p_refl = self.p_refl
#         data.p_trans = self.p_trans
#         data.omega_t = self.omega_t

#     def run(self, system, steps, potential, data, **kwargs):

#         dt = kwargs.get('dt', self.dt)
#         dyn_T = kwargs.get('dyn_T', False)

#         self.system = system

#         self.prepare(steps, potential)

#         # get boundary freezing vectors
#         grid_bounds = potential.cell[0]  # TODO: 2D?
#         self.lower_bound = kwargs.get('lower_bound', grid_bounds[0]+3.)
#         self.upper_bound = kwargs.get('upper_bound', grid_bounds[1]-3.)

#         # get dividing surface, TODO: 2D?
#         grid = np.linspace(grid_bounds[0], grid_bounds[1], 2000)
#         r_Vmax = grid[np.argmax(potential(grid))]
#         data.barrier = np.max(potential(grid))
#         data.r_barrier = r_Vmax
#         self.rb = kwargs.get('div_surf', r_Vmax)
#         self.active = np.full(self.system.n_particles, True, dtype=bool)

#         print('Integrating...')
#         for i_step in range(steps):
#             self.update_omega(dyn_T, potential, i_step)

#             self.propagate_system(potential, dt)

#             self.apply_thermostat(dt)

#             self.system.r = np.mean(self.system.r_beads, 1)
#             self.system.v = np.mean(self.system.v_beads, 1)

#             # Update active particles
#             self.active = np.logical_and(self.system.r >= self.lower_bound,
#                                          self.system.r <= self.upper_bound)
#             self.active = self.active.flatten()

#             e_pot = self.system.compute_bead_potential_energy(potential)
#             e_kin = self.system.compute_kinetic_energy()

#             self.store_result(e_pot, e_kin, i_step)

#         print('INTEGRATED')
#         print(str(np.sum(self.active))+' particles did not reach a border\n')

#         # count reflected/transmitted particles, in general:
#         # count particles on either side of the barrier
#         N = self.system.n_particles
#         self.p_refl = float(np.count_nonzero(self.system.r < self.rb))/N
#         self.p_trans = float(np.count_nonzero(self.system.r > self.rb))/N

#         if (self.p_refl+self.p_trans-1.) > 1E-4:
#             print('Congratulations, my friend!')
#             print('You have just been elected the Most Outstanding Lucky Loser Of the Week (MOLLOW)')
#             print('It seems like one or more particles are located exactly at the dividing surface.')

#         self.assign_data(data)
