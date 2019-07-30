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

            bins = np.arange(potential.cell[0][0], potential.cell[0][1], 0.1)
            H, self.rbins = np.histogram(self.system.r, bins=bins)
            self.vals = np.zeros((N, len(H)))

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


class RPMD_scattering(Integrator):
    """
    Velocity verlet integrator for RPMD scatter trajectories
    """

    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)
        self.basis = self.data.rpmd.basis

        ## create thermostat
        no_ts = {'name':'no_thermostat', 'cfreq':0., 'T_set':0.}
        ts_dict = kwargs.get('thermostat', no_ts)
        ts_name = ts_dict['name']
        ts_cfreq = ts_dict['cfreq']
        ts_Tset = ts_dict['T_set']
        self.thermo = create_thermostat(name=ts_name, cfreq=ts_cfreq, T_set=ts_Tset)

        ## get boundary freezing vectors
        grid_bounds = self.data.cell[0]    #TODO: 2D?
        self.lower_bound = kwargs.get('lower_bound', grid_bounds[0]+3.)
        self.upper_bound = kwargs.get('upper_bound', grid_bounds[1]-3.)

        ## get dividing surface, TODO: 2D?
        grid = np.linspace(grid_bounds[0], grid_bounds[1], 2000)
        r_Vmax = grid[np.argmax(self.pot(grid))]
        self.data.rpmd.barrier = np.max(self.pot(grid))
        self.data.rpmd.r_barrier = r_Vmax
        self.rb = kwargs.get('div_surf', r_Vmax)
        self.mobile = self.basis.npar

    def run(self, steps, dt, **kwargs):
        import pickle as pick

        ## get general information
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        dyn_T = kwargs.get('dyn_T', False)

        ## initialize arrays for dynamical properties
        rb_t = np.zeros((Np,Nb,steps+1,ndim))
        rb_t[:,:,0,:] = np.array(self.basis.r_beads)
        vb_t = np.zeros((Np,Nb,steps+1,ndim))
        vb_t[:,:,0,:] = np.array(self.basis.v_beads)
        r_t = np.zeros((Np,steps+1,ndim))
        v_t = np.zeros((Np,steps+1,ndim))
        E_pot = np.zeros((Np,steps+1))
        E_kin = np.zeros((Np,steps+1))
        E_tot = np.zeros((Np,steps+1))
        E_mean = np.zeros(Np)
        dErel_max = np.zeros(Np)
        e_pot = np.zeros((Np,Nb,steps+1))
        e_kin = np.zeros((Np,Nb,steps+1))
        e_tot = np.zeros((Np,Nb,steps+1))
        omega_t = np.zeros((Np,steps+1))

        print(gray+'Integrating...'+endcolor)
        for i_p in range(Np):
            dt_ts = np.zeros(Nb)
            for i_step in range(steps):
                ## get omega according to dyn_T
                if not dyn_T:
                    om = self.basis.om[i_p]
                elif dyn_T == 'Rugh':
                    om = self.basis.get_omega_Rugh(r_t[i_p,i_step], self.pot, v_t[i_p,i_step], m[i_p])
                else:
                    raise ValueError("Scheme for dynamical Temperature '"+dyn_T+"' not known. Use False or 'Rugh'.")
                omega_t[i_p,i_step] = om

                ## energy stuff
                e_pot[i_p,:,i_step] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_step], self.pot, m[i_p], om)
                e_kin[i_p,:,i_step] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_step])

                ## check if particle has reached a border. If so freeze and exit current loop
                v_t[i_p,i_step] = np.mean(vb_t[i_p,:,i_step],0)
                r_t[i_p,i_step] = np.mean(rb_t[i_p,:,i_step],0)
                if (r_t[i_p,i_step] <= self.lower_bound) or \
                   (r_t[i_p,i_step] >= self.upper_bound):
                    r_t[i_p,i_step:] = np.array([r_t[i_p,i_step]]*(steps-i_step+1))
                    rb_t[i_p,:,i_step:] = np.array([[rb_t[i_p,b,i_step]]*(steps-i_step+1) for b in range(rb_t.shape[1])])
                    vb_t[i_p,:,i_step:] = np.array([[vb_t[i_p,b,i_step]]*(steps-i_step+1) for b in range(vb_t.shape[1])])
                    e_pot[i_p,:,i_step:] = np.array([[e_pot[i_p,b,i_step]]*(steps-i_step+1) for b in range(e_pot.shape[1])])
                    e_kin[i_p,:,i_step:] = np.array([[e_kin[i_p,b,i_step]]*(steps-i_step+1) for b in range(e_kin.shape[1])])
                    omega_t[i_p,i_step:] = np.array([om,]*(steps-i_step+1)).flatten()
                    self.mobile -= 1
                    break

                ## propagation
                F = self.basis.get_forces_beads(rb_t[i_p,:,i_step], self.pot, m[i_p], om)
                v1 = vb_t[i_p,:,i_step] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_step+1] = rb_t[i_p,:,i_step]+v1*dt
                F = self.basis.get_forces_beads(rb_t[i_p,:,i_step+1], self.pot, m[i_p], om)
                vb_t[i_p,:,i_step+1] = v1+F/m[i_p]*dt/2.

                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_step+1], dt_ts = self.thermo(vb_t[i_p,:,i_step+1], m[i_p], dt_ts, ndim)

            e_pot[i_p,:,steps] = self.basis.get_potential_energy_beads(rb_t[i_p,:,steps], self.pot, m[i_p], om)
            e_kin[i_p,:,steps] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,steps])
            e_tot = e_kin+e_pot
            E_pot[i_p] = np.mean(e_pot[i_p],0)
            E_kin[i_p] = np.mean(e_kin[i_p],0)
            E_tot[i_p] = np.mean(e_tot[i_p],0)
            E_mean[i_p] = np.mean(E_tot[i_p])
            dErel_max[i_p] = max(  abs( (E_tot[i_p]-E_mean[i_p]) / E_mean[i_p] )  )

        print(gray+'INTEGRATED')
        print(str(self.mobile)+' particles did not reach a border\n'+endcolor)

        ## count reflected/transmitted particles, in general:
        ## count particles that have and those that have not crossed the barrier
        p_refl = float(np.count_nonzero(r_t[:,-1]<self.rb))/Np
        p_trans = float(np.count_nonzero(r_t[:,-1]>self.rb))/Np

        if (p_refl+p_trans-1.) > 1E-4:
            print(red+'Congratulations, my friend!')
            print('You have just been elected the Most Outstanding Lucky Loser Of the Week (MOLLOW)')
            print('It seems like one or more particles are located exactly at the dividing surface.'+endcolor)

        ## write rb_t, r_t, energies, propabilities to binary output file
        out = open('rpmd_scatter.end', 'wb')
        rpmd_data = {'rb_t':rb_t, 'r_t':r_t, 'vb_t':vb_t, 'v_t':v_t,\
                     'p_refl':p_refl, 'p_trans':p_trans, \
                     'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E_tot, \
                     'Eb_kin':e_kin, 'Eb_pot':e_pot, 'Eb_tot':e_tot, \
                     'E_mean':E_mean, 'dErel_max':dErel_max}
        pick.dump(rpmd_data,out)

        self.data.rpmd.rb_t = rb_t
        self.data.rpmd.vb_t = vb_t
        self.data.rpmd.r_t = r_t
        self.data.rpmd.v_t = v_t

        self.data.rpmd.Eb_kin_t = e_kin
        self.data.rpmd.Eb_pot_t = e_pot
        self.data.rpmd.Eb_t = e_tot
        self.data.rpmd.E_kin_t = E_kin
        self.data.rpmd.E_pot_t = E_pot
        self.data.rpmd.E_t = E_tot
        self.data.rpmd.E_mean = E_mean
        self.data.rpmd.dErel_max = dErel_max

        self.data.rpmd.p_refl = p_refl
        self.data.rpmd.p_trans = p_trans
        self.data.rpmd.omega_t = omega_t



#--EOF--#
