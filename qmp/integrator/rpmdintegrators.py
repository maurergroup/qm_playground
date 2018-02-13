#qmp.integrator.rpmdintegrator
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
from qmp.tools.termcolors import *
import numpy as np


def remove_restart(filename):
    """
    Removes filename from current directory, if existing
    """
    try:
        os.remove(filename)
    except OSError:
        pass


class RPMD_VelocityVerlet(Integrator):
    """
    Velocity verlet integrator for classical dynamics
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
        
        Np = self.basis.npar
        restart = kwargs.get('restart_file', False)
        if not restart:
            ## initialize zero arrays
            self.rb_t = np.zeros((Np,Nb,steps+1,ndim))
            self.rb_t[:,:,0,:] = np.array(self.basis.r_beads)
            self.r_t = np.zeros((Np,steps+1,ndim))

            self.vb_t = np.zeros((Np,Nb,steps+1,ndim))
            self.vb_t[:,:,0,:] = np.array(self.basis.v_beads)
            self.v_t = np.zeros((Np,steps+1,ndim))

            self.E_pot = np.zeros((Np,steps+1))
            self.E_kin = np.zeros((Np,steps+1))
            self.E_tot = np.zeros((Np,steps+1))

            self.e_pot = np.zeros((Np,Nb,steps+1))
            self.e_kin = np.zeros((Np,Nb,steps+1))
            self.e_tot = np.zeros((Np,Nb,steps+1))

            h = np.histogram(self.rb_t[:,0,:],bins=np.arange(self.data.cell[0][0], self.data.cell[0][1], 0.1))
            self.vals = np.zeros((Np,len(h[0])))
            self.rbins = h[1]
            
            self.omega_t = np.zeros((Np,steps+1))
            
            self.p_start = 0
            self.s_start = 0
        else:
            ## try to load arrays from restart file
            try:
                import cPickle as pick
                restart_file = open(restart,'rb')
                current_data = pick.load(restart_file)
                self.rb_t = current_data['rb_t']
                self.r_t = current_data['r_t']
                self.vb_t = current_data['vb_t']
                self.v_t = current_data['v_t']
                self.E_tot = current_data['E_tot']
                self.E_kin = current_data['E_kin']
                self.E_pot = current_data['E_pot']
                self.e_tot = current_data['Eb_tot']
                self.e_kin = current_data['Eb_kin']
                self.e_pot = current_data['Eb_pot']
                self.vals = current_data['prop_vals']
                self.rbins = current_data['bins']
                self.omega_t = current_data['omega_t']
                self.p_start = current_data['i_p']
                self.s_start = current_data['i_s']+1
            except:
                raise ValueError("Could not load restart file '"+str(restart)+"'.")
        
        
    def run(self, steps, dt, **kwargs):
        import cPickle as pick
        ## general information
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        
        ## positions, velocities, energies, etc.
        rb_t = self.rb_t
        vb_t = self.vb_t
        r_t = self.r_t
        v_t = self.v_t
        
        E_pot = self.E_pot
        E_kin = self.E_kin
        E_tot = self.E_tot
        e_pot = self.e_pot
        e_kin = self.e_kin
        e_tot = self.e_tot
        
        vals = self.vals
        rbins = self.rbins
        
        dyn_T = kwargs.get('dyn_T', False)
        omega_t = self.omega_t
        
        print(gray+'Integrating...'+endcolor)
        for i_p in xrange(Np):
            dt_ts = np.zeros(Nb)
            for i_s in xrange(steps):
                ## get omega according to dyn_T
                if not dyn_T:
                    om = self.basis.om[i_p]
                elif dyn_T == 'Rugh':    
                    om = self.basis.get_omega_Rugh(r_t[i_p,i_s], self.pot, v_t[i_p,i_s], m[i_p])
                else:
                    raise ValueError("Scheme for dynamical Temperature '"+dyn_T+"' not known. Use False or 'Rugh'.")
                omega_t[i_p,i_s] = om
                
                ## energy stuff
                e_pot[i_p,:,i_s] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                e_kin[i_p,:,i_s] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_s])
                e_tot[i_p,:,i_s] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
                
                v_t[i_p,i_s] = np.mean(vb_t[i_p,:,i_s],0)
                r_t[i_p,i_s] = np.mean(rb_t[i_p,:,i_s],0)
                if i_s > 10000:
                    vals[i_p] += np.histogram(r_t[i_p,i_s],bins=rbins,density=True)[0]
               
                ## propagation
                F = self.basis.get_forces(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                v1 = vb_t[i_p,:,i_s] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_s+1] = rb_t[i_p,:,i_s]+v1*dt
                F = self.basis.get_forces(rb_t[i_p,:,i_s+1], self.pot, m[i_p], om[i_p])
                vb_t[i_p,:,i_s+1] = v1+F/m[i_p]*dt/2.
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_s+1], dt_ts = self.thermo(vb_t[i_p,:,i_s+1], m[i_p], dt_ts, ndim)
                
                ## write binary restart file
                if np.mod(i_s+1, 1000000) == 0:
                    out = open('rpmd_dyn.rst', 'wb')
                    rpmd_data = {'rb_t':rb_t, 'r_t':r_t, 'vb_t':vb_t, 'v_t':v_t, \
                                 'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E_tot, \
                                 'Eb_kin':e_kin, 'Eb_pot':e_pot, 'Eb_tot':e_tot, \
                                 'bins':rbins, 'prop_vals':vals, 'omega_t':omega_t, \
                                 'i_p':i_p, 'i_s':i_s}
                    pick.dump(rpmd_data,out)
                
            vals[i_p] /= i_s
            
            e_pot[i_p,:,steps] = self.basis.get_potential_energy_beads(rb_t[i_p,:,steps], self.pot, m[i_p], om[i_p])
            e_kin[i_p,:,steps] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,steps])
            e_tot[i_p,:,steps] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
            
            E_pot[i_p] = np.mean(e_pot[i_p],0)
            E_kin[i_p] = np.mean(e_kin[i_p],0)
            E_tot[i_p] = np.mean(e_tot[i_p],0)
            
        vals_tot = np.mean(vals,0)
        print(gray+'INTEGRATED'+endcolor)
        
        
        ## write rb_t, r_t, energies, propabilities to binary output file
        out = open('rpmd_dyn.end', 'wb')
        rpmd_data = {'rb_t':rb_t, 'r_t':r_t, 'vb_t':vb_t, 'v_t':v_t, \
                     'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E_tot, \
                     'Eb_kin':e_kin, 'Eb_pot':e_pot, 'Eb_tot':e_tot, \
                     'bins':rbins, 'prop_vals':vals, 'prop_tot':vals_tot, \
                     'omega_t':omega_t}
        pick.dump(rpmd_data,out)
        
        ## write information to model.data.rpmd
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
        
        self.data.rpmd.prop_vals = vals
        self.data.rpmd.prop_tot = vals_tot
        self.data.rpmd.prop_bins = rbins
        
        self.data.rpmd.omega_t = omega_t
        
        ## remove restart file(s)
        remove_restart('rpmd_dyn.rst')
        
    

class RPMD_equilibrium_properties(Integrator):
    """
    Velocity verlet integrator to obtain equilibrium properties from RPMD Trajectories
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
        
        Np = self.basis.npar
        restart = kwargs.get('restart_file', False)
        if not restart:
            ## initialize zero arrays
            self.rb_t = np.array(self.basis.r_beads)
            self.r_mean = np.zeros(Np)
            self.vb_t = np.array(self.basis.v_beads)
            self.E_pot = np.zeros(Np)
            self.E_kin = np.zeros(Np)
            self.E_tot = np.zeros(Np)
            h = np.histogram(self.rb_t[:,0,:],bins=np.arange(self.data.cell[0][0], self.data.cell[0][1], 0.1))
            self.vals = np.zeros((Np,len(h[0])))
            self.rbins = h[1]
            self.p_start = 0
            self.s_start = 0
        else:
            ## try to load arrays from restart file
            try:
                import cPickle as pick
                restart_file = open(restart,'rb')
                current_data = pick.load(restart_file)
                self.s_start = current_data['i_s']
                self.rb_t = current_data['rb_t']
                self.r_mean = current_data['r_mean']
                self.E_tot = current_data['E_tot']
                self.E_kin = current_data['E_kin']
                self.E_pot = current_data['E_pot']
                self.vals = current_data['prop_dist']
                self.rbins = current_data['bins']
                self.p_start = current_data['i_p']
                self.s_start += 1
            except:
                raise ValueError("Could not load restart file '"+str(restart)+"'.")
        
    
    def run(self, steps, dt, **kwargs):
        import cPickle as pick
        ##general information
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        dyn_T = kwargs.get('dyn_T', False)
        
        ## get arrays for position, velocity, etc.
        rb_t = self.rb_t
        r_mean = self.r_mean
        vb_t = self.vb_t
        E_pot = self.E_pot
        E_kin = self.E_kin
        E_tot = self.E_tot
        vals = self.vals
        rbins = self.rbins
        
        print(gray+'Integrating...'+endcolor)
        for i_p in xrange(self.p_start, Np):     ## loop over beads rewritten in vector-matrix-formalism
            dt_ts = np.zeros(Nb)
            for i_s in xrange(self.s_start, steps):
                ## get omega according to dyn_T
                if not dyn_T:
                    om = self.basis.om[i_p]
                elif dyn_T == 'Rugh':    
                    om = self.basis.get_omega_Rugh(r_t[i_p,i_s], self.pot, v_t[i_p,i_s], m[i_p])
                else:
                    raise ValueError("Scheme for dynamical Temperature '"+dyn_T+"' not known. Use False or 'Rugh'.")
                
                ## energy stuff
                e_pot = self.basis.get_potential_energy_beads(rb_t[i_p], self.pot, m[i_p], om)
                e_kin = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p])
                e_tot = e_kin+e_pot
                E_pot[i_p] += np.mean(e_pot)
                E_kin[i_p] += np.mean(e_kin)
                E_tot[i_p] += np.mean(e_tot)
                
                ## let system equilibrate for a few 100 steps
                if i_s > 1000:
                    vals[i_p] += np.histogram(np.mean(rb_t[i_p],0),bins=rbins,density=True)[0]
                    r_mean[i_p] += np.mean(rb_t[i_p],0)
                
                ## propagation
                F = self.basis.get_forces(rb_t[i_p], self.pot, m[i_p], om)
                v1 = vb_t[i_p] + F/m[i_p]*dt/2.
                rb_t[i_p] = rb_t[i_p]+v1*dt
                F = self.basis.get_forces(rb_t[i_p], self.pot, m[i_p], om)
                vb_t[i_p] = v1+F/m[i_p]*dt/2.
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p], dt_ts = self.thermo(vb_t[i_p], m[i_p], dt_ts, ndim)
                
                ## write binary restart files
                if np.mod(i_s+1, 1000000) == 0:
                    out = open('rpmd_avgs.rst', 'wb')
                    rpmd_data = {'bins':rbins, 'prop_dist':vals, 'r_mean':r_mean, \
                                 'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E_tot,\
                                 'i_p':i_p, 'i_s':i_s}
                    pick.dump(rpmd_data,out)
                
            vals[i_p] /= i_s
            r_mean[i_p] /= i_s
            e_pot = self.basis.get_potential_energy_beads(rb_t[i_p], self.pot, m[i_p], om)
            e_kin = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p])
            e_tot = e_kin+e_pot
            E_pot[i_p] += np.mean(e_pot)
            E_pot[i_p] /= i_s
            E_kin[i_p] += np.mean(e_kin)
            E_kin[i_p] /= i_s
            E_tot[i_p] += np.mean(e_tot)
            E_tot[i_p] /= i_s
            
        vals_tot = np.mean(vals,0)
        r_mean_tot = np.mean(r_mean)
        print(gray+'INTEGRATED'+endcolor)
        
        ## write rb_t, r_t, energies, propabilities to binary output file
        out = open('rpmd_avgs.end', 'wb')
        rpmd_data = {'bins':rbins, 'prop_dist_p':vals, 'prop_dist_tot':vals_tot, \
                     'r_mean_p':r_mean, 'r_mean_tot':r_mean_tot, \
                     'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E_tot}
        pick.dump(rpmd_data,out)
        
        ## write information to model.data.rpmd
        self.data.rpmd.E_kin = E_kin
        self.data.rpmd.E_pot = E_pot
        self.data.rpmd.E_t = E_tot
        self.data.rpmd.prop_vals = vals
        self.data.rpmd.prop_tot = vals_tot
        self.data.rpmd.prop_bins = rbins
        self.data.rpmd.r_mean = r_mean
        self.data.rpmd.r_mean_tot = r_mean_tot
        
        ## remove restart file
        remove_restart('rpmd_avgs.rst')
        
    

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
        import cPickle as pick
        
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
        for i_p in xrange(Np): 
            dt_ts = np.zeros(Nb)
            for i_s in xrange(steps):
                ## get omega according to dyn_T
                if not dyn_T:
                    om = self.basis.om[i_p]
                elif dyn_T == 'Rugh':    
                    om = self.basis.get_omega_Rugh(r_t[i_p,i_s], self.pot, v_t[i_p,i_s], m[i_p])
                else:
                    raise ValueError("Scheme for dynamical Temperature '"+dyn_T+"' not known. Use False or 'Rugh'.")
                omega_t[i_p,i_s] = om
                
                ## energy stuff
                e_pot[i_p,:,i_s] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om)
                e_kin[i_p,:,i_s] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_s])
                
                ## check if particle has reached a border. If so freeze and exit current loop
                v_t[i_p,i_s] = np.mean(vb_t[i_p,:,i_s],0)
                r_t[i_p,i_s] = np.mean(rb_t[i_p,:,i_s],0)
                if (r_t[i_p,i_s] <= self.lower_bound) or \
                   (r_t[i_p,i_s] >= self.upper_bound):
                    r_t[i_p,i_s:] = np.array([r_t[i_p,i_s]]*(steps-i_s+1))
                    rb_t[i_p,:,i_s:] = np.array([[rb_t[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(rb_t.shape[1])])
                    vb_t[i_p,:,i_s:] = np.array([[vb_t[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(vb_t.shape[1])])
                    e_pot[i_p,:,i_s:] = np.array([[e_pot[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(e_pot.shape[1])])
                    e_kin[i_p,:,i_s:] = np.array([[e_kin[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(e_kin.shape[1])])
                    omega_t[i_p,i_s:] = np.array([om,]*(steps-i_s+1)).flatten()
                    self.mobile -= 1
                    break
                
                ## propagation
                F = self.basis.get_forces_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om)
                v1 = vb_t[i_p,:,i_s] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_s+1] = rb_t[i_p,:,i_s]+v1*dt
                F = self.basis.get_forces_beads(rb_t[i_p,:,i_s+1], self.pot, m[i_p], om)
                vb_t[i_p,:,i_s+1] = v1+F/m[i_p]*dt/2.
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_s+1], dt_ts = self.thermo(vb_t[i_p,:,i_s+1], m[i_p], dt_ts, ndim)
                
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
