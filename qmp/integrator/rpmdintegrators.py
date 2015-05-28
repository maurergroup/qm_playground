"""
rpmd integrators.py
"""

from qmp.integrator.integrator import Integrator
from qmp.integrator.dyn_tools import create_thermostat
from qmp.termcolors import *
import numpy as np


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
        
        
    def run(self, steps, dt, **kwargs):
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        om = self.basis.om
        rb_t = np.zeros((Np,Nb,steps+1,ndim))
        rb_t[:,:,0,:] = np.array(self.basis.r_beads)
        vb_t = np.zeros((Np,Nb,steps+1,ndim))
        vb_t[:,:,0,:] = np.array(self.basis.v_beads)
        r_t = np.zeros((Np,steps+1,ndim))
        v_t = np.zeros((Np,steps+1,ndim))
        E_pot = np.zeros((Np,steps+1))
        E_kin = np.zeros((Np,steps+1))
        E_tot = np.zeros((Np,steps+1))
        e_pot = np.zeros((Np,Nb,steps+1))
        e_kin = np.zeros((Np,Nb,steps+1))
        e_tot = np.zeros((Np,Nb,steps+1))
        h = np.histogram(r_t[:,0,:],bins=np.arange(self.data.cell[0][0], self.data.cell[0][1], 0.1))
        vals = np.zeros((Np,len(h[0])))
        rbins = h[1]
        
        print gray+'Integrating...'+endcolor
        for i_p in xrange(Np):     ## loop over beads rewritten in vector-matrix-formalism
            dt_ts = np.zeros(Nb)
            for i_s in xrange(steps):
                ## energy stuff
                e_pot[i_p,:,i_s] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                e_kin[i_p,:,i_s] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_s])
                e_tot[i_p,:,i_s] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
                r = np.mean(rb_t[i_p,:,i_s],0)
                if i_s > 10000:
                    vals[i_p] += np.histogram(r,bins=rbins,density=True)[0]
               
                ## propagation
                F = self.basis.get_forces(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                v1 = vb_t[i_p,:,i_s] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_s+1] = rb_t[i_p,:,i_s]+v1*dt
                F = self.basis.get_forces(rb_t[i_p,:,i_s+1], self.pot, m[i_p], om[i_p])
                vb_t[i_p,:,i_s+1] = v1+F/m[i_p]*dt/2.
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_s+1], dt_ts = self.thermo(vb_t[i_p,:,i_s+1], m[i_p], dt_ts, ndim)
                
            e_pot[i_p,:,steps] = self.basis.get_potential_energy_beads(rb_t[i_p,:,steps], self.pot, m[i_p], om[i_p])
            e_kin[i_p,:,steps] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,steps])
            e_tot[i_p,:,steps] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
            r_t[i_p] = np.mean(rb_t[i_p],0)            
            v_t[i_p] = np.mean(vb_t[i_p],0)
            E_pot[i_p] = np.mean(e_pot[i_p],0)
            E_kin[i_p] = np.mean(e_kin[i_p],0)
            E_tot[i_p] = np.mean(e_tot[i_p],0)
            
        vals_tot = np.mean(vals,0)
        print gray+'INTEGRATED'+endcolor
        
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
        
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        om = self.basis.om
        rb_t = self.rb_t
        r_mean = self.r_mean
        vb_t = self.vb_t
        E_pot = self.E_pot
        E_kin = self.E_kin
        E_tot = self.E_tot
        vals = self.vals
        rbins = self.rbins
        
        print gray+'Integrating...'+endcolor
        for i_p in xrange(self.p_start, Np):     ## loop over beads rewritten in vector-matrix-formalism
            dt_ts = np.zeros(Nb)
            for i_s in xrange(self.s_start, steps):
                ## energy stuff
                e_pot = self.basis.get_potential_energy_beads(rb_t[i_p], self.pot, m[i_p], om[i_p])
                e_kin = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p])
                e_tot = e_kin+e_pot
                E_pot[i_p] += np.mean(e_pot)
                E_kin[i_p] += np.mean(e_kin)
                E_tot[i_p] += np.mean(e_tot)
                
                if i_s > 100:
                    vals[i_p] += np.histogram(np.mean(rb_t[i_p],0),bins=rbins,density=True)[0]
                    r_mean[i_p] += np.mean(rb_t[i_p],0)
                
                ## propagation
                F = self.basis.get_forces(rb_t[i_p], self.pot, m[i_p], om[i_p])
                v1 = vb_t[i_p] + F/m[i_p]*dt/2.
                rb_t[i_p] = rb_t[i_p]+v1*dt
                F = self.basis.get_forces(rb_t[i_p], self.pot, m[i_p], om[i_p])
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
            e_pot = self.basis.get_potential_energy_beads(rb_t[i_p], self.pot, m[i_p], om[i_p])
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
        print gray+'INTEGRATED'+endcolor
        
        ## write rb_t, r_t, energies, propabilities to binary output file
        out = open('rpmd_avgs.end', 'wb')
        rpmd_data = {'bins':rbins, 'prop_dist_p':vals, 'prop_dist_tot':vals_tot, \
                     'r_mean_p':r_mean, 'r_mean_tot':r_mean_tot, \
                     'E_kin':E_kin/i_s, 'E_pot':E_pot/i_s, 'E_tot':E_tot/i_s}
        pick.dump(rpmd_data,out)
        
        self.data.rpmd.E_kin = E_kin
        self.data.rpmd.E_pot = E_pot
        self.data.rpmd.E_t = E_tot
        self.data.rpmd.prop_vals = vals
        self.data.rpmd.prop_tot = vals_tot
        self.data.rpmd.prop_bins = rbins
        self.data.rpmd.r_mean = r_mean
        self.data.rpmd.r_mean_tot = r_mean_tot
        



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
        
        ## get dividing surface TODO: 2D?
        grid = np.linspace(grid_bounds[0], grid_bounds[1], 2000)
        r_Vmax = grid[np.argmax(self.pot(grid))]
        self.data.rpmd.barrier = np.max(self.pot(grid))
        self.data.rpmd.r_barrier = r_Vmax
        self.rb = kwargs.get('div_surf', r_Vmax)
        
    def run(self, steps, dt, **kwargs):
        import cPickle as pick
        
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m
        om = self.basis.om
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
        unfreezed = Np
        
        print gray+'Integrating...'+endcolor
        for i_p in xrange(Np):     ## loop over beads rewritten in vector-matrix-formalism
            dt_ts = np.zeros(Nb)
            for i_s in xrange(steps):
                ## energy stuff
                e_pot[i_p,:,i_s] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                e_kin[i_p,:,i_s] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_s])
                
                ## check if particle has reached a border. If so freeze particle
                r_t[i_p,i_s] = np.mean(rb_t[i_p,:,i_s],0)
                if (r_t[i_p,i_s] <= self.lower_bound) or \
                   (r_t[i_p,i_s] >= self.upper_bound):
                    r_t[i_p,i_s:] = np.array([r_t[i_p,i_s]]*(steps-i_s+1))
                    rb_t[i_p,:,i_s:] = np.array([[rb_t[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(rb_t.shape[1])])
                    vb_t[i_p,:,i_s:] = np.array([[vb_t[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(vb_t.shape[1])])
                    e_pot[i_p,:,i_s:] = np.array([[e_pot[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(e_pot.shape[1])])
                    e_kin[i_p,:,i_s:] = np.array([[e_kin[i_p,b,i_s]]*(steps-i_s+1) for b in xrange(e_kin.shape[1])])
                    unfreezed -= 1
                    break
                
                ## propagation
                F = self.basis.get_forces(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                v1 = vb_t[i_p,:,i_s] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_s+1] = rb_t[i_p,:,i_s]+v1*dt
                F = self.basis.get_forces(rb_t[i_p,:,i_s+1], self.pot, m[i_p], om[i_p])
                vb_t[i_p,:,i_s+1] = v1+F/m[i_p]*dt/2.
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_s+1], dt_ts = self.thermo(vb_t[i_p,:,i_s+1], m[i_p], dt_ts, ndim)
                
            e_pot[i_p,:,steps] = self.basis.get_potential_energy_beads(rb_t[i_p,:,steps], self.pot, m[i_p], om[i_p])
            e_kin[i_p,:,steps] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,steps])
            e_tot = e_kin+e_pot
            E_pot[i_p] = np.mean(e_pot[i_p],0)
            E_kin[i_p] = np.mean(e_kin[i_p],0)
            E_tot[i_p] = np.mean(e_tot[i_p],0)
            E_mean[i_p] = np.mean(E_tot[i_p])
            dErel_max[i_p] = max(  abs( (E_tot[i_p]-E_mean[i_p]) / E_mean[i_p] )  )
            v_t[i_p] = np.mean(vb_t[i_p],0)
            
        print gray+'INTEGRATED'
        print str(unfreezed)+' particles did not reach a border\n'+endcolor
        
        

        ## count reflected/transmitted particles, in general:
        ## count particles that have and those that have not crossed the barrier
        p_refl = float(np.count_nonzero(r_t[:,-1]<self.rb))/Np
        p_trans = float(np.count_nonzero(r_t[:,-1]>self.rb))/Np
        
        if (p_refl+p_trans-1.) > 1E-3:
            print red+'Congratulations, my friend!'
            print 'You have just been elected the Most Outstanding Lucky Loser Of the Week (MOLLOW)'
            print 'It seems like one or more particles are located exactly at the dividing surface.'+endcolor
            
        
        ## write rb_t, r_t, energies, propabilities to binary output file
        out = open('rpmd_scatter.end', 'wb')
        rpmd_data = {'rb_t':rb_t, 'r_t':r_t, 'p_refl':p_refl, 'p_trans':p_trans, 'E_kin':E_kin, \
                     'E_pot':E_pot, 'E_tot':E_tot}
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
        
    

#--EOF--#