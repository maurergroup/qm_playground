"""
rpmd integrators.py
"""

from qmp.integrator.integrator import Integrator
from qmp.integrator.dyn_tools import create_thermostat
import numpy as np


class RPMD_VelocityVerlet(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """
    
    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)
        self.basis = self.data.rpmd.basis

        
    def run(self, steps, dt, **kwargs):
        
        no_ts = {'name':'no_thermostat', 'cfreq':0., 'T_set':0.}
        ts_dict = kwargs.get('thermostat', no_ts)
        ts_name = ts_dict['name']
        ts_cfreq = ts_dict['cfreq']
        ts_Tset = ts_dict['T_set']
        thermo = create_thermostat(name=ts_name, cfreq=ts_cfreq, T_set=ts_Tset)
        
        Np = self.basis.npar
        Nb = self.basis.nb
        ndim = self.basis.ndim
        m = self.basis.m_beads
        mp = self.basis.masses
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
        
        print 'Integrating...'
        for i_p in xrange(Np):     ## loop over beads rewritten in vector-matrix-formalism
            dt_ts = np.zeros(Nb)
            for i_s in xrange(steps):
                ## energy stuff
                e_pot[i_p,:,i_s] = self.basis.get_potential_energy_beads(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                e_kin[i_p,:,i_s] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,i_s])
                e_tot[i_p,:,i_s] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
                
                ## propagation
                F = self.basis.get_forces(rb_t[i_p,:,i_s], self.pot, m[i_p], om[i_p])
                v1 = vb_t[i_p,:,i_s] + F/m[i_p]*dt/2.
                rb_t[i_p,:,i_s+1] = rb_t[i_p,:,i_s]+v1*dt
                F = ( F + self.basis.get_forces(rb_t[i_p,:,i_s+1], self.pot, m[i_p], om[i_p]) )/2.
                vb_t[i_p,:,i_s+1] = vb_t[i_p,:,i_s]+F/m[i_p]*dt
                
                ## thermostatting
                dt_ts += dt
                vb_t[i_p,:,i_s+1], dt_ts = thermo(vb_t[i_p,:,i_s+1], m[i_p], dt_ts, ndim)
                
            e_pot[i_p,:,steps] = self.basis.get_potential_energy_beads(rb_t[i_p,:,steps], self.pot, m[i_p], om[i_p])
            e_kin[i_p,:,steps] = self.basis.get_kinetic_energy(m[i_p], vb_t[i_p,:,steps])
            e_tot[i_p,:,steps] = e_kin[i_p,:,i_s]+e_pot[i_p,:,i_s]
            r_t[i_p] = np.mean(rb_t[i_p],0)
            v_t[i_p] = np.mean(vb_t[i_p],0)
            E_pot[i_p] = np.mean(e_pot[i_p],0)
            E_kin[i_p] = np.mean(e_kin[i_p],0)
            E_tot[i_p] = np.mean(e_tot[i_p],0)
            
                
        
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
        
