"""
rpmd integrators.py
"""

from qmp.integrator.integrator import Integrator
import numpy as np


class rpmd_velocity_verlet_integrator(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """
    
    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)
        
        
    def run(self, steps, dt, psi_0):
        
        Np = self.data.rpmd.basis.npar
        Nb = self.data.rpmd.basis.nb
        ndim = self.data.rpmd.basis.ndim
        m = self.data.rpmd.basis.masses
        om = self.data.rpmd.basis.om
        rb_t = np.zeros((Np,Nb,steps+1,ndim))
        rb_t[:,:,0,:] = np.array(self.data.rpmd.basis.r_beads)
        vb_t = np.zeros((Np,Nb,steps+1,ndim))
        vb_t[:,:,0,:] = np.array(self.data.rpmd.basis.v_beads)
        r_t = np.zeros((Np,steps+1,ndim))
        v_t = np.zeros((Np,steps+1,ndim))
        E_pot = np.zeros((Np,steps+1))
        E_kin = np.zeros((Np,steps+1))
        E_tot = np.zeros((Np,steps+1))
        e_pot = np.zeros((Np,Nb,steps+1))
        e_kin = np.zeros((Np,Nb,steps+1))
        e_tot = np.zeros((Np,Nb,steps+1))
        
        for i_p in xrange(Np):
            for i_b in xrange(Nb):    ## maybe rewrite in matrix-vector-formalism?
                for i_s in xrange(steps):
                    e_pot[i_p,i_b,i_s] = self.data.rpmd.basis.get_potential_energy(rb_t[i_p,i_b,i_s], rb_t[i_p,i_b-1,i_s], self.pot, m[i_p], om[i_p])
                    e_kin[i_p,i_b,i_s] = self.data.rpmd.basis.get_kinetic_energy(m[i_p], vb_t[i_p,i_b,i_s])
                    e_tot[i_p,i_b,i_s] = e_kin[i_p,i_b,i_s]+e_pot[i_p,i_b,i_s]
                    
                    F = self.data.rpmd.basis.get_forces(rb_t[i_p,i_b,i_s], rb_t[i_p,i_b-1,i_s], self.pot, m[i_p], om[i_p])
                    v1 = vb_t[i_p,i_b,i_s] + F/m[i_p]*dt/2.
                    rb_t[i_p,i_b,i_s+1] = rb_t[i_p,i_b,i_s]+v1*dt
                    F = ( F + self.data.rpmd.basis.get_forces(rb_t[i_p,i_b,i_s+1], rb_t[i_p,i_b-1,i_s], self.pot, m[i_p], om[i_p]) )/2.
                    vb_t[i_p,i_b,i_s+1] = vb_t[i_p,i_b,i_s]+F/m[i_p]*dt
                    
		e_pot[i_p,i_b,steps] = self.data.rpmd.basis.get_potential_energy(rb_t[i_p,i_b,steps], rb_t[i_p,i_b-1,steps], self.pot, m[i_p], om[i_p])
		e_kin[i_p,i_b,steps] = self.data.rpmd.basis.get_kinetic_energy(m[i_p], vb_t[i_p,i_b,steps])
		e_tot[i_p,i_b,steps] = e_kin[i_p,i_b,i_s]+e_pot[i_p,i_b,i_s]
	    E_pot[i_p] = np.mean(e_pot[i_p],0)
	    E_kin[i_p] = np.mean(e_kin[i_p],0)
	    E_tot[i_p] = np.mean(e_pot+e_kin)
            r_t[i_p] = np.mean(rb_t[i_p],0)
            v_t[i_p] = np.mean(vb_t[i_p],0)
                
        
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
        