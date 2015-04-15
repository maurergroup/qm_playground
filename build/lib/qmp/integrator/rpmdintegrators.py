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
        rb_t = np.array(self.data.rpmd.basis.r_beads)
        vb_t = np.array(self.data.rpmd.basis.v_beads)
        r_t = np.zeros((Np,steps+1,ndim))
        v_t = np.zeros((Np,steps+1,ndim))
        E_pot = np.zeros((Np,steps+1))
        E_kin = np.zeros((Np,steps+1))
        E_tot = np.zeros((Np,steps+1))
        
        for i_par in xrange(Np):
            rb = np.array([rb_t[i_par].flatten().reshape(Nb,ndim)])
            vb = np.array([vb_t[i_par].flatten().reshape(Nb,ndim)])
            for i_step in xrange(steps):
                e_pot = self.data.rpmd.basis.get_potential_energy(rb[i_step], self.pot, m[i_par], om[i_par])
                e_kin = self.data.rpmd.basis.get_kinetic_energy(m[i_par], vb[i_step])
                E_pot[i_par,i_step] = np.mean(e_pot)
                E_kin[i_par,i_step] = np.mean(e_kin)
                E_tot[i_par,i_step] = np.mean(e_pot+e_kin)
                
                F = self.data.rpmd.basis.get_forces(rb[i_step], self.pot, m[i_par], om[i_par])
                v1 = vb[i_step].T + F/m[i_par]*dt/2.
                rb = np.append(rb, (rb[i_step].T+v1*dt).T).reshape(i_step+2,Nb,ndim)
                F = ( F + self.data.rpmd.basis.get_forces(rb[i_step+1], self.pot, m[i_par], om[i_par]) )/2.
                vb = np.append(vb, (vb[i_step].T+F/m[i_par]*dt).T).reshape(i_step+2,Nb,ndim)
                
            e_pot = self.data.rpmd.basis.get_potential_energy(rb[steps], self.pot, m[i_par], om[i_par])
            e_kin = self.data.rpmd.basis.get_kinetic_energy(m[i_par], vb[steps])
            E_pot[i_par,steps] = np.mean(e_pot)
            E_kin[i_par,steps] = np.mean(e_kin)
            E_tot[i_par,steps] = np.mean(e_pot+e_kin)
            
            r_t[i_par] = np.mean(rb,1)
            v_t[i_par] = np.mean(vb,1)
            
        
        self.data.rpmd.r_t = r_t
        self.data.rpmd.v_t = v_t
        self.data.rpmd.E_kin_t = np.array(E_kin)
        self.data.rpmd.E_pot_t = np.array(E_pot)
        self.data.rpmd.E_t = np.array(E_tot)
        