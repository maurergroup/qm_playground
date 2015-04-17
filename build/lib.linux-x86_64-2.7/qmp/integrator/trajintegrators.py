"""
trajintegrators.py
"""

from qmp.integrator.integrator import Integrator
import numpy as np


class velocity_verlet_integrator(Integrator):
    """
    Velocity verlet integrator for classical dynamics
    """
    
    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)
        
        
    def run(self, steps, dt, psi_0):
        
        N = self.data.traj.basis.npar
        ndim = self.data.traj.basis.ndim
        m = self.data.traj.basis.masses
        r_t = np.array([self.data.traj.basis.r])
        v_t = np.array([self.data.traj.basis.v])
        E_pot = []
        E_kin = []
        E_tot = []
        
        for i in xrange(steps):
            e_pot = self.data.traj.basis.get_potential_energy(r_t[i], self.pot)
            e_kin = self.data.traj.basis.get_kinetic_energy(m, v_t[i])
            E_pot.append(e_pot)
            E_kin.append(e_kin)
            E_tot.append(e_pot+e_kin)
            
            F = self.data.traj.basis.get_forces(r_t[i], self.pot)
            v1 = v_t[i].T + F/m*dt/2.
            r_t = np.append(r_t, (r_t[i].T+v1*dt).T).reshape(i+2,N,ndim)
            F = ( F + self.data.traj.basis.get_forces(r_t[i+1], self.pot) )/2.
            v_t = np.append(v_t, (v_t[i].T+F/m*dt).T).reshape(i+2,N,ndim)
            
        e_pot = self.data.traj.basis.get_potential_energy(r_t[-1], self.pot)
        e_kin = self.data.traj.basis.get_kinetic_energy(m, v_t[-1])
        E_pot.append(e_pot)
        E_kin.append(e_kin)
        E_tot.append(e_pot+e_kin)
        
        self.data.traj.r_t = r_t
        self.data.traj.v_t = v_t
        self.data.traj.E_kin_t = np.array(E_kin)
        self.data.traj.E_pot_t = np.array(E_pot)
        self.data.traj.E_t = np.array(E_tot)
        