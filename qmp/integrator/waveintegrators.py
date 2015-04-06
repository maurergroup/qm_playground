"""
waveintegrators.py
"""

from qmp.utilities import *
from qmp.integrator.integrator import Integrator
import numpy as np

class prim_propagator(Integrator):
    """
    Primitive exp(-iEt) propagator
    """

    def __init__(self, **kwargs):
        """
        Primprop init
        """
        Integrator.__init__(self, **kwargs)

        #prepare data object for time integration
        #we need trajectories for different observables
        #every integrator owns a Logger which is either 
        #empty or wants data

        self.data.c = np.zeros_like(self.data.wvfn.E) 
        self.data.c[0] = 1.0 

    def run(self, steps, dt):
        """
        Propagates the system for 'steps' steps and 
        a timestep of dt.
        """

        #wvfn = self.data.wvfn

        #construct H
        #T=self.data.wvfn.basis.construct_Tmatrix()
        #V=self.data.wvfn.basis.construct_Vmatrix(self.pot)
        #H = T+V
        
        print 'Herrooo'

        #prop = np.exp(-1j*H*(dt/hbar) )    #(x,x)
        prop = np.diag(np.exp(-1j*self.data.wvfn.E*dt/hbar))    #(states,states)
        
        c = self.data.c     #(states,1) is there somewhere eigendecomposition for c?
                            #where does c come from?
        
        psi_basis = self.data.wvfn.psi     #(x,states)
        psi = [psi_basis.dot(c)]    #(x,1)
        
        for i in xrange(1,steps+1):
            #print 'Time Step : ', i
            self.counter +=1 
            c = np.dot(prop,c)
            psi = np.append(psi, np.dot(psi_basis,c))
            psi = psi.reshape(i+1,psi_basis.shape[0])
            #print c
            #E = np.dot(c\cc,psi),np.dot(H,np.dot(c,psi))
            
        self.data.wvfn.psi_t = np.array(psi)
        
