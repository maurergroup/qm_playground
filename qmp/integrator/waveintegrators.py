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

        wvfn = self.data.wvfn

        #construct H
        T=self.data.basis.construct_Tmatrix()
        V=self.data.basis.construct_Vmatrix(self.pot)
        H = T+V
        
        print 'Herrooo'

        prop = np.exp(-1j*H*(dt/hbar) )
        c = self.data.c
        psi = self.data.wvfn.psi
        for i in range(steps):
            print 'Time Step : ', i
            self.counter +=1 
            c = np.dot(prop,c)
            print c
            #E = np.dot(c\cc,psi),np.dot(H,np.dot(c,psi))
