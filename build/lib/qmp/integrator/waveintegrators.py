"""
waveintegrators.py
"""

from qmp.utilities import *
from qmp.integrator.integrator import Integrator
import numpy as np

class eigen_propagator(Integrator):
    """
    Primitive exp(-iEt) propagator for psi in eigenbasis
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        Integrator.__init__(self, **kwargs)

        #prepare data object for time integration
        #we need trajectories for different observables
        #every integrator owns a Logger which is either 
        #empty or wants data

        self.data.c = np.zeros_like(self.data.wvfn.E)
        

    def run(self, steps, dt, psi_0x, psi_0p):
        """
        Propagates psi_0x for 'steps' timesteps of length 'dt'.
        """

        if self.data.solved == False:
            raise ValueError('System has to be solved first for eigendecomposition!')
                
        psi_basis = self.data.wvfn.psi     #(x,states)
        
        if steps == 0:
            self.data.c[0] = 1
            c = self.data.c
        elif (np.all(self.data.c) == 0.) and (np.all(psi_0x) == 0.):
            raise ValueError('Integrator needs either expansion coefficients \
                             or initial wave function to propagate system!')
        elif np.all(psi_0x) == 0.:
            c = self.data.c
        elif (len(psi_0x.flatten()) != self.data.wvfn.psi.shape[0]):
            raise ValueError('Initial wave function needs to be defined on \
                             same grid as system was solved on!')
        else:
            c = np.dot(psi_0x.flatten(), psi_basis)
        
        prop = np.diag(np.exp(-1j*self.data.wvfn.E*dt/hbar))    #(states,states)
        psi = [psi_basis.dot(c)]    #(x,1)
        E = [np.dot(np.conjugate(c), c)]
        
        self.counter = 0
        print 'Integrating...'
        for i in xrange(1,steps+1):
            #print 'Time Step : ', i
            self.counter += 1 
            c = np.dot(prop,c)
            psi = np.append(psi, np.dot(psi_basis,c))
            psi = psi.reshape(i+1,psi_basis.shape[0])
            E.append(np.dot(np.conjugate(c), c))
            
        self.data.wvfn.psi_t = np.array(psi)
        self.data.wvfn.E_t = np.array(E)
        

class prim_propagator(Integrator):
    """
    Primitive exp(-iHt) propagator for psi in arbitrary basis in spatial representation
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        Integrator.__init__(self, **kwargs)

        #prepare data object for time integration
        #we need trajectories for different observables
        #every integrator owns a Logger which is either 
        #empty or wants data

        ##something?
        
    def run(self, steps, dt, psi_0x, psi_0p):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        import scipy.linalg as la
        
        #construct H
        T=self.data.wvfn.basis.construct_Tmatrix()
        V=self.data.wvfn.basis.construct_Vmatrix(self.pot)
        
        if (psi_0x.all() == 0.) or (len(psi_0x.flatten()) != T.shape[1]):
            raise ValueError('Please provide initial wave function on appropriate grid')

        H = np.array(T+V)
        prop = la.expm(-1j*H*dt/hbar)    #(x,x)
        psi = np.array([psi_0x.flatten()])    #(1,x)
        E = [np.dot(np.conjugate(psi_0x.flatten()), np.dot(H,psi_0x.flatten()))]
        self.counter = 0
        
        print 'Integrating...'
        for i in xrange(steps):
            #print 'Time Step : ', i
            self.counter +=1 
            psi = np.append(psi, np.dot(prop,psi[i]))
            psi = np.reshape(psi, (i+2,T.shape[0]))
            E.append(np.dot(psi[i+1].conjugate(), np.dot(H,psi[i+1])))
            
        self.data.wvfn.psi_t = np.array(psi)
        self.data.wvfn.E_t = np.array(E)
        

class split_operator_propagator(Integrator):
    """
    Split operator propagator for psi(x,0)
        Trotter series: exp(iHt) ~= exp(iVt/2)*exp(iTt)*exp(iVt/2)
        => exp(iHt)*psi(x) ~= exp(iVt/2)*exp(iTt)*exp(iVt/2)*psi(x)
        => use spatial representation for exp(iVt/2) and momentum representation for exp(iTt)
        => psi(x,t) = exp(iVt/2)*iFT(t*p**2/2m*FT(exp(iVt/2)*psi(x,0)))
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        Integrator.__init__(self, **kwargs)

        #prepare data object for time integration
        #we need trajectories for different observables
        #every integrator owns a Logger which is either 
        #empty or wants data

        ##whatever has to be here

    def run(self, steps, dt, psi_0x, psi_0p):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """

        V = self.data.wvfn.basis.construct_Vmatrix(self.pot)

        if (psi_0x.all() == 0.) or (len(psi_0x.flatten()) != V.shape[1]):
            raise ValueError('Please provide initial wave function on appropriate grid')

        m = self.data.mass
        expV = np.exp(-1j*V*dt/hbar/2.)
        p_op = -1j*hbar*self.data.wvfn.basis.Nabla_psi()
        psi = np.array([psi_0x.flatten()])
        if np.all(psi_0p) == 0.:
            psi_p = np.dot(np.conjugate(psi_0x).dot(p_op), psi_0x)
        else:
            psi_p = psi_0p
        
        self.counter = 0
        
        print 'Integrating...'
        for i in xrange(steps):
            self.counter += 1
            expT = np.exp(np.diag(-1j*(dt/hbar)*(psi_p**2)/(2.*m)))
            psi1 = np.dot(expV, psi[i])
            psi2 = np.dot(expT, FT(psi1))
            psi3 = np.dot(expV, iFT(psi2))
            psi = (np.append(psi, psi3)).reshape(i+2,V.shape[0])
            psi_p = np.dot(np.conjugate(psi[i]).dot(p_op), psi[i])
            
        self.data.wvfn.psi_t = np.array(psi)
        
