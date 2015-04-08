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
        

    def run(self, steps, dt, psi_0):
        """
        Propagates psi_0 for 'steps' timesteps of length 'dt'.
        """

        if self.data.wvfn.E is None:
            raise ValueError('System has to be solved first for eigendecomposition!')
        
        psi_basis = self.data.wvfn.psi     #(x,states)
        
        if steps == 0:
            self.data.c[0] = 1
            c = self.data.c
        elif (np.all(self.data.c) == 0.) and (psi_0 == 0.):
            raise ValueError('Integrator needs either expansion coefficients \
                             or initial wave function to propagate system!')
        elif psi_0 == 0.:
            c = self.data.c
        elif (len(psi_0.flatten()) != self.data.wvfn.psi.shape[0]):
            raise ValueError('Initial wave function needs to be defined on \
                             same grid as system was solved on!')
        else:
            c = np.dot(psi_0.flatten(), psi_basis)
        
        prop = np.diag(np.exp(-1j*self.data.wvfn.E*dt/hbar))    #(states,states)
        psi = [psi_basis.dot(c)]    #(x,1)
        
        print 'Integrating...'
        for i in xrange(1,steps+1):
            #print 'Time Step : ', i
            self.counter +=1 
            c = np.dot(prop,c)
            psi = np.append(psi, np.dot(psi_basis,c))
            psi = psi.reshape(i+1,psi_basis.shape[0])
            #E = np.dot(c\cc,psi),np.dot(H,np.dot(c,psi))
            
        self.data.wvfn.psi_t = np.array(psi)
        

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
        
    def run(self, steps, dt, psi_0):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        import scipy.sparse.linalg as la
        import scipy.sparse as sparse
        
        #construct H
        T=self.data.wvfn.basis.construct_Tmatrix()
        V=self.data.wvfn.basis.construct_Vmatrix(self.pot)
        
        if (psi_0.all() == 0.) or (len(psi_0.flatten()) != T.shape[1]):
            raise ValueError('Please provide initial wave function on appropriate grid')

        H = np.array(T+V)
        prop = np.exp(-1j*H*dt/hbar)    #(x,x)
        psi = np.array([psi_0])    #(x,1)
        
        print 'Integrating...'
        for i in xrange(steps):
            #print 'Time Step : ', i
            self.counter +=1 
            psi = np.append(psi, np.dot(prop,psi[i]))
            psi = psi.reshape(i+2,T.shape[0])
            #E = np.dot(c\cc,psi),np.dot(H,np.dot(c,psi))
            
        self.data.wvfn.psi_t = np.array(psi)
        

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

    def run(self, steps, dt):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """

        V = self.data.wvfn.basis.construct_Vmatrix(self.pot)

        if (psi_0.all() == 0.) or (len(psi_0.flatten()) != V.shape[1]):
            raise ValueError('Please provide initial wave function on appropriate grid')

        expV = np.exp(-1j*V*dt/hbar)

        #fancy implementation

        #wvfn = self.data.wvfn

        #construct H
        #T=self.data.wvfn.basis.construct_Tmatrix()
        #V=self.data.wvfn.basis.construct_Vmatrix(self.pot)
        #H = T+V
        
        #print 'Herrooo'

        #prop = np.exp(-1j*H*(dt/hbar) )    #(x,x)
        #prop = np.diag(np.exp(-1j*self.data.wvfn.E*dt/hbar))    #(states,states)
        
        #c = self.data.c     #(states,1)
        
        #psi_basis = self.data.wvfn.psi     #(x,states)
        #psi = [psi_basis.dot(c)]    #(x,1)
        
        print 'Integrating...'
        for i in xrange(1,steps+1):
            #print 'Time Step : ', i
            self.counter +=1 
            #c = np.dot(prop,c)
            #psi = np.append(psi, np.dot(psi_basis,c))
            #psi = psi.reshape(i+1,psi_basis.shape[0])
            #E = np.dot(c\cc,psi),np.dot(H,np.dot(c,psi))
            
        self.data.wvfn.psi_t = np.array(psi)
        
