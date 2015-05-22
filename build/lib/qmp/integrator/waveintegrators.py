"""
waveintegrators.py
"""

from qmp.utilities import *
from qmp.integrator.integrator import Integrator
from qmp.integrator.dyn_tools import project_wvfn
from qmp.termcolors import *
import numpy as np
import os


def remove_restart(filename):
    """
    Removes filename from current directory, if existing
    """
    try:
        os.remove(filename)
    except OSError:
        pass


class eigen_propagator(Integrator):
    """
    Projects initial wavefunction onto eigenbasis, propagates expansion coefficients
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
        

    def run(self, steps, dt, **kwargs):
        """
        Propagates psi_0 for 'steps' timesteps of length 'dt'.
        """

        psi_0 = kwargs.get('psi_0', 0.)
        psi_basis = self.data.wvfn.psi     #(x,states)

        if steps == 0:
            self.data.c[0] = 1
            c = self.data.c
        elif not (np.any(self.data.c) != 0.) and not (np.any(psi_0) != 0.):
            raise ValueError('Integrator needs either expansion coefficients \
or initial wave function to propagate system!')
        elif not (np.any(psi_0) != 0.):
            c = self.data.c
            norm = np.sqrt(np.conjugate(c).dot(c))
            c /= norm
        elif (len(psi_0.flatten()) != psi_basis.shape[0]):
            raise ValueError('Initial wave function needs to be defined on \
same grid as system was solved on!')
        else:
            states = psi_basis.shape[1]
            print gray+'Projecting wavefunction onto basis of '+str(states)+' eigenstates'+endcolor
            if psi_basis.shape[0] != states:
                print gray+'**WARNING: This basis is incomplete, coefficients and wavefunction might contain errors**'+endcolor
            c = np.array([project_wvfn(psi_0, psi_basis)])
        
        prop = np.diag(np.exp(-1j*self.data.wvfn.E*dt/hbar))    #(states,states)
        psi = [psi_basis.dot(c[0])]    #(x,1)
        E = [np.dot(np.conjugate(c[0]), (c[0]*self.data.wvfn.E))]
        
        self.counter = 0
        print gray+'Integrating...'+endcolor
        for i in xrange(1,steps+1):
            self.counter += 1 
            c = np.append(c, np.dot(prop,c[i-1])).reshape(i+1, states)
            psi = np.append(psi, np.dot(psi_basis,c[i])).reshape(i+1,psi_basis.shape[0])
            E.append(np.dot(np.conjugate(c[i]), (c[i]*self.data.wvfn.E)))
            
        print gray+'INTEGRATED\n'+endcolor
            
        self.data.wvfn.psi_t = np.array(psi)
        self.data.wvfn.c_t = np.array(c)
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
        
    def run(self, steps, dt, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        import scipy.linalg as la
        
        psi_0 = kwargs.get('psi_0')
        
        #construct H
        T=self.data.wvfn.basis.construct_Tmatrix()
        V=self.data.wvfn.basis.construct_Vmatrix(self.pot)
        
        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != T.shape[1]):
            raise ValueError('Please provide initial wave function on appropriate grid')

        H = np.array(T+V)
        prop = la.expm(-1j*H*dt/hbar)    #(x,x)
        psi = np.array([psi_0.flatten()])    #(1,x)
        E = [np.dot(np.conjugate(psi_0.flatten()), np.dot(H,psi_0.flatten()))]
        self.counter = 0
        
        print gray+'Integrating...'+endcolor
        for i in xrange(steps):
            self.counter +=1 
            psi = np.append(psi, np.dot(prop,psi[i]))
            psi = np.reshape(psi, (i+2,T.shape[0]))
            E.append(np.dot(psi[i+1].conjugate(), np.dot(H,psi[i+1])))
            
        print gray+'INTEGRATED\n'+endcolor
            
        self.data.wvfn.psi_t = np.array(psi)
        self.data.wvfn.E_t = np.array(E)
        

class SOFT_propagation(Integrator):
    """
    Split operator propagator for psi(x,0)
        Trotter series: exp(iHt) ~= exp(iVt/2)*exp(iTt)*exp(iVt/2)
        => exp(iHt)*psi(x) ~= exp(iTt/2)*exp(iVt)*exp(iTt/2)*psi(x)
        => use spatial representation for exp(iVt) and momentum representation for exp(iTt/2)
        => psi(x,t) = iFT(exp(t*p**2/4m) * FT(exp(iVt) * iFT(exp(t*p**2/4m) * FT(psi(x,0)))))
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
    
    
    def run(self, steps, dt, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT
        from numpy.fft import fftfreq as FTp
        import cPickle as pick
        
        psi_0 = kwargs.get('psi_0')
        add_info = kwargs.get('additional', None)
        
        V = self.data.wvfn.basis.get_potential_flat(self.pot)
        N = V.size
        
        if type(psi_0)==str:
            try:
                restart_file = open(psi_0,'rb')
                current_data = pick.load(restart_file)
                psi = current_data['psi']
                rho = current_data['rho']
                E = current_data['E_tot']
                E_kin = current_data['E_kin']
                E_pot = current_data['E_pot']
                i_start = current_data['i']+1
            except:
                raise ValueError('Given input does neither refer to wave function nor to an existing restart file.')
        else:
            psi = [np.array(psi_0.flatten())]
            rho = np.conjugate(psi)*psi
            E, E_kin, E_pot = [], [], []
            i_start = 0
        
        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != N):
            raise ValueError('Please provide initial wave function on appropriate grid')
        
        if (add_info == 'coefficients'):
            psi_basis = self.data.wvfn.psi
            states = psi_basis.shape[1]
            print gray+'Projecting wavefunction onto basis of '+str(states)+' eigenstates'+endcolor
            if psi_basis.shape[0] != states:
                print gray+'**WARNING: This basis is incomplete, coefficients might contain errors**'+endcolor
            c_t = [project_wvfn(psi_0, psi_basis)]
            
        m = self.data.mass
        dx = self.data.wvfn.basis.dx
        nx = self.data.wvfn.basis.N
        nDim = self.data.ndim
        
        expV = np.exp(-1j*V*dt/hbar)
        if nDim == 1:
            p = np.pi*FTp(N, dx)
            p = p**2
        elif nDim == 2:
            p = FTp(nx,dx).conj()*FTp(nx,dx)
            p = np.pi**2*(np.kron(np.ones(nx), p) + np.kron(p, np.ones(nx)))
        else:
            raise NotImplementedError('Only evolving 1D and 2D systems implemented')
        
        expT = np.exp(-1j*(dt/hbar)*p/m)
        
        self.counter = 0
        
        print gray+'Integrating...'+endcolor
        for i in xrange(i_start, steps):
            self.counter += 1
            psi1 = iFT( expT*FT(psi[i]) ) 
            psi2 = FT( expV*psi1 ) 
            psi3 = iFT( expT*psi2 )
            rho += np.conjugate(psi3)*psi3
            psi.append(psi3)
            if add_info == 'coefficients':
                c_t.append(project_wvfn(psi3, psi_basis))
            
            e_kin = (np.conjugate(psi3).dot( iFT(2.*p/m * FT(psi3)) ))
            e_pot = np.conjugate(psi3).dot(V*psi3)
            E_kin.append(e_kin)
            E_pot.append(e_pot)
            E.append(e_kin+e_pot)
            
            if (np.mod(i+1, 1000000)==0):
                out = open('wave_dyn.rst', 'wb')
                wave_data = {'psi':psi, 'rho':rho, 'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E, 'i':i}
                pick.dump(wave_data,out)
            
            
        print gray+'INTEGRATED\n'+endcolor
        
        psi_ft = FT(psi[-1])
        e_kin = np.dot(np.conjugate(psi[-1]), iFT( 2.*p/m*psi_ft))
        e_pot = np.conjugate(psi[-1]).dot(V*psi[-1])
        E_kin.append(e_kin)
        E_kin = np.array(E_kin)
        E_pot.append(e_pot)
        E_pot = np.array(E_pot)
        E.append(e_kin+e_pot)
        E = np.array(E)
        c_t = np.array(c_t)
        self.data.wvfn.psi_t = np.array(psi)
        if add_info == 'coefficients':
            self.data.wvfn.c_t = c_t
        self.data.wvfn.E_t = E
        self.data.wvfn.E_kin_t = np.array(E_kin)
        self.data.wvfn.E_pot_t = np.array(E_pot)
        self.data.wvfn.E_mean = np.sum(E)/i
        self.data.wvfn.E_k_mean = np.sum(E_kin)/i
        self.data.wvfn.E_p_mean = np.sum(E_pot)/i
        self.data.wvfn.rho_mean = rho/i

        ## write psi, rho to binary output file
        out = open('wave_dyn.end', 'wb')
        wave_data = {'psi':psi, 'rho':rho, 'E_kin':E_kin, 'E_pot':E_pot, 'E_tot':E}
        pick.dump(wave_data,out)
        ## remove restart files
        remove_restart('wave_dyn.rst')
            
        
        
    

class SOFT_average_properties(Integrator):
    """
    SOFT propagator for psi(r,0) to determine expectation values from long simulations
    
    output:
    =======
        rho:    average density at r, \sum_{steps} |psi(r,step)|^2/steps
        E_tot:  average total energy, \sum_{steps} E_tot/steps (should be constant)
        E_kin:  average kinetic energy, \sum_{steps} E_kin/steps
        E_pot:  average potential energy, \sum+{steps} E_pot/steps
        (TODO: optionally output average coefficients of basis functions
               => propability distribution)
    """
    
    def __init__(self, **kwargs):
        """
        Initialization
        """
        Integrator.__init__(self, **kwargs)
    
        #something?
    
    
    def run(self, steps, dt, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        """
        ## import stuff
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT
        from numpy.fft import fftfreq as FTp
        import cPickle as pick
        
        psi_0 = kwargs.get('psi_0')
        add_info = kwargs.get('additional', None)
        
        V = self.data.wvfn.basis.get_potential_flat(self.pot)
        N = V.size
        
        if type(psi_0)==str:
            try:
                restart_file = open(psi_0,'rb')
                current_data = pick.load(restart_file)
                i_start = current_data['i']
                psi = current_data['psi']
                rho = current_data['rho']*i_start
                E = current_data['E_tot']*i_start
                E_kin = current_data['E_kin']*i_start
                E_pot = current_data['E_pot']*i_start
                i_start+=1
            except:
                raise ValueError('Given input does neither refer to wave function nor to an existing restart file.')
        else:
            psi = np.array(psi_0.flatten())
            rho = np.conjugate(psi)*psi
            E, E_kin, E_pot = 0., 0., 0.
            i_start = 0
        
        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != N):
            raise ValueError('Please provide initial wave function on appropriate grid')
        
        if (add_info == 'coefficients'):
            psi_basis = self.data.wvfn.psi
            states = psi_basis.shape[1]
            print gray+'Projecting wavefunction onto basis of '+str(states)+' eigenstates'+endcolor
            if psi_basis.shape[0] != states:
                print gray+'**WARNING: This basis is incomplete, coefficients might contain errors**'+endcolor
            c_mean = project_wvfn(psi_0, psi_basis)
            
        m = self.data.mass
        dx = self.data.wvfn.basis.dx
        nx = self.data.wvfn.basis.N
        nDim = self.data.ndim
        
        expV = np.exp(-1j*V*dt/hbar)
        if nDim == 1:
            p = np.pi*FTp(N, dx)
            p = p**2
        elif nDim == 2:
            p = FTp(nx,dx).conj()*FTp(nx,dx)
            p = np.pi**2*(np.kron(np.ones(nx), p) + np.kron(p, np.ones(nx)))
        else:
            raise NotImplementedError('Only evolving 1D and 2D systems implemented')
        
        expT = np.exp(-1j*(dt/hbar)*p/m)
        
        self.counter = 0
        
        print gray+'Integrating...'+endcolor
        for i in xrange(i_start, steps):
            self.counter += 1
            psi1 = iFT( expT*FT(psi) ) 
            psi2 = FT( expV*psi1 ) 
            psi = iFT( expT*psi2 )
            rho += np.conjugate(psi)*psi
            if add_info == 'coefficients':
                c_mean+=project_wvfn(psi, psi_basis)
            
            e_kin = (np.conjugate(psi3).dot( iFT(2.*p/m * FT(psi3)) ))
            e_pot = np.conjugate(psi3).dot(V*psi3)
            E_kin+=e_kin
            E_pot+=e_pot
            E+=(e_kin+e_pot)
            
            if (np.mod(i+1, 1000000)==0):
                out = open('wave_avgs.rst', 'wb')
                current_data = {'psi':psi, 'rho':rho/i, 'E_kin':E_kin/i, 'E_pot':E_pot/i, 'E_tot':E/i, 'i':i}
                pick.dump(current_data,out)
            
            
        print gray+'INTEGRATED\n'+endcolor
        
        e_kin = np.dot(np.conjugate(psi), iFT( 2.*p/m*FT(psi)))
        e_pot = np.conjugate(psi).dot(V*psi)
        E_kin+=e_kin
        E_pot+=e_pot
        E+=(e_kin+e_pot)
        if add_info == 'coefficients':
            self.data.wvfn.c_mean = c_mean/i
        
        self.data.wvfn.E_tot = E/i
        self.data.wvfn.E_kin = E_kin/i
        self.data.wvfn.E_pot = E_pot/i
        self.data.wvfn.rho = rho/i
        
        ## write energies, rho to binary output file
        out = open('wave_avgs.end', 'wb')
        wave_data = {'rho':rho/i, 'E_kin':E_kin/i, 'E_pot':E_pot/i, 'E_tot':E/i}
        pick.dump(wave_data,out)
        ## remove restart files
        remove_restart('wave_avgs.rst')


class SOFT_scattering(Integrator):
    """
    SOFT propagator for scattering processes
                     _               ..
       :            / \ ->          ;  ;                        :
       :           /   \ ->         ;  ;                        :
       :          /     \ ->        ;  ;                        :
       :_________/       \__________;  ;________________________:
      r_l                            rb                        r_r
    (border)       (wave)         (barrier)                  (border)
    
    Stops if reflected or transmitted part of wave package hits
    border at r_l or r_r, respectively (=> t_stop).
    
    input:
    ======
        wave:     defined by psi_0 (incl. momentum!)
        barrier:  defined by potential class
        div_surf: dividing surface (1D: rb), given as keyword argument
    
    output:
    =======
        psi_t:    final wave package at t_stop
        p_refl:   integrated reflected part of wave, \int rho(r,t_stop) for r<rb
        p_trans:  integrated transmitted part of wave, \int rho(r,t_stop) for r>rb
        energy:   E_kin, E_pot, E_tot as functions of simulation time
        status:   information whether scattering process should be complete
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        Integrator.__init__(self, **kwargs)
        
        ## get borders, r_l and r_b, and dividing surface (1D: rb)
        ## TODO: 2D dividing surface?
        grid = self.data.wvfn.basis.x
        Vmax = np.argmax(self.pot(grid))
        rb = kwargs.get('div_surf', Vmax)
        self.rb_idx = np.argmin(abs(grid-rb))
        
        
    def run(self, steps, dt, **kwargs):
        """
        Propagates the system for 'steps' timesteps of length 'dt'.
        Stops at (rho(r_l,t_stop) + rho(r_r,t_stop)) > 1E-8.
        """
        ## import stuff
        from numpy.fft import fft as FT
        from numpy.fft import ifft as iFT
        from numpy.fft import fftfreq as FTp
        import cPickle as pick
        
        psi_0 = kwargs.get('psi_0')
        add_info = kwargs.get('additional', None)
        
        V = self.data.wvfn.basis.get_potential_flat(self.pot)
        N = V.size
        
        if type(psi_0)==str:
            try:
                restart_file = open(psi_0,'rb')
                current_data = pick.load(restart_file)
                psi = current_data['psi']
                #rho_mean = current_data['rho_mean']/current_data['i']
                E = current_data['E_tot']
                E_kin = current_data['E_kin']
                E_pot = current_data['E_pot']
                i_start = current_data['i']+1
            except:
                raise ValueError('Given input does neither refer to wave function nor to an existing restart file.')
        else:
            psi = [np.array(psi_0.flatten())]
            #rho_mean = np.conjugate(psi[0])*psi[0]
            E, E_kin, E_pot = [], [], []
            i_start = 0
        
        if (not psi_0.any() != 0.) or (len(psi_0.flatten()) != N):
            raise ValueError('Please provide initial wave function on appropriate grid')
        
        if (add_info == 'coefficients'):
            psi_basis = self.data.wvfn.psi
            states = psi_basis.shape[1]
            print gray+'Projecting wavefunction onto basis of '+str(states)+' eigenstates'+endcolor
            if psi_basis.shape[0] != states:
                print gray+'**WARNING: This basis is incomplete, coefficients might contain errors**'+endcolor
            c_t = [project_wvfn(psi_0, psi_basis)]
            
        
        status = 'Wave did not hit border(s). Scattering process might be incomplete.'
        m = self.data.mass
        dx = self.data.wvfn.basis.dx
        nx = self.data.wvfn.basis.N
        nDim = self.data.ndim
        
        expV = np.exp(-1j*V*dt/hbar)
        if nDim == 1:
            p = np.pi*FTp(N, dx)
            p = p**2
        elif nDim == 2:
            p = FTp(nx,dx).conj()*FTp(nx,dx)
            p = np.pi**2*(np.kron(np.ones(nx), p) + np.kron(p, np.ones(nx)))
        else:
            raise NotImplementedError('Only evolving 1D and 2D systems implemented')
        
        expT = np.exp(-1j*(dt/hbar)*p/m)
        
        self.counter = 0
        
        print gray+'Integrating...'+endcolor
        for i in xrange(i_start, steps):
            self.counter += 1
            psi1 = iFT( expT*FT(psi[i]) ) 
            psi2 = FT( expV*psi1 ) 
            psi3 = iFT( expT*psi2 )
            rho_current = np.conjugate(psi3)*psi3
            #rho_mean += rho_current
            psi.append(psi3)
            if add_info == 'coefficients':
                c_t.append(project_wvfn(psi3, psi_basis))
            
            e_kin = (np.conjugate(psi3).dot( iFT(2.*p/m * FT(psi3)) ))
            e_pot = np.conjugate(psi3).dot(V*psi3)
            E_kin.append(e_kin)
            E_pot.append(e_pot)
            E.append(e_kin+e_pot)
            if (rho_current[0]+rho_current[-1] > 1E-4):
                status = 'Wave hit border(s). Simulation terminated.'
                break
            
            if (np.mod(i+1, 1000000)==0):
                out = open('wave_scatter.rst', 'wb')
                current_data = {'psi':psi, 'rho_mean':rho_mean/i, 'E_tot':E, 'E_kin':E_kin, \
                                'E_pot':E_pot, 'i':i}
                pick.dump(current_data,out)
            
        
        print gray+'INTEGRATED'
        print status+'\n'+endcolor
        
        psi_ft = FT(psi[-1])
        e_kin = np.dot(np.conjugate(psi[-1]), iFT( 2.*p/m*psi_ft))
        e_pot = np.conjugate(psi[-1]).dot(V*psi[-1])
        E_kin.append(e_kin)
        E_kin = np.array(E_kin)
        E_pot.append(e_pot)
        E_pot = np.array(E_pot)
        E.append(e_kin+e_pot)
        E = np.array(E)
        
        psi = np.array(psi)
        self.data.wvfn.psi_t = psi
        p_refl = np.sum(rho_current[:self.rb_idx])
        p_trans = np.sum(rho_current[self.rb_idx:])
        
        ## write psi, rho to binary output file
        out = open('wave_scatter.end', 'wb')
        wave_data = {'psi':psi, 'p_refl':p_refl, 'p_trans':p_trans, 'E_kin':E_kin, \
                     'E_pot':E_pot, 'E_tot':E}
        pick.dump(wave_data,out)
        
        if add_info == 'coefficients':
            self.data.wvfn.c_t = np.array(c_t)
        
        self.data.wvfn.E_t = E
        self.data.wvfn.E_kin_t = E_kin
        self.data.wvfn.E_pot_t = E_pot
        self.data.wvfn.p_refl = p_refl
        self.data.wvfn.p_trans = p_trans
        self.data.wvfn.status = status
        #self.data.wvfn.rho_mean = rho_mean/i
        
        ## remove restart files
        remove_restart('wave_scatter.rst')




#--EOF--#