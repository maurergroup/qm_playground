import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def project_gaussian(evecs, basis, sigma=1., x0=0., p0=0.):
    """
    project wave packet onto eigenvectors, return vector of coefficients
    
    parameters:
    ===========
        evecs: matrix containing eigenvectors(eigenvalue problem solved beforehand)
        basis: grid wavepacket will be created on
        sigma: variance of wave packet
        x0:    center of wave packet
    """
    x = basis
    gauss_wave = create_gaussian(x, x0=x0, sigma=sigma, p0=p0)
    c = np.dot(gauss_wave, evecs)
    return c

    
def project_gaussian2D(evecs, xgrid, ygrid, sigma=1., x0=[0.,0.], p0=[0.,0.]):
    """
    project 2D wave packet onto eigenvectors, return vector of coefficients
    
    parameters:
    ===========
        evecs: matrix containing flattened eigenvectors
        *grid: x/y grid wavepacket will be created on
        sigma: variance of wave packet, opt: [sigma_x, sigma_y]
        x0:    center of wave packet [x0,y0]
        """
    gauss_wave = create_gaussian2D(xgrid, ygrid, x0=x0, p0=p0, sigma=sigma)
    c = np.dot(gauss_wave.flatten(), evecs)
    return c


def create_gaussian(x, x0=0., p0=0., sigma=1.):
    """
    creates gaussian wave
    
    parameters:
    ===========
        x:      grid for wave packet
        x0:     center/expectation value of gaussian (default 0.)
        p0:     initial momentum of wave (default 0.)
        sigma:  variance of gaussian (default 1.)
    """
    wave = np.exp( -((x-x0)**2/sigma**2/4.) + 1j*p0*(x-x0))/(np.sqrt(np.sqrt(2.*np.pi)*sigma))
    return wave
    
def create_gaussian2D(xgrid, ygrid, x0=[0.,0.], p0=[0.,0.], sigma=[1.,1.]):
    """
    creates 2D gaussian wave
    
    parameters:
    ===========
        *grid:  x/y grid wave will be constructed on
        x0:     (initial) center/expectation value of wave (default [0.,0.])
        p0:     initial momentum of wave (default [0.,0.])
        sigma:  variance of gaussian in x and y direction (default [1.,1.])
    """
    if (type(sigma) == float) or (type(sigma) == int):
        sigma = [sigma, sigma]
    
    wave = np.exp( -(1/2.)*(((xgrid-x0[0])/sigma[0])**2 + ((ygrid-x0[1])/sigma[1])**2) + 1j*(p0[0]*(xgrid-x0[0]) + p0[1]*(ygrid-x0[1])) )
    return wave


def create_thermostat(name='no_thermostat', **kwargs):
    """
    returns thermostat as defined by **kwargs
    """
    from qmp.utilities import kB
    from scipy.stats import norm
    
    def andersen_ts(v, m, dt, ndim):
        cfreq = kwargs.get('cfreq', 0.001)
        T_set = kwargs.get('T_set', 293.)
        
        p_rand = np.random.random(v.shape[0])
        mask = np.array([(p_rand < cfreq*dt)]*ndim).T
        #if p_rand < cfreq*dt:
        s = np.sqrt(kB*T_set/m)
        p_rand = np.random.random(v.shape)
        #v = norm.ppf(p_rand,scale=s)
        #dt = 0.
        
        v_rand = norm.ppf(p_rand,scale=s)
        v = v_rand*mask + v*(1.-mask)
        dt = dt*mask[:,0]
        
        return v, dt
    
    def no_ts(v, m, dt):
        return v, 0
    
    if name == 'no_thermostat':
        return no_ts
    elif (name == 'Andersen') or (name == 'andersen'):
        return andersen_ts
    else:
        raise KeyError("Thermostat '"+name+"' is not implemented yet or misspelled. Available: 'Andersen', 'no_thermostat'")
    