import numpy as np

def project_gaussian(evecs, basis, amplitude=1., sigma=1., x0=0.):
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
    gauss_wave = create_gaussian(x, x0=x0, amplitude=amplitude, sigma=sigma)
    c = np.dot(gauss_wave, evecs)
    return c

    
def project_gaussian2D(evecs, xgrid, ygrid, amplitude=1., sigma=[1.,1.], x0=[0.,0.]):
    """
    project 2D wave packet onto eigenvectors, return vector of coefficients
    
    parameters:
    ===========
        evecs: matrix containing flattened eigenvectors
        *grid: x/y grid wavepacket will be created on
        sigma: variance of wave packet, opt: [sigma_x, sigma_y]
        x0:    center of wave packet [x0,y0]
        """
    gauss_wave = create_gaussian2D(xgrid, ygrid, x0=x0, amplitude=amplitude, sigma=sigma)
    c = np.dot(gauss_wave.flatten(), evecs)
    return c


def create_gaussian(x, x0=0., amplitude=1., sigma=1.):
    """
    creates gaussian wave
    
    parameters:
    ===========
        x:          grid for gaussian
        x0:         center/expectation value of gaussian (default 0.)
        amplitude:  amplitude of wave (default 1.)
        sigma:      variance of gaussian (default 1.)
    """
    return amplitude*np.exp( -(1/2.)*((x-x0)/sigma)**2 )/(np.sqrt(2*np.pi*sigma**2))
    
def create_gaussian2D(xgrid, ygrid, x0=[0.,0.], amplitude=1., sigma=1.):
    """
    creates 2D gaussian wave
    
    parameters:
    ===========
        *grid:      x/y grid wave will be constructed on
        x0:         center/expectation value of wave (default [0.,0.])
        amplitude:  amplitude of wave (default 1.)
        sigma:      variance of gaussian, optional: sigma = [sigma_x, sigma_y] (default [1.,1.])
    """
    if len([sigma]) != 2:
        sigma = [sigma, sigma]
    
    return amplitude*np.exp( -(1/2.)*((xgrid-x0[0])**2/sigma[0] + (ygrid-x0[1])**2/sigma[1]) )