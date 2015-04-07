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
    gauss_wave = amplitude*np.exp( -(1/2.)*((x-x0)/sigma)**2 )
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
    if len([sigma]) != 2:
        sigma = [sigma, sigma]
        
    gauss_wave = amplitude*np.exp( -(1/2.)*((xgrid-x0[0])**2/sigma[0] + (ygrid-x0[1])**2/sigma[1]) )
    c = np.dot(gauss_wave.flatten(), evecs)
    return c