import numpy as np

def project_gaussian(evecs, basis, sigma, x0):
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
    gauss_wave = gaussian(x, sigma, x0)
    c = np.dot(gauss_wave, evecs)
    return c

    
def gaussian(x, sigma, x0):
    """
    return normally distributed gaussian with
    variance sigma and expectation value x0
    """
    return np.exp( -(1/2.)*((x-x0)/sigma)**2 )/sigma#/np.sqrt(2.*np.pi)