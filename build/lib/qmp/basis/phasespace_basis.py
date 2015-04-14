"""
initialization arrays for classical dynamics
"""


from qmp.basis.basis import basis
from qmp.utilities import *
import numpy as np


class phasespace(basis):
    """
    initialization array in phase space
    """

    def __init__(self, coordinates, velocities, masses):
        """
        
        """

        basis.__init__(self)
        
        self.r = np.array(coordinates)
        self.npar = self.r.shape[0]
        if self.r.size == self.npar:
            self.ndim = 1
        else:
            self.ndim = self.r.shape[1]
        self.v = np.array(velocities)
        self.masses = np.array(masses)
        
        if self.masses.size != self.masses.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif (self.masses.size != self.npar) or \
             (self.r.shape != self.v.shape):
            raise ValueError('Please provide consistent masses, coordinates, and velocities')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D implemented yet')
        

    def __eval__(self):
        return self.r, self.v

    def get_kinetic_energy(self, masses, v):
        return masses*np.sum(v*v,1)/2.

    def get_potential_energy(self, r, pot):
        return pot(*np.array(r).T)
#        except:
#            raise ValueError('Could not evaluate potential with given coordinates')
            
    
    def get_forces(self, r, pot):
        if self.ndim == 1:
            return -1.*num_deriv(pot, *np.array(r).T)
        elif self.ndim == 2:
            return -1.*num_deriv_2D(pot, *np.array(r).T)
    