import numpy as np


class PhaseSpace:

    def __init__(self, coordinates, velocities, masses):

        self.r = np.array(coordinates)
        self.v = np.array(velocities)
        self.masses = np.array(masses)

        self.n_particles = self.r.shape[0]
        if self.r.size == self.n_particles:
            self.ndim = 1
        else:
            self.ndim = self.r.shape[1]

        if self.masses.size != self.masses.shape[0]:
            raise ValueError('Masses must be given as List of integers')
        elif (self.masses.size != self.n_particles) or \
             (self.r.shape != self.v.shape):
            raise ValueError(
                    'Inconsistent masses, coordinates, and velocities.')
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D implemented.')

    def compute_kinetic_energy(self):
        return 0.5 * self.masses * np.sum(self.v * self.v, axis=1)

    def compute_potential_energy(self, potential):
        if self.ndim == 1:
            return potential(self.r).flatten()
        elif self.ndim == 2:
            return potential(self.r[:, 0], self.r[:, 1])

    def compute_force(self, potential):
        return -1 * potential.deriv(self.r)
