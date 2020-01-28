from abc import ABC

import numpy as np

from .ring_polymer import RingPolymer


def get_nonadiabatic_coupling(state1, state2, derivative, e1, e2):
    return state1.T.conj() @ derivative @ state2 / (e2 - e1)


def get_overlap_integral(state1, state2):
    return state1.T.conj() @ state2


class NonadiabaticRingPolymer(ABC, RingPolymer):

    def __init__(self, coordinates, velocities, masses,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses, n_beads, T,
                         start_file, equilibration_end)

        self.n_states = n_states
        self.S_matrix = None

    def compute_V_matrix(self, r, potential):
        V = np.zeros((self.n_particles, self.n_beads,
                      self.n_states, self.n_states),
                     dtype=complex)
        dim_split_r = np.array(np.split(r, self.ndim, axis=2))
        for i in range(self.n_states):
            for j in range(self.n_states):
                V[:, :, i, j] = potential(*dim_split_r, i=i, j=j).squeeze()
        return V

    def diagonalise_matrix(self, V):
        S_matrix = np.zeros((self.n_particles, self.n_beads,
                             self.n_states, self.n_states),
                            dtype=complex)
        lambdas = np.zeros_like(S_matrix, dtype=float)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                eigval, S_matrix[i, j] = np.linalg.eigh(V[i, j])
                lambdas[i, j] = np.diag(eigval)
                if self.S_matrix is not None:
                    S_matrix[i, j] = self._correct_phase(S_matrix[i, j], self.S_matrix[i, j])
        return S_matrix, lambdas

    def _correct_phase(self, mat1, mat2):
        """ Correct phase of eigenvectors
        This uses the method used in Chem. Sci., 2019, 10 8100 detailed in the
        supplementary information.
        """
        p = np.zeros((self.n_states), dtype=complex)
        overlap = get_overlap_integral(mat1, mat2)
        for i in range(self.n_states):
            for j in range(self.n_states):
                if np.abs(overlap[i, j]) > 0.5:
                    p[i] = np.sign(np.max(np.abs(overlap[i, j])))*np.sign(overlap[i, j])
        if np.any(p == 0):
            raise ValueError('Error evaluating the phase vector.')
        return p * mat1

    def diagonalise_V(self):
        self.S_matrix, self.lambdas = self.diagonalise_matrix(self.V_matrix)

    def compute_V_prime_matrix(self, potential):
        V = np.zeros((self.n_particles,
                      self.n_beads, self.ndim,
                      self.n_states, self.n_states),
                     dtype=complex)
        for i in range(self.n_particles):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    V[i, :, :, j, k] = potential.deriv(self.r[i], i=j, j=k)
        self.V_prime_matrix = V
