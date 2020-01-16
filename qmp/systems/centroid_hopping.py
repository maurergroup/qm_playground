import numpy as np

from .hopping import Hopping
from .nonadiabatic_ring_polymer import NonadiabaticRingPolymer


class CentroidHopping(NonadiabaticRingPolymer, Hopping):

    def __init__(self, coordinates, velocities, masses,
                 initial_state, potential, start_file=None,
                 equilibration_end=None,
                 n_states=2, n_beads=4, T=298):

        NonadiabaticRingPolymer.__init__(self, coordinates, velocities,
                                         masses, start_file, equilibration_end,
                                         n_beads, T, n_states)

        self.state = initial_state
        self.initial_r = coordinates
        self.initial_v = velocities
        self.potential = potential
        self.start_file = start_file
        self.equilibration_end = equilibration_end

        self.state_occupations = np.full((self.n_particles, self.n_beads),
                                         self.state, dtype=int)
        self.coeffs = None
        self.centroid_U = None

        self.density_matrix = np.zeros((self.n_states,
                                        self.n_states))
        self.density_matrix[self.state, self.state] = 1.0

        self.v += self.initial_v

    def update_electronics(self, potential):
        """Update all the electronic quantities.

        Whenever the position is updated this function should be called to give
        consistent electronics.
        """
        self.V_matrix = self.compute_V_matrix(self.r, potential)
        self.compute_V_prime_matrix(potential)
        self.diagonalise_V()
        self._compute_adiabatic_force()

        self.centroid_r = np.array([np.mean(self.r)])
        self.centroid_v = np.array([np.mean(self.v)])

        self.centroid_potential = self.construct_V_matrix(self.centroid_r,
                                                          potential)
        self.centroid_deriv = self.construct_Nabla_matrix(self.centroid_r,
                                                          potential)
        self.diagonalise_centroid()

        self.compute_hamiltonian(self.centroid_U, self.centroid_potential)
        self.compute_derivative_coupling(self.centroid_U, self.centroid_deriv)

    def diagonalise_centroid(self):
        E, U = np.linalg.eigh(self.centroid_potential)

        if self.centroid_U is not None:
            column_sum = np.sum(U * self.centroid_U, axis=1)
            U = np.where(column_sum < 0, -U, U)

        self.centroid_U = U
        self.energies = E

    def get_velocity(self):
        return self.centroid_v

    def get_position(self):
        return self.centroid_r

    def construct_V_matrix(self, r, potential):
        """Construct an n by n matrix for V evaluated at position r."""
        r = r.flatten()
        v11 = np.diag(potential(r, i=0, j=0))
        v22 = np.diag(potential(r, i=1, j=1))
        v12 = np.diag(potential(r, i=0, j=1))
        return np.block([[v11, v12], [v12, v22]])

    def construct_Nabla_matrix(self, r, potential):
        """Construct an n by n matrix for dV/dR from the potential.

        As with construct_V_matrix, the diabatic matrix elements should be
        given to the potential as a list of functions. This then calculates
        them for the current position and reshapes.
        """
        r = r.flatten()
        d11 = np.diag(potential.deriv(r, i=0, j=0))
        d22 = np.diag(potential.deriv(r, i=1, j=1))
        d12 = np.diag(potential.deriv(r, i=0, j=1))
        return np.block([[d11, d12], [d12, d22]])

    def _compute_adiabatic_force(self):
        F = (np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3))
             @ self.V_prime_matrix
             @ self.S_matrix[:, :, None])
        self.adiabatic_V_prime = F

    def compute_acceleration(self, potential):
        """Evaluate electronics and return force."""
        self.update_electronics(potential)
        self.acceleration = self._compute_bead_force()/self.masses

    def _compute_bead_force(self):
        diag = np.diagonal(self.adiabatic_V_prime, axis1=-2, axis2=-1)
        i, j = np.indices(self.state_occupations.shape)
        F = -np.real(diag[i, j, :, self.state_occupations[i, j]])
        return F

    def _attempt_hop(self):
        """Carry out a hop if the particle has sufficient kinetic energy.

        If the energy is sufficient the velocity is rescaled accordingly and
        the state changed. Otherwise nothing happens.
        """
        super()._attempt_hop()
        self.state_occupations[:, :] = self.state
