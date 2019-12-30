import numpy as np

from .hopping import Hopping
from .ring_polymer import RingPolymer


class RPSH(RingPolymer, Hopping):

    def __init__(self, coordinates, velocities, masses,
                 initial_state, potential, start_file=None,
                 equilibration_end=None,
                 nstates=2, n_beads=4, T=298):

        RingPolymer.__init__(self, coordinates, velocities, masses, n_beads, T)

        self.state = initial_state
        self.initial_r = coordinates
        self.initial_v = velocities
        self.potential = potential
        self.start_file = start_file
        self.equilibration_end = equilibration_end
        self.nstates = nstates

        self.coeffs = None
        self.centroid_U = None

        if start_file is not None and equilibration_end is not None:
            self.set_position_from_trajectory(start_file, equilibration_end)

        self.density_matrix = np.zeros((self.nstates,
                                        self.nstates))
        self.density_matrix[self.state, self.state] = 1.0

        self.v += self.initial_v

    def update_electronics(self, potential):
        """Update all the electronic quantities.

        Whenever the position is updated this function should be called to give
        consistent electronics.
        """
        self.V = self.construct_V_matrix(self.r, potential)
        self.D = self.construct_Nabla_matrix(self.r, potential)
        energies, self.coeffs = self.compute_coeffs()
        self.compute_force()

        self.centroid_r = np.array([np.mean(self.r)])
        self.centroid_v = np.array([np.mean(self.v)])

        self.centroid_potential = self.construct_V_matrix(self.centroid_r,
                                                          potential)
        self.centroid_deriv = self.construct_Nabla_matrix(self.centroid_r,
                                                          potential)
        self.diagonalise_centroid()

        self.compute_hamiltonian(self.centroid_U, self.centroid_potential)
        self.compute_derivative_coupling(self.centroid_U, self.centroid_deriv)

    def compute_coeffs(self):
        """Compute the eigenvalues and eigenstates of the V matrix.

        These are used as a basis for the calculation of the rest of the
        electronic properties.
        """
        energies = np.zeros((self.n_beads, self.nstates))
        coeffs = np.zeros((self.n_beads, self.nstates, self.nstates))

        V = self.V
        n = self.n_beads
        for i in range(n):
            mat = [[V[i, i],   V[i, i+n]],
                   [V[i+n, i], V[i+n, i+n]]]
            w, v = np.linalg.eigh(mat)
            energies[i] = w
            coeffs[i] = v

        u11 = np.diag(coeffs[:, 0, 0])
        u12 = np.diag(coeffs[:, 0, 1])
        u21 = np.diag(coeffs[:, 1, 0])
        u22 = np.diag(coeffs[:, 1, 1])
        U = np.block([[u11, u12], [u21, u22]])

        if self.coeffs is not None:
            column_sum = np.sum(U * self.coeffs, axis=1)
            U = np.where(column_sum < 0, -U, U)

        return energies, U

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

    def compute_force(self):
        """Compute <psi_i|dH/dR|psi_j>"""
        force_matrix = -self.coeffs.T @ self.D @ self.coeffs
        self.force = force_matrix.diagonal()

    def compute_acceleration(self, potential):
        """Evaluate electronics and return force."""
        self.update_electronics(potential)
        force = np.split(self.force, 2)[self.state]
        self.acceleration = force.reshape(self.n_particles, self.n_beads,
                                          self.ndim) / self.masses
