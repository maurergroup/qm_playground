from .rpmd import RPMD
from .hopping import Hopping
import numpy as np
import copy


class RPSH(Hopping, RPMD):

    def __init__(self, coordinates, velocities, masses,
                 initial_state, potential, nstates=2, n_beads=4,
                 T=298, init_type='velocity'):

        self.initial_r = coordinates
        self.initial_v = velocities
        self.initial_state = initial_state

        RPMD.__init__(self, coordinates, velocities, masses, n_beads, T,
                      init_type)

        self.potential = potential
        self.nstates = nstates

        self.r = self.r[0, :, 0]
        self.v = self.v[0, :, 0]

        self.coeffs = None
        self.centroid_U = None

    def copy_initial_values(self):
        super().copy_initial_values()

        self.initialise_beads()
        self.r = self.r[0, :, 0]
        self.v = self.v[0, :, 0]

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
