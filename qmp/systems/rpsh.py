from .rpmd import RPMD
from .hopping import Hopping
import numpy as np
import scipy.sparse as sp


class RPSH(RPMD, Hopping):

    def __init__(self, coordinates, velocities, masses,
                 initial_state, potential, nstates=2, n_beads=4,
                 T=298, init_type='velocity'):

        RPMD.__init__(self, coordinates, velocities, masses, n_beads, T,
                      init_type)

        self.potential = potential
        self.nstates = nstates

        self.r = self.r[0, :, 0]
        self.v = self.v[0, :, 0]

        self.initial_r = self.r
        self.initial_v = self.v
        self.initial_state = initial_state
        self.coeffs = None
        self.centroid_U = None

    def construct_V_matrix(self, potential):
        """Construct an n by n matrix for V evaluated at position r."""
        v11 = sp.diags(potential(self.r, i=0, j=0))
        v22 = sp.diags(potential(self.r, i=1, j=1))
        v12 = sp.diags(potential(self.r, i=0, j=1))
        V = sp.bmat([[v11, v12], [v12, v22]])
        return V

    def construct_Nabla_matrix(self, potential):
        """Construct an n by n matrix for dV/dR from the potential.

        As with construct_V_matrix, the diabatic matrix elements should be
        given to the potential as a list of functions. This then calculates
        them for the current position and reshapes.
        """
        d11 = sp.diags(potential.deriv(self.r, n=0))
        d22 = sp.diags(potential.deriv(self.r, n=3))
        d12 = sp.diags(potential.deriv(self.r, n=1))
        d = sp.bmat([[d11, d12], [d12, d22]])
        return d

    def construct_centroid_potential(self, potential):
        v11 = potential(self.centroid_r, i=0, j=0)
        v22 = potential(self.centroid_r, i=1, j=1)
        v12 = potential(self.centroid_r, i=0, j=1)
        V = np.array([[v11, v12], [v12, v22]])
        self.centroid_potential = V

    def construct_centroid_derivative(self, potential):
        d11 = potential.deriv(self.centroid_r, n=0)
        d22 = potential.deriv(self.centroid_r, n=3)
        d12 = potential.deriv(self.centroid_r, n=1)
        d = np.array([[d11, d12], [d12, d22]])
        self.centroid_deriv = d

    def diagonalise_centroid(self):
        E, U = np.linalg.eigh(self.centroid_potential)

        if self.centroid_U is not None:
            column_sum = (U * self.centroid_U).sum(1)
            for i, item in enumerate(column_sum):
                if item < 0:
                    U[:, i] *= -1

        self.centroid_U = U
        self.centroid_energies = E

    def compute_coeffs(self):
        """Compute the eigenvalues and eigenstates of the V matrix.

        These are used as a basis for the calculation of the rest of the
        electronic properties.
        """
        energies = np.zeros((self.n_beads, self.nstates))
        coeffs = np.zeros((self.n_beads, self.nstates, self.nstates))

        V = self.V.A
        for i in range(self.n_beads):
            mat = [[V[i, i], V[i, i+self.n_beads]],
                   [V[i+self.n_beads, i], V[i+self.n_beads, i+self.n_beads]]]
            w, v = np.linalg.eigh(mat)
            energies[i] = w
            coeffs[i] = v

        u11 = sp.diags(coeffs[:, 0, 0])
        u12 = sp.diags(coeffs[:, 0, 1])
        u21 = sp.diags(coeffs[:, 1, 0])
        u22 = sp.diags(coeffs[:, 1, 1])
        U = sp.bmat([[u11, u12], [u21, u22]], format='csr')

        if self.coeffs is not None:
            column_sum = (U * self.coeffs).sum(1)
            for i, item in enumerate(column_sum):
                if item < 0:
                    U[:, i] *= -1

        return energies, U

    def compute_force(self):
        """Compute <psi_i|dH/dR|psi_j>"""
        force_matrix = -self.coeffs.T @ self.D @ self. coeffs
        return force_matrix.diagonal()

    def compute_hamiltonian(self):
        """Compute <psi|V|psi>"""
        return self.centroid_U.T @ self.centroid_potential @ self.centroid_U

    def compute_derivative_coupling(self):
        """Compute <psi|D|psi> / (V_jj - V_ii)

        This uses the Hellman-Feynman theorem.
        """
        out = self.centroid_U.T @ self.centroid_deriv @ self.centroid_U

        dE = self.centroid_energies[0] - self.centroid_energies[1]
        if abs(dE) < 1.0e-14:
            dE = np.copysign(1.0e-14, dE)

        out[0] /= dE
        out[1] /= -dE

        return out

    def compute_propagating_hamiltonian(self):
        """ Compute H - ihRd

        This is used to propagate the density matrix.
        """
        return self.hamiltonian - 1.0j*self.derivative_coupling*self.centroid_v

    def update_electronics(self, potential):
        """Update all the electronic quantities.

        Whenever the position is updated this function should be called to give
        consistent electronics.
        """
        self.centroid_r = np.mean(self.r)
        self.centroid_v = np.mean(self.v)

        self.V = self.construct_V_matrix(potential)
        self.D = self.construct_Nabla_matrix(potential)
        self.energies, self.coeffs = self.compute_coeffs()

        self.construct_centroid_potential(potential)
        self.construct_centroid_derivative(potential)
        self.diagonalise_centroid()

        self.force = self.compute_force()
        self.hamiltonian = self.compute_hamiltonian()
        self.derivative_coupling = self.compute_derivative_coupling()

    def compute_acceleration(self, potential):
        """Evaluate electronics and return force."""
        self.update_electronics(potential)
        force = np.split(self.force, 2)[self.current_state]
        return force / self.masses

    def propagate_density_matrix(self, dt):
        """ Propagates the density matrix by dt.

        This is taken from smparker's FSSH implementation.
        """

        V = self.compute_propagating_hamiltonian()

        diags, coeff = np.linalg.eigh(V)
        prop = np.diag(np.exp(-1j * diags * dt))

        U = np.linalg.multi_dot([coeff, prop, coeff.T.conj()])
        self.density_matrix = np.dot(U, np.dot(self.density_matrix, U.T.conj()))

    def get_probabilities(self, dt):
        """Calculate the hopping probability.

        Returns an array where prob[i] is the probability of hopping from the
        current state to state i.
        """
        A = self.density_matrix
        R = self.centroid_v
        d = self.derivative_coupling
        n = self.current_state

        prob = 2 * np.real(A[n] * R * d[n] / A[n, n]) * dt

        prob[n] = 0.0
        prob = prob.clip(0.0, 1.0)

        return prob

    def has_reflected(self):
        return self.centroid_r < self.potential.cell[0][0]

    def has_transmitted(self):
        return self.centroid_r > self.potential.cell[0][1]

    def attempt_hop(self, desired_state):
        """Carry out a hop if the particle has sufficient kinetic energy.

        If the energy is sufficient the velocity is rescaled accordingly and
        the state changed. Otherwise nothing happens.
        """
        hamiltonian = self.hamiltonian
        old = hamiltonian[self.current_state,
                          self.current_state]
        new = hamiltonian[desired_state, desired_state]
        deltaV = new - old
        kinetic = 0.5 * self.masses * self.centroid_v ** 2

        if kinetic >= deltaV:
            self.rescale_velocity(deltaV, desired_state)
            self.current_state = desired_state

    def rescale_velocity(self, deltaV, desired_state):
        """ Rescale velocity.

        Rescaling is carried out in the direction of the derivative coupling
        after a successful hop.

        Parameters
        ----------
        deltaV : float
            Difference in energy between the adiabatic states.
        desired_state : int
            The state that is being hopped to.
        """
        d = self.derivative_coupling[desired_state,
                                     self.current_state]

        direction = d / np.sqrt(np.dot(d, d))
        Md = self.masses * direction
        a = np.dot(Md, Md)
        b = 2.0 * np.dot(self.masses * self.centroid_v, Md)
        c = -2.0 * self.masses * -deltaV
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.v += np.real(scal) * direction / self.n_beads
