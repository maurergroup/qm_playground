import numpy as np
from qmp.systems.phasespace import PhaseSpace
import copy


class Hopping(PhaseSpace):

    def __init__(self, coordinates, velocities, masses,
                 initial_state, potential, nstates=2):
        PhaseSpace.__init__(self, coordinates, velocities, masses)

        self.potential = potential
        self.nstates = nstates

        self.initial_r = coordinates
        self.initial_v = velocities
        self.initial_state = initial_state
        self.coeffs = None

    def reset_system(self, potential):
        """ Resets all quantities to the initial state.

        This is used to refresh the system before starting a new trajectory.
        """
        self.r = copy.deepcopy(self.initial_r)
        self.v = copy.deepcopy(self.initial_v)
        self.current_state = copy.deepcopy(self.initial_state)

        self.update_electronics(potential)

        self.density_matrix = np.zeros((self.nstates,
                                        self.nstates))
        self.density_matrix[self.current_state, self.current_state] = 1.0

    def construct_V_matrix(self, potential):
        """Construct an n by n matrix for V evaluated at position r."""
        V = np.zeros((self.nstates, self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                V[i, j] = potential(self.r, i=i, j=j)
        return V

    def construct_Nabla_matrix(self, potential):
        """Construct an n by n matrix for dV/dR from the potential.

        As with construct_V_matrix, the diabatic matrix elements should be
        given to the potential as a list of functions. This then calculates
        them for the current position and reshapes.
        """
        flat = np.array(
            [potential.deriv(self.r, n=i) for i in range(self.nstates**2)]
            )
        D = flat.reshape((self.nstates, self.nstates))
        return D

    def compute_coeffs(self):
        """Compute the eigenvalues and eigenstates of the V matrix.

        These are used as a basis for the calculation of the rest of the
        electronic properties.
        """
        energies, coeff = np.linalg.eigh(self.V)

        if self.coeffs is not None:
            for mo in range(self.nstates):
                if (np.dot(coeff[:, mo], self.coeffs[:, mo]) < 0.0):
                    coeff[:, mo] *= -1.0

        return energies, coeff

    def compute_force(self):
        """Compute <psi_i|dH/dR|psi_j>"""
        force_matrix = -np.einsum('ji,jk,km->im', self.coeffs.conj(),
                                  self.D, self.coeffs)
        return np.diag(force_matrix)

    def compute_hamiltonian(self):
        """Compute <psi|V|psi>"""
        return np.dot(self.coeffs.T, np.dot(self.V, self.coeffs))

    def compute_derivative_coupling(self):
        """Compute <psi|D|psi> / (V_jj - V_ii)

        This uses the Hellman-Feynman theorem.
        """
        out = np.einsum('ji,jk,km->im', self.coeffs, self.D, self.coeffs)

        for j in range(self.nstates):
            for i in range(j):
                dE = self.energies[j]-self.energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i, j] /= dE
                out[j, i] /= -dE

        return out

    def compute_propagating_hamiltonian(self):
        """ Compute H - ihRd

        This is used to propagate the density matrix.
        """
        return self.hamiltonian - 1.0j * self.derivative_coupling * self.v

    def update_electronics(self, potential):
        """Update all the electronic quantities.

        Whenever the position is updated this function should be called to give
        consistent electronics.
        """
        self.V = self.construct_V_matrix(potential)
        self.D = self.construct_Nabla_matrix(potential)
        self.energies, self.coeffs = self.compute_coeffs()

        self.force = self.compute_force()
        self.hamiltonian = self.compute_hamiltonian()
        self.derivative_coupling = self.compute_derivative_coupling()

    def compute_acceleration(self, potential):
        """Evaluate electronics and return force."""
        self.update_electronics(potential)
        return self.force[self.current_state] / self.masses

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
        R = self.v
        d = self.derivative_coupling
        n = self.current_state

        prob = 2 * np.real(A[n] * R * d[n] / A[n, n]) * dt

        prob[n] = 0.0
        prob = prob.clip(0.0, 1.0)

        return prob

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
        kinetic = 0.5 * self.masses * self.v ** 2

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
        b = 2.0 * np.dot(self.masses * self.v, Md)
        c = -2.0 * self.masses * -deltaV
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.v += np.real(scal) * direction

    def has_reflected(self):
        return self.r < self.potential.cell[0][0]

    def has_transmitted(self):
        return self.r > self.potential.cell[0][1]
