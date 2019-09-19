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
        electronic properties. The second part is to correct the phase.
        """
        energies, coeff = np.linalg.eigh(self.V)

        if self.coeffs is not None:
            column_sum = np.sum(coeff * self.coeffs, axis=1)
            coeff = np.where(column_sum < 0, -coeff, coeff)

        return energies, coeff

    def compute_force(self):
        """Compute <psi_i|dH/dR|psi_j>"""
        force_matrix = -self.coeffs.T @ self.D @ self.coeffs
        return np.diag(force_matrix)

    def compute_hamiltonian(self):
        """Compute <psi|V|psi>"""
        return self.coeffs.T @ self.V @ self.coeffs

    def compute_derivative_coupling(self):
        """Compute <psi|D|psi> / (V_jj - V_ii)

        This uses the Hellman-Feynman theorem.
        """
        out = self.coeffs.T @ self.D @ self.coeffs

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

        U = coeff @ prop @ coeff.T.conj()
        self.density_matrix = U @ self.density_matrix @ U.T.conj()

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
        n ^= 1

        return prob[n].clip(0, 1)

    def attempt_hop(self):
        """Carry out a hop if the particle has sufficient kinetic energy.

        If the energy is sufficient the velocity is rescaled accordingly and
        the state changed. Otherwise nothing happens.
        """
        desired_state = self.current_state ^ 1

        old = self.energies[self.current_state]
        new = self.energies[desired_state]

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
