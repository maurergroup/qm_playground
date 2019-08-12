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

    def reset_system(self):
        """ Resets all quantities to the initial state.
        This is used to refresh the system before starting a new trajectory.
        """
        self.r = copy.deepcopy(self.initial_r)
        self.last_velocity = copy.deepcopy(self.initial_v)
        self.v = copy.deepcopy(self.initial_v)
        self.current_state = copy.deepcopy(self.initial_state)

        self.update_electronics()

        self.density_matrix = np.zeros((self.nstates,
                                        self.nstates))
        self.density_matrix[self.current_state, self.current_state] = 1.0

    def construct_V_matrix(self):
        """ Constructs an n by n matrix for V evaluated at position r.
        """
        V = np.zeros((self.nstates, self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                V[i, j] = self.potential(self.r, i=i, j=j)
        return V

    def construct_Nabla_matrix(self):
        """ Constructs an n by n matrix for dV/dR from the potential.
        As with construct_V_matrix, the diabatic matrix elements should be
        given to the potential as a list of functions. This then calculates
        them for the current position and reshapes.
        """
        flat = np.array(
            [self.potential.deriv(self.r, n=i) for i in range(self.nstates**2)]
            )
        D = flat.reshape((self.nstates, self.nstates))
        return D

    def compute_coeffs(self):
        """ Computes the eigenvalues and eigenstates of the V matrix
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
        """ Computes <psi_i|dH/dR|psi_i>
        """
        force_matrix = -np.einsum('ji,jk,km->im', self.coeffs.conj(),
                                  self.D, self.coeffs)
        return np.diag(force_matrix)

    def compute_hamiltonian(self):
        """ Computes <psi|V|psi>
        """
        return np.dot(self.coeffs.T, np.dot(self.V, self.coeffs))

    def compute_derivative_coupling(self):
        """ Computes <psi|D|psi> / (V_jj - V_ii)
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
        """ Computes H - ihRd
        This is used to propagate the density matrix.
        """
        velocity = (self.last_velocity + self.v) * 0.5
        return self.hamiltonian - 1.0j * self.derivative_coupling * velocity

    def update_electronics(self):
        """ Updates all the electronic quantities.
        Whenever the position is updated this function should be called to give
        consistent electronics.
        """
        self.V = self.construct_V_matrix()
        self.D = self.construct_Nabla_matrix()
        self.energies, self.coeffs = self.compute_coeffs()

        self.force = self.compute_force()
        self.hamiltonian = self.compute_hamiltonian()
        self.derivative_coupling = self.compute_derivative_coupling()
