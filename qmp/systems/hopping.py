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

    def reset_system(self):
        self.r = copy.deepcopy(self.initial_r)
        self.last_velocity = copy.deepcopy(self.initial_v)
        self.v = copy.deepcopy(self.initial_v)
        self.current_state = copy.deepcopy(self.initial_state)

        self.update_electronics()

        self.density_matrix = np.zeros((self.nstates,
                                        self.nstates))
        self.density_matrix[self.current_state, self.current_state] = 1.0

    def construct_V_matrix(self):
        flat = np.array(
            [self.potential(self.r, n=i) for i in range(self.nstates**2)]
            )
        V = flat.reshape((self.nstates, self.nstates))
        return V

    def construct_Nabla_matrix(self):
        flat = np.array(
            [self.potential.deriv(self.r, n=i) for i in range(self.nstates**2)]
            )
        D = flat.reshape((self.nstates, self.nstates))
        return D

    def compute_coeffs(self):
        return np.linalg.eigh(self.V)

    def compute_force(self):
        force_matrix = -np.einsum('ji,jk,km->im', self.coeffs.conj(),
                                  self.D, self.coeffs)
        return np.diag(force_matrix)

    def compute_hamiltonian(self):
        # computes <psi|V|psi>
        return np.dot(self.coeffs.T, np.dot(self.V, self.coeffs))

    def compute_derivative_coupling(self):
        # compute <psi|D|psi>
        out = np.einsum('ji,jk,km->im', self.coeffs, self.D, self.coeffs)

        # I do not know the purpose of this section but will steal it from the
        # other code for now.
        for j in range(self.nstates):
            for i in range(j):
                dE = self.energies[j]-self.energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i, j] /= dE
                out[j, i] /= -dE

        return out

    def compute_propagating_hamiltonian(self):
        velocity = (self.last_velocity + self.v) * 0.5
        return self.hamiltonian - 1.0j * self.derivative_coupling * velocity

    def update_electronics(self):
        self.V = self.construct_V_matrix()
        self.D = self.construct_Nabla_matrix()
        self.energies, self.coeffs = self.compute_coeffs()

        self.force = self.compute_force()
        self.hamiltonian = self.compute_hamiltonian()
        self.derivative_coupling = self.compute_derivative_coupling()
