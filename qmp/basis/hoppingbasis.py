from qmp.basis.basis import basis
import numpy as np
import math


class Hopping(basis):

    def __init__(self, position, momentum, mass,
                 initial_state, potential, nstates=2):
        basis.__init__(self)

        velocity = momentum / mass

        self.mass = mass
        self.potential = potential
        self.nstates = nstates

        self.reset_system(position, velocity, initial_state)

    def reset_system(self, position, velocity, initial_state):
        self.x = position
        self.last_velocity = velocity
        self.velocity = velocity
        self.current_state = initial_state

        self.update_electronics(position)

        self.density_matrix = np.zeros((self.nstates, self.nstates))
        self.density_matrix[self.current_state, self.current_state] = 1.0

    def construct_V_matrix(self):
        flat = np.array(
            [self.potential(self.x, n=i) for i in range(self.nstates**2)]
            )
        V = flat.reshape((self.nstates, self.nstates))
        return V

    def construct_Nabla_matrix(self):
        flat = np.array(
            [self.potential.deriv(self.x, n=i) for i in range(self.nstates**2)]
            )
        D = flat.reshape((self.nstates, self.nstates))
        return D

    def compute_coeffs(self):
        return np.linalg.eigh(self.V)

    def compute_force(self):
        force_matrix = -np.linalg.multi_dot([self.coeffs.T.conj(),
                                             self.D, self.coeffs])
        return np.diag(force_matrix)
        # Below also works and is faster for bigger matrices I prefer the above
        # more readable version.

        # half = np.einsum("ij,jp->ip", self.D, self.coeffs)
        # out = np.zeros(2)
        # for i in range(2):
        #     out[i] += -np.einsum('i,i', self.coeffs[:,i], half[:,i])
        # return out

    def compute_hamiltonian(self):
        return np.linalg.multi_dot([self.coeffs.T, self.V, self.coeffs])

    def compute_derivative_coupling(self):
        out = np.linalg.multi_dot([self.coeffs.T, self.D, self.coeffs])

        # I do not know the purpose of this section but will steal it from the
        # other code for now. Though I suppose realistically I've actually
        # stolen basically all of this, it's about understanding though, that's
        # the important part, right?
        for j in range(self.nstates):
            for i in range(j):
                dE = self.energies[j] - self.energies[i]
                if abs(dE) < 1.0e-14:
                    dE = math.copysign(1.0e-14, dE)

                out[i, j] /= dE
                out[j, i] /= -dE

        return out

    def compute_propagating_hamiltonian(self):
        velocity = (self.last_velocity + self.velocity) * 0.5
        return self.hamiltonian - 1.0j * self.derivative_coupling * velocity

    def update_electronics(self, position):
        self.x = position

        self.V = self.construct_V_matrix()
        self.D = self.construct_Nabla_matrix()
        self.energies, self.coeffs = self.compute_coeffs()

        self.force = self.compute_force()
        self.hamiltonian = self.compute_hamiltonian()
        self.derivative_coupling = self.compute_derivative_coupling()
