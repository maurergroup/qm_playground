import numpy as np

from qmp.systems.hopping import Hopping


class QTSH(Hopping):

    def _attempt_hop(self):
        self.state ^= 1

    def propagate_positions(self, dt):
        self.r = self.r + self.v * dt
        self.r = (self.r
                  - 2 * np.imag(self.density_matrix[0, 1])
                  * self.derivative_coupling[self.state, self.state ^ 1]
                  / self.masses
                  * dt)

    def compute_acceleration(self, potential):

        self.update_electronics(potential)

        force = np.array(np.split(self.force, 2)[self.state])
        force += self.get_quantum_force()
        self.acceleration = force / self.masses

    def get_quantum_force(self):

        d_dot = (self.derivative_coupling-self.old_coupling)/self.dt
        force = 2 * np.imag(self.density_matrix[0, 1]) * d_dot[self.state,self.state^1]
        return force

    def compute_derivative_coupling(self, coeffs, D):
        """Compute <psi|D|psi> / (V_jj - V_ii)

        This uses the Hellman-Feynman theorem.
        """
        out = coeffs.T @ D @ coeffs

        dE = self.energies[0] - self.energies[1]
        if abs(dE) < 1.0e-14:
            dE = np.copysign(1.0e-14, dE)

        out[0] /= dE
        out[1] /= -dE

        self.old_coupling = self.derivative_coupling
        self.derivative_coupling = out
