import numpy as np

from .rpmd import RPMD


class MF_RPMD(RPMD):
    """ Mean-Field RPMD as shown in Tim Hele's MChem thesis."""

    def __init__(self, coordinates, velocities, masses,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses, n_beads, T)

        self.n_states = n_states

        if start_file is not None and equilibration_end is not None:
            self.set_position_from_trajectory(start_file, equilibration_end)

        self.s_deriv = np.zeros((self.n_particles, self.n_beads, self.ndim,
                                 self.n_states, self.n_states))
        self.lam_deriv = np.zeros_like(self.s_deriv)
        self.lam_exp_deriv = np.zeros_like(self.s_deriv)

        self.D_matrix = np.zeros_like(self.s_deriv)
        self.M_matrix = np.zeros((self.n_particles, self.n_beads,
                                  self.n_states, self.n_states))
        self.F_matrix = np.zeros_like(self.M_matrix)
        self.G_matrix = np.zeros_like(self.M_matrix)
        self.H_matrix = np.zeros_like(self.M_matrix)

    def compute_V_matrix(self, r, potential):
        V = np.zeros((self.n_particles,
                      self.n_beads,
                      self.n_states,
                      self.n_states))
        dim_split_r = np.array(np.split(r, self.ndim, axis=2))
        for i in range(self.n_states):
            for j in range(self.n_states):
                V[:, :, i, j] = potential(*dim_split_r, i=i, j=j).squeeze()
        return V

    def diagonalise_matrix(self, V):
        S_matrix = np.zeros((self.n_particles, self.n_beads,
                             self.n_states, self.n_states))
        lambdas = np.zeros_like(S_matrix)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                eigval, S_matrix[i, j] = np.linalg.eigh(V[i, j])
                lambdas[i, j] = np.diag(eigval)
                if S_matrix[i, j, 0, 0] < 0:
                    S_matrix[i, j] *= -1
        return S_matrix, lambdas

    def exponentiate_lambdas(self, lambdas):
        out = np.zeros_like(lambdas)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                out[i, j] = np.diag(np.exp(-self.beta*np.diag(lambdas[i, j])))
        return out

    def set_V_matrix(self, potential):
        self.V_matrix = self.compute_V_matrix(self.r, potential)

    def diagonalise_at_current_position(self):
        self.S_matrix, self.lambdas = self.diagonalise_matrix(self.V_matrix)

    def set_exp_lambdas(self):
        self.exp_lambdas = self.exponentiate_lambdas(self.lambdas)

    def calculate_derivatives(self, potential, h=0.001):

        for i in range(self.ndim):

            s_plus, lam_plus, exp_plus = (
                self.get_finite_difference_quantities(i, 0.5*h, potential))
            s_minus, lam_minus, exp_minus = (
                self.get_finite_difference_quantities(i, -0.5*h, potential))

            self.s_deriv[:, :, i] = 0.5*(s_plus-s_minus)/h
            self.lam_deriv[:, :, i] = 0.5*(lam_plus-lam_minus)/h
            self.lam_exp_deriv[:, :, i] = 0.5*(exp_plus-exp_minus)/h

    def get_finite_difference_quantities(self, i, r, potential):
        step = np.zeros_like(self.r)
        step[:, :, i] = r

        v_plus = self.compute_V_matrix(self.r+step, potential)
        s_plus, lam_plus = self.diagonalise_matrix(v_plus)
        exp_plus = self.exponentiate_lambdas(lam_plus)
        return s_plus, lam_plus, exp_plus

    def calculate_D_and_M_matrices(self):
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                s = self.S_matrix[i, j]
                exp = self.exp_lambdas[i, j]
                self.M_matrix[i, j] = s.T@exp@s

                for k in range(self.ndim):
                    s_dq = self.s_deriv[i, j, k]
                    exp_dq = self.lam_exp_deriv[i, j, k]
                    self.D_matrix[i, j, k] = s_dq.T@exp@s + s.T@exp_dq@s + s.T@exp@s_dq

    def calculate_F_matrices(self):
        self.F_matrix[:, 0] = self.M_matrix[:, 0]
        for i in range(self.n_particles):
            for j in range(1, self.n_beads):
                self.F_matrix[i, j] = self.F_matrix[i, j-1] @ self.M_matrix[i, j]

    def calculate_G_matrices(self):
        self.G_matrix[:, -1] = self.M_matrix[:, -1]
        for i in range(self.n_particles):
            for j in range(self.n_beads-2, 0, -1):
                self.G_matrix[i, j] = self.M_matrix[:, j] @ self.G_matrix[i, j+1]
        self.G_matrix[:, 0] = self.F_matrix[:, -1]

    def calculate_hole_matrices(self):
        self.H_matrix[:, 0] = self.G_matrix[:, 1]
        self.H_matrix[:, -1] = self.F_matrix[:, -2]
        for i in range(self.n_particles):
            for j in range(1, self.n_beads-1):
                self.H_matrix[i, j] = self.G_matrix[i, j+1] @ self.F_matrix[i, j-1]

    def compute_force(self):
        force = np.zeros((self.n_particles, self.n_beads, self.ndim))
        for i in range(self.n_particles):
            denominator = np.trace(self.F_matrix[i, -1])
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    numerator = np.trace(self.D_matrix[i, j, k] @ self.H_matrix[i, j])
                    force[i, j, k] = numerator / denominator / self.beta
        return force

    def compute_acceleration(self, potential):
        self.set_V_matrix(potential)
        self.diagonalise_at_current_position()
        self.set_exp_lambdas()

        self.calculate_derivatives(potential)
        self.calculate_D_and_M_matrices()
        self.calculate_F_matrices()
        self.calculate_G_matrices()
        self.calculate_hole_matrices()

        F = self.compute_force()
        self.acceleration = F / self.masses[:, np.newaxis, np.newaxis]

    def compute_bead_potential_energy(self, potential):
        self.set_V_matrix(potential)
        self.diagonalise_at_current_position()
        self.set_exp_lambdas()

        self.calculate_derivatives(potential)
        self.calculate_D_and_M_matrices()
        self.calculate_F_matrices()
        pot = np.zeros((self.n_particles))
        for i in range(self.n_particles):
            pot[i] = -np.log(np.trace(self.F_matrix[i, -1])) / self.beta / self.n_beads
        return pot
