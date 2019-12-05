import numpy as np

from .rpmd import RPMD


class NRPMD(RPMD):

    def __init__(self, coordinates, velocities, masses, initial_state,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2, seed=None):

        super().__init__(coordinates, velocities, masses, n_beads, T)

        self.n_states = n_states
        self.initialise_mapping_variables(initial_state, seed)

        if start_file is not None and equilibration_end is not None:
            self.set_position_from_trajectory(start_file, equilibration_end)


    def initialise_mapping_variables(self, initial_state, seed=None):
        if seed is not None:
            np.random.seed(seed)
        theta = np.random.rand(self.n_particles, self.n_beads,
                               self.n_states) * 2 * np.pi

        self.q_map = np.sqrt(1 / (np.tan(-theta)**2 + 1))
        self.q_map[:, :, initial_state] *= np.sqrt(3)
        self.p_map = self.q_map * np.tan(-theta)

    def compute_V_matrix(self, potential):
        V = np.zeros((self.n_particles,
                      self.n_beads,
                      self.n_states, self.n_states))
        dim_split_r = np.array(np.split(self.r, self.ndim, axis=2))
        for i in range(self.n_states):
            for j in range(self.n_states):
                V[:, :, i, j] = potential(*dim_split_r, i=i, j=j).squeeze()
        self.V_matrix = V

    def diagonalise_V(self):
        S_matrix = np.zeros((self.n_particles, self.n_beads, self.n_states,
                             self.n_states))
        lambdas = np.zeros_like(S_matrix)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                eigval, S_matrix[i, j] = np.linalg.eigh(self.V_matrix[i, j])
                lambdas[i, j] = np.diag(eigval)
        self.S_matrix = S_matrix
        self.lambdas = lambdas

    def compute_propagators(self, dt):
        C_matrix = np.zeros((self.n_particles, self.n_beads, self.n_states,
                             self.n_states))
        D_matrix = np.zeros_like(C_matrix)

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                cos = np.diag(np.cos(np.diag(self.lambdas[i, j]) * dt))
                sin = np.diag(np.sin(-np.diag(self.lambdas[i, j]) * dt))
                C_matrix[i, j] = self.S_matrix[i, j] @ cos @ self.S_matrix[i, j].T
                D_matrix[i, j] = self.S_matrix[i, j] @ sin @ self.S_matrix[i, j].T

        self.C_matrix = C_matrix
        self.D_matrix = D_matrix

    def compute_V_prime_matrix(self, potential):
        V = np.zeros((self.n_particles,
                      self.n_beads, self.ndim,
                      self.n_states, self.n_states))
        for i in range(self.n_particles):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    V[i, :, :, j, k] = potential.deriv(self.r[i], i=j, j=k)
        self.V_prime_matrix = V

    def compute_adiabatic_derivative(self):
        dv = self.V_prime_matrix
        W = np.zeros((self.n_particles, self.n_beads, self.ndim,
                      self.n_states, self.n_states))

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    W[i, j, k] = self.S_matrix[i, j].T @ dv[i, j, k] @ self.S_matrix[i, j]

        self.W_matrix = W

    def compute_gamma_and_epsilon(self, dt):
        W = self.W_matrix
        gamma = np.zeros_like(W)
        epsilon = np.zeros_like(W)

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    for n in range(self.n_states):
                        for m in range(n+1, self.n_states):
                            lam = self.lambdas[i, j, m, m] - self.lambdas[i, j, n, n]
                            g = self.compute_gamma_element(lam, dt, W[i, j, k, m, n])
                            e = self.compute_epsilon_element(lam, dt, W[i, j, k, m, n])
                            gamma[i, j, k, m, n] = g
                            epsilon[i, j, k, m, n] = e

                    gamma[i, j, k] = np.tril(gamma[i, j, k], -1) + np.triu(gamma[i, j, k].T, 1)
                    gamma[i, j, k] += np.diag(np.diag(W[i, j, k]) * dt)
                    epsilon[i, j, k] = np.tril(epsilon[i, j, k], -1) - np.triu(epsilon[i, j, k].T, 1)

        self.gamma = gamma
        self.epsilon = epsilon

    @staticmethod
    def compute_gamma_element(lam, dt, W):
        return 1 / lam * np.sin(lam * dt) * W

    @staticmethod
    def compute_epsilon_element(lam, dt, W):
        return 1 / lam * (1 - np.cos(lam * dt)) * W

    def rotate_matrices(self):
        self.E = np.zeros_like(self.gamma)
        self.F = np.zeros_like(self.epsilon)

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    S = self.S_matrix[i, j]
                    gamma = self.gamma[i, j, k]
                    epsilon = self.epsilon[i, j, k]
                    self.E[i, j, k] = S @ gamma @ S.T
                    self.F[i, j, k] = S @ epsilon @ S.T

    def propagate_mapping_variables(self):
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                q = self.C_matrix[i, j] @ self.q_map[i, j]
                q = q - self.D_matrix[i, j] @ self.p_map[i, j]
                p = self.C_matrix[i, j] @ self.p_map[i, j]
                p = p + self.D_matrix[i, j] @ self.q_map[i, j]
                self.q_map[i, j] = q
                self.p_map[i, j] = p

    def propagate_bead_velocities(self, dt):
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    q = self.q_map[i, j]
                    p = self.p_map[i, j]
                    E = self.E[i, j, k]
                    F = self.F[i, j, k]
                    dv = self.V_prime_matrix[i, j, k]
                    trV = np.trace(dv)
                    self.v[i, j, k] = self.v[i, j, k] - (0.5
                        * (q.T@E@q + p.T@E@p - 2*q.T@F@p - trV*dt)
                        / self.masses[i])

    def calculate_total_electronic_probability(self):
        self.total_prob = 0.5 * np.sum(self.q_map**2 + self.p_map**2 - 1) / self.n_beads

    def calculate_state_probability(self):
        self.state_prob = 0.5 * np.sum(self.q_map**2 + self.p_map**2 - 1, axis=1) / self.n_beads

    def compute_bead_potential_energy(self, potential):
        self.compute_V_matrix(potential)
        energy = np.zeros((self.n_particles, self.n_beads))

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                q = self.q_map[i, j]
                p = self.p_map[i, j]
                V = self.V_matrix[i, j]
                trV = np.trace(V)
                energy[i, j] = energy[i, j] + 0.5 * (q.T@V@q + p.T@V@p - trV)
        return energy
