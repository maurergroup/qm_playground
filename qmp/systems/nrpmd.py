import numpy as np

from .nonadiabatic_ring_polymer import NonadiabaticRingPolymer


class NRPMD(NonadiabaticRingPolymer):

    def __init__(self, coordinates, velocities, masses, initial_state,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses, start_file,
                         equilibration_end, n_beads, T, n_states)

        self.initial_state = initial_state

        self.initialise_mapping_variables()

        self.gamma = np.zeros((self.n_particles, self.n_beads, self.ndim,
                               self.n_states, self.n_states), dtype=complex)
        self.epsilon = np.zeros_like(self.gamma, dtype=complex)

    def reinitialise(self):
        super().reinitialise()
        self.initialise_mapping_variables()

    def initialise_mapping_variables(self):
        theta = np.random.rand(self.n_particles, self.n_beads,
                               self.n_states) * 2 * np.pi

        self.q_map = np.sqrt(1 / (np.tan(-theta)**2 + 1))
        self.q_map[:, :, self.initial_state] *= np.sqrt(3)
        self.p_map = self.q_map * np.tan(-theta)

    def compute_propagators(self, dt):
        _, _, n, m = np.indices(self.lambdas.shape)

        cos = np.zeros((self.n_particles, self.n_beads,
                        self.n_states, self.n_states), dtype=complex)
        cos[n == m] = np.cos(self.lambdas[n == m] * dt)

        sin = np.zeros_like(cos)
        sin[n == m] = -np.sin(self.lambdas[n == m] * dt)

        C = (self.S_matrix @ cos @ np.transpose(self.S_matrix, (0, 1, 3, 2)))
        D = (self.S_matrix @ sin @ np.transpose(self.S_matrix, (0, 1, 3, 2)))

        self.C_matrix = C
        self.D_matrix = D

    def compute_adiabatic_derivative(self):
        W = (np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3))
             @ self.V_prime_matrix
             @ self.S_matrix[:, :, None])
        self.W_matrix = W

    def compute_gamma_and_epsilon(self, dt):
        W = self.W_matrix

        i, j, n, m = np.indices(self.lambdas.shape)
        _, _, _, s, t = np.indices(self.W_matrix.shape)

        lambdas = self.lambdas[i, j, m, m] - self.lambdas[i, j, n, n]

        self.gamma[s != t] = self.compute_gamma_element(lambdas[n != m], dt, W[s != t])
        self.gamma[s == t] = W[s == t] * dt
        self.epsilon[s != t] = self.compute_epsilon_element(lambdas[n != m], dt, W[s != t])

    @staticmethod
    def compute_gamma_element(lam, dt, W):
        return 1 / lam * np.sin(lam * dt) * W

    @staticmethod
    def compute_epsilon_element(lam, dt, W):
        return 1 / lam * (1 - np.cos(lam * dt)) * W

    def rotate_matrices(self):
        self.E = np.zeros_like(self.gamma)
        self.F = np.zeros_like(self.epsilon)

        self.E = (self.S_matrix[:, :, None]
                  @ self.gamma
                  @ np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3)))
        self.F = (self.S_matrix[:, :, None]
                  @ self.epsilon
                  @ np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3)))

    def propagate_mapping_variables(self):
        q = np.real(np.einsum('ijkl,ijl->ijk', self.C_matrix, self.q_map)
                    - np.einsum('ijkl,ijl->ijk', self.D_matrix, self.p_map))
        p = np.real(np.einsum('ijkl,ijl->ijk', self.C_matrix, self.p_map)
                    + np.einsum('ijkl,ijl->ijk', self.D_matrix, self.q_map))

        self.q_map = q
        self.p_map = p

    def propagate_bead_velocities(self, dt):

        qEq = np.einsum('ijl,ijklm,ijm->ijk', self.q_map, self.E, self.q_map)
        pEp = np.einsum('ijl,ijklm,ijm->ijk', self.p_map, self.E, self.p_map)
        qFp = np.einsum('ijl,ijklm,ijm->ijk', self.q_map, self.F, self.p_map)
        traceV = np.einsum('ijkll', self.V_prime_matrix)

        force = 0.5 * np.real(qEq + pEp - 2*qFp - traceV*dt)
        self.v = self.v - force / self.masses

    def calculate_total_electronic_probability(self):
        self.total_prob = 0.5 * np.sum(self.q_map**2 + self.p_map**2 - 1) / self.n_beads

    def calculate_state_probability(self):
        self.state_prob = 0.5 * np.sum(self.q_map**2 + self.p_map**2 - 1, axis=1) / self.n_beads

    def _compute_bead_potential_energy(self, potential):
        qVq = np.einsum('ijl,ijlm,ijm->ij', self.q_map, self.V_matrix, self.q_map)
        pVp = np.einsum('ijl,ijlm,ijm->ij', self.p_map, self.V_matrix, self.p_map)
        traceV = np.einsum('ijll', self.V_matrix)
        energy = 0.5 * np.real(qVq + pVp - traceV)
        return energy

    def has_reflected(self, potential):
        return np.mean(self.r) < potential.cell[0][0]

    def has_transmitted(self, potential):
        return np.mean(self.r) > potential.cell[0][1]
