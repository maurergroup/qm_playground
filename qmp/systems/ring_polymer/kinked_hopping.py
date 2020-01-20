import numpy as np

from ..hopping import Hopping
from .nonadiabatic_ring_polymer import (NonadiabaticRingPolymer,
                                        get_nonadiabatic_coupling,
                                        get_overlap_integral)


class KinkedHopping(NonadiabaticRingPolymer, Hopping):

    def __init__(self, coordinates, velocities, masses, initial_state,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses,
                         start_file, equilibration_end,
                         n_beads, T, n_states)

        self.state = initial_state

        self.centroid_U = None
        self.density_matrix = np.zeros((self.n_states,
                                        self.n_states), dtype=complex)
        self.density_matrix[self.state, self.state] = 1.0

        self.nonadiabatic_coupling = np.zeros((self.n_particles, self.n_beads,
                                               self.ndim, self.n_states,
                                               self.n_states),
                                              dtype=complex)
        self.overlap_integrals = np.zeros((self.n_particles, self.n_beads),
                                          dtype=complex)
        self.state_occupations = np.full((self.n_particles, self.n_beads),
                                         self.state, dtype=int)
        self.adiabatic_V_prime = np.zeros((self.n_particles, self.n_beads,
                                           self.ndim,
                                           self.n_states, self.n_states),
                                          dtype=complex)
        self.density_prop = np.zeros((self.n_particles, self.n_beads,
                                      self.ndim, self.n_states, self.n_states),
                                     dtype=complex)

    def compute_acceleration(self, potential):
        self._update_electronics(potential)
        kink = self._compute_kink_force(potential)
        bead = self._compute_bead_force()
        self.acceleration = (kink + bead) / self.masses

    def _update_electronics(self, potential):

        self.V_matrix = self.compute_V_matrix(self.r, potential)
        self.compute_V_prime_matrix(potential)
        self.diagonalise_V()
        self._compute_adiabatic_force()
        self._compute_nonadiabatic_coupling()
        self.overlap_integrals = self._compute_overlap_integrals()
        self._compute_density_propagator()

        self.centroid_r = np.array([np.mean(self.r)])
        self.centroid_v = np.array([np.mean(self.v)])

        self.centroid_potential = self.construct_V_matrix(self.centroid_r,
                                                          potential)
        self.centroid_deriv = self.construct_Nabla_matrix(self.centroid_r,
                                                          potential)
        self.diagonalise_centroid()

        self.compute_hamiltonian(self.centroid_U, self.centroid_potential)
        self.compute_derivative_coupling(self.centroid_U, self.centroid_deriv)

    def diagonalise_centroid(self):
        E, U = np.linalg.eigh(self.centroid_potential)

        if self.centroid_U is not None:
            column_sum = np.sum(U * self.centroid_U, axis=1)
            U = np.where(column_sum < 0, -U, U)

        self.centroid_U = U
        self.energies = E

    def _compute_adiabatic_force(self):
        F = (np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3))
             @ self.V_prime_matrix
             @ self.S_matrix[:, :, None])
        self.adiabatic_V_prime = F

    def _compute_nonadiabatic_coupling(self):
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    d = np.zeros((self.n_states, self.n_states), dtype=complex)
                    for m in range(self.n_states):
                        for n in range(m+1, self.n_states):
                            eigs = np.diag(self.lambdas[i, j])
                            state1 = self.S_matrix[i, j, :, m]
                            state2 = self.S_matrix[i, j, :, n]
                            derivative = self.V_prime_matrix[i, j, k]
                            e1 = eigs[m]
                            e2 = eigs[n]
                            d[m, n] = get_nonadiabatic_coupling(state1, state2, derivative, e1, e2)
                            d[n, m] = get_nonadiabatic_coupling(state2, state1, derivative, e2, e1)
                    self.nonadiabatic_coupling[i, j, k] = d

    def _compute_overlap_integrals(self, state_occupations=None):
        if state_occupations is None:
            state_occupations = self.state_occupations
        overlap_integrals = np.zeros_like(self.overlap_integrals)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                adj_j = (j+1) % self.n_beads
                adj_n = state_occupations[i, adj_j]
                n = state_occupations[i, j]
                s = get_overlap_integral(self.S_matrix[i, j, :, n],
                                         self.S_matrix[i, adj_j, :, adj_n])
                overlap_integrals[i, j] = s
        return overlap_integrals

    def _get_finite_difference_quantities(self, i, r, potential):
        step = np.zeros_like(self.r)
        step[:, :, i] = r

        v_plus = self.compute_V_matrix(self.r+step, potential)
        s_plus, lam_plus = self.diagonalise_matrix(v_plus)
        return s_plus

    def _compute_density_propagator(self):
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                V = self.lambdas[i, j]
                for k in range(self.ndim):
                    d = self.nonadiabatic_coupling[i, j, k]
                    v = self.v[i, j, k]
                    self.density_prop[i, j, k] = V - 1j*d*v

    def _compute_kink_force(self, potential):
        F = np.zeros((self.n_particles, self.n_beads, self.ndim))
        s_plus = np.zeros((self.ndim, self.n_particles, self.n_beads,
                           self.n_states, self.n_states), dtype=complex)
        s_minus = np.zeros((self.ndim, self.n_particles, self.n_beads,
                           self.n_states, self.n_states), dtype=complex)
        s_plusplus = np.zeros((self.ndim, self.n_particles, self.n_beads,
                           self.n_states, self.n_states), dtype=complex)
        s_minusminus = np.zeros((self.ndim, self.n_particles, self.n_beads,
                           self.n_states, self.n_states), dtype=complex)
        h = 0.001

        for k in range(self.ndim):
            s_plus[k] = self._get_finite_difference_quantities(k, h, potential)
            s_minus[k] = self._get_finite_difference_quantities(k, -h, potential)
            s_minusminus[k] = self._get_finite_difference_quantities(k, -2*h, potential)
            s_plusplus[k] = self._get_finite_difference_quantities(k, 2*h, potential)
        deriv = (s_minusminus/12 - 2/3*s_minus + 2/3*s_plus -s_plusplus/12)/h

        for i in range(self.n_particles):
            for j in range(self.n_beads):
                s_back = self.overlap_integrals[i, j-1]
                s_forward = self.overlap_integrals[i, j]
                # if s_back == 0:
                #     s_back = 1e-17
                # if s_forward == 0:
                #     s_forward = 1e-17
                # if np.abs(s_back) < 1e-15:
                #     s_back = np.copysign(1e-15, np.real(s_back))
                # if np.abs(s_forward) < 1e-15:
                #     s_forward = np.copysign(1e-15, np.real(s_forward))
                # print(s_back, s_forward)

                n = self.state_occupations[i, j]
                n1 = self.state_occupations[i, j-1]
                forward_j = (j+1) % self.n_beads
                n2 = self.state_occupations[i, forward_j]

                for k in range(self.ndim):
                    s1 = get_overlap_integral(self.S_matrix[i, j-1, :, n1],
                                              deriv[k, i, j, :, n])
                    s2 = get_overlap_integral(deriv[k, i, j, :, n],
                                              self.S_matrix[i, forward_j, :, n2])
                    F[i, j, k] = np.real(s1/s_back + s2/s_forward)
        F /= self.beta
        return F * self.n_beads

    def _compute_bead_force(self):
        diag = np.diagonal(self.adiabatic_V_prime, axis1=-2, axis2=-1)
        i, j = np.indices(self.state_occupations.shape)
        F = -np.real(diag[i, j, :, self.state_occupations[i, j]])
        return F

    def _compute_bead_potential(self, state_occupation):
        i, j = np.indices(state_occupation.shape)
        diag = np.diagonal(self.lambdas, axis1=-2, axis2=-1)
        energy = diag[i, j, state_occupation[i, j]]
        return energy

    def compute_kink_potential(self):
        return self._compute_kink_potential(self.overlap_integrals)

    def _compute_kink_potential(self, overlaps):
        kink = (np.log(np.abs(np.real(np.prod(overlaps, axis=1))))
                / self.beta)
        return -kink

    def _compute_total_energy(self, state_occupation=None):
        if state_occupation is None:
            state_occupation = self.state_occupations
        overlaps = self._compute_overlap_integrals(state_occupation)
        kink = self._compute_kink_potential(overlaps)
        bead = self._compute_bead_potential(state_occupation)
        return np.sum(kink + np.mean(bead))

    def compute_potential_energy(self, potential):
        """Compute the potential energy for each bead."""
        bead = self._compute_bead_potential(self.state_occupations)
        spring = self.compute_spring_potential()
        return bead + spring

    def get_velocity(self):
        return self.centroid_v

    def get_position(self):
        return self.centroid_r

    def execute_hopping(self, dt):
        """Carry out the hop between surfaces."""

        g = self._get_probabilities(dt)
        can_hop = self._check_possible_hop(g)
        # print()
        # if can_hop:
        # self._attempt_hop()

    def _get_probabilities(self, dt):
        """Calculate the hopping probability.

        Returns an array where prob[i] is the probability of hopping from the
        current state to state i.
        """
        A = self.density_matrix
        R = self.get_velocity()
        d = self.derivative_coupling

        probability = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                if A[i, i] == 0:
                    continue
                probability[i, j] = np.real(d[i, j] * A[j, i] / A[i, i])
        probability *= 2 * R * dt
        np.fill_diagonal(probability, 0)
        return probability.clip(0, 1)

    def _check_possible_hop(self, g):
        """Determine whether a hop should occur.

        A random number is compared with the hopping probability, if the
        probability is higher than the random number, a hop occurs. This
        currently only works for a two level system.

        Parameters
        ----------
        g : float
            The hopping probability.
        """
        zeta = np.random.rand(self.n_particles, self.n_beads)
        # dest = np.zeros((self.n_particles, self.n_beads))
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                current_state = self.state_occupations[i, j]
                for m in range(self.n_states):
                    if m != current_state:
                        if g[current_state, m] > zeta[i, j]:
                            print('hop')
                            print(m)
                            self.state_occupations[i, j] = m
                    # dest[i, j] = np.searchsorted(g[state], current) - 1

    def _attempt_hop(self):
        """Carry out a hop if the particle has sufficient kinetic energy.

        If the energy is sufficient the velocity is rescaled accordingly and
        the state changed. Otherwise nothing happens.
        """
        super()._attempt_hop()
        self.state_occupations[:, :] = self.state

    # def execute_hopping(self, dt):

    #     states = self._choose_new_states()
    #     p = self._compute_probabilities(states, dt)
    #     rand = np.random.rand()
    #     if p > rand:
    #         self.state_occupations = states

    # def _choose_new_states(self):
    #     states = np.zeros_like(self.state_occupations)

    #     for i in range(self.n_particles):
    #         index = np.random.randint(0, self.n_beads)
    #         one_change = copy.copy(self.state_occupations[i])
    #         one_change[index] ^= 1
    #         all_change = self.state_occupations[i] ^ 1

    #         choices = np.array([one_change, all_change])
    #         index = np.random.randint(0, 2)
    #         choice = choices[index]
    #         states[i] = choice
    #     return states

    # def _compute_probabilities(self, states, dt, efficiency=5):
    #     current = self._compute_total_energy()
    #     prospective = self._compute_total_energy(states)

    #     delta = current - prospective
    #     prob = np.exp(self.beta * delta)
    #     return prob * dt * efficiency
