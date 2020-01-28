import numpy as np
from scipy.linalg import expm

from .nonadiabatic_ring_polymer import (NonadiabaticRingPolymer,
                                        get_nonadiabatic_coupling,
                                        get_overlap_integral)


class KinkedHopping(NonadiabaticRingPolymer):

    def __init__(self, coordinates, velocities, masses, initial_state,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses,
                         start_file, equilibration_end,
                         n_beads, T, n_states)

        self.state = initial_state

        self.centroid_U = None
        self.density_matrix = np.zeros((self.n_particles, self.n_beads,
                                        self.n_states, self.n_states),
                                       dtype=complex)
        self.density_matrix[:, :, self.state, self.state] = 1.0

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
        self.density_prop = np.zeros((self.n_particles, self.ndim,
                                      self.n_states, self.n_states),
                                     dtype=complex)

    def compute_acceleration(self, potential):
        self._update_electronics(potential)
        self._compute_kink_force(potential)
        bead = self._compute_bead_force()
        self.acceleration = (self.kink_force + bead) / self.masses

    def _update_electronics(self, potential):

        self.V_matrix = self.compute_V_matrix(self.r, potential)
        self.compute_V_prime_matrix(potential)
        self.diagonalise_V()
        self._compute_adiabatic_force()
        self._compute_nonadiabatic_coupling()
        self.overlap_integrals = self._compute_overlap_integrals()
        self._compute_density_propagator()

    def _compute_adiabatic_force(self):
        F = (np.transpose(self.S_matrix[:, :, None], (0, 1, 2, 4, 3))
             @ self.V_prime_matrix
             @ self.S_matrix[:, :, None])
        self.adiabatic_V_prime = F

    def _compute_nonadiabatic_coupling(self):
        self.old_coupling = np.copy(self.nonadiabatic_coupling)
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
        self.density_prop = np.zeros((self.n_particles, self.n_beads,
                                      self.ndim,
                                      self.n_states, self.n_states),
                                     dtype=complex)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                V = self.lambdas[i, j]
                for k in range(self.ndim):
                    d = self.nonadiabatic_coupling[i, j, k]
                    v = self.v[i, j, k]
                    self.density_prop[i, j, k] = V - 1j*d*v
        # self.density_prop /= self.n_beads

    def propagate_density_matrix(self, dt):
        """ Propagates the density matrix by dt.

        This is taken from smparker's FSSH implementation.
        """

        new_density = np.zeros_like(self.density_matrix)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                for k in range(self.ndim):
                    U = expm(-1j * self.density_prop[i, j, k] * dt)
                    new_density[i, j] += U @ self.density_matrix[i, j] @ U.T.conj()
        self.density_matrix = new_density

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
        self.kink_force = F * self.n_beads

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
        kink = np.log(np.abs(np.prod(np.real(overlaps), axis=1)))
        return -kink / self.beta

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

    def execute_hopping(self, dt):
        """Carry out the hop between surfaces."""

        g = self._get_probabilities(dt)
        self._hop_those_beads(g)

    def _get_probabilities(self, dt):
        """Calculate the hopping probability for each bead.
        """
        probability = np.zeros((self.n_particles, self.n_beads,
                                self.n_states))
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                A = self.density_matrix[i, j]
                state = self.state_occupations[i, j]
                if A[state, state] == 0:
                    continue
                for k in range(self.ndim):
                    R = self.v[i, j, k]
                    for n in range(self.n_states):
                        if n != state:
                            d = self.nonadiabatic_coupling[i, j, k, state, n]
                            p = np.real(d * A[n, state] / A[state, state]) * R
                            probability[i, j, n] += p
        return probability.clip(0, 1) * 2 * dt

    def _hop_those_beads(self, g):
        """Hop each bead to other states if probability is big.

        A random number is compared with the hopping probability, if the
        probability is higher than the random number, a hop occurs.

        Parameters
        ----------
        g : float
            The hopping probability.
        """
        zeta = np.random.rand(self.n_particles, self.n_beads)
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                current_state = self.state_occupations[i, j]
                for m in reversed(range(self.n_states)):
                    if m != current_state:
                        if g[i, j, m] > zeta[i, j]:
                            self.state_occupations[i, j] = m
                            break
        self.state = np.mean(self.state_occupations)

    def apply_quantum_force(self, dt):

        force = self._get_quantum_force(dt)
        self.v = self.v + force * dt / self.masses

        dq = self._get_quantum_delta_q()
        self.r = self.r - dq * dt

    def _get_quantum_force(self, dt):
        force = np.zeros((self.n_particles, self.n_beads, self.ndim))
        d_dot = (self.nonadiabatic_coupling-self.old_coupling)/dt
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                A = self.density_matrix[i, j]
                state = self.state_occupations[i, j]
                for k in range(self.ndim):
                    for n in range(self.n_states):
                        if n != state:
                            force[i, j, k] += np.real(2 * np.imag(A[state, n])
                                                      * d_dot[i, j, k, state, n])
        return force

    def _get_quantum_delta_q(self):
        dq = np.zeros_like(self.r)
        d = self.nonadiabatic_coupling
        for i in range(self.n_particles):
            for j in range(self.n_beads):
                A = self.density_matrix[i, j]
                state = self.state_occupations[i, j]
                for k in range(self.ndim):
                    for n in range(self.n_states):
                        if n != state:
                            dq[i, j, k] += np.real(2 * np.imag(A[state, n])
                                                   * d[i, j, k, state, n]
                                                   / self.masses[i])
        return dq


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
