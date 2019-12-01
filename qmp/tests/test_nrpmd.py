import unittest

import numpy as np
import numpy.testing as test

from qmp.potential import Potential
from qmp.systems.nrpmd import NRPMD


class NRPMDTestCase(unittest.TestCase):

    def setUp(self):
        # 2D testing
        # cell = [[0, 20], [0, 20]]
        # coords = [[1, 4]]
        # velocities = [[0, 0]]
        # masses = [1860]
        # 1D testing
        cell = [[0, 20]]
        coords = [[10]]
        velocities = [[0]]
        masses = [1860]
        n_beads = 4
        T = 300
        self.initial_state = 1
        n_states = 2
        self.system = NRPMD(coords, velocities, masses, n_beads=n_beads, T=T,
                            initial_state=self.initial_state, n_states=n_states, seed=10)

        # def f(a, b):
        #     return a * b
        # def f2(a, b):
        #     return 2 * a * b
        # def off(a, b):
        #     return 0.1 * a
        def f(a):
            return 2*a
        def f2(a):
            return -2*a
        def off(a):
            return np.full_like(a, -0.1)

        self.potential = Potential(cell, f=[f, off, off, f2], n=2)

    def test_initialise_mapping_variables(self):
        self.initial_state = 1
        self.system.initialise_mapping_variables(self.initial_state)
        self.system.calculate_total_electronic_probability()

        # Check total probability is 1 for each bead.
        test.assert_array_almost_equal(self.system.total_prob, 1)

        # Check that initial state has probability of 1 and other state has
        # probability of 0.
        a = 0.5*(self.system.q_map[0,0,0]**2+self.system.p_map[0,0,0]**2-1)
        b = 0.5*(self.system.q_map[0,0,1]**2+self.system.p_map[0,0,1]**2-1)
        test.assert_array_almost_equal(a, 0)
        test.assert_array_almost_equal(b, 1)

    def test_compute_V_matrix(self):
        self.system.compute_V_matrix(self.potential)
        v_size = self.system.V_matrix.shape
        test.assert_array_almost_equal((1, 4, 2, 2), v_size)

    def test_compute_V_prime_matrix(self):
        self.system.compute_V_prime_matrix(self.potential)
        v_size = self.system.V_prime_matrix.shape
        test.assert_array_almost_equal((1, 4, 1, 2, 2), v_size)

    def test_diagonalise_V(self):
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()

        # Check transformation diagonalises
        diag = self.system.S_matrix[0, 0].T @ self.system.V_matrix[0, 0] @ self.system.S_matrix[0, 0]
        test.assert_array_almost_equal(self.system.lambdas[0, 0], diag)
        # Check the eigenvalue matrix is diagonal but of course it is because
        # that's how I made it
        test.assert_array_almost_equal(np.diag(np.diag(self.system.lambdas[0, 0])),
                                       self.system.lambdas[0, 0])

    def test_compute_propagators(self):
        dt = 1
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_propagators(dt)
        # This must work for the propagation to be correct

    def test_propagate_mapping_variables(self):
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()

        # Check 0 timestep leaves variables unchanged
        dt = 0
        self.system.compute_propagators(dt)
        before_q = np.copy(self.system.q_map)
        before_p = np.copy(self.system.p_map)
        self.system.propagate_mapping_variables()
        test.assert_array_almost_equal(before_q, self.system.q_map)
        test.assert_array_almost_equal(before_p, self.system.p_map)

        # Check for conservation of total electronic probability
        self.system.calculate_total_electronic_probability()
        before = np.copy(self.system.total_prob)

        dt = 1000
        self.system.compute_propagators(dt)

        for i in range(1000):
            self.system.propagate_mapping_variables()

        self.system.calculate_total_electronic_probability()
        test.assert_array_almost_equal(before, self.system.total_prob)

    def test_compute_adiabatic_derivative(self):
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_V_prime_matrix(self.potential)
        self.system.compute_adiabatic_derivative()
        # There is no way on earth this is incorrect.

    def test_compute_gamma_and_epsilon(self):
        dt = 1
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_V_prime_matrix(self.potential)
        self.system.compute_adiabatic_derivative()
        self.system.compute_gamma_and_epsilon(dt)

        # gamma is symmetric
        test.assert_array_almost_equal(self.system.gamma[0, 0, 0],
                                       self.system.gamma[0, 0, 0].T)
        # epsilon is skew symmetric
        test.assert_array_almost_equal(self.system.epsilon[0, 0, 0],
                                       -self.system.epsilon[0, 0, 0].T)

    def test_rotate_matrices(self):
        dt = 1
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_V_prime_matrix(self.potential)
        self.system.compute_adiabatic_derivative()
        self.system.compute_gamma_and_epsilon(dt)
        self.system.rotate_matrices()

        # E is symmetric
        test.assert_array_almost_equal(self.system.E[0, 0, 0], self.system.E[0, 0, 0].T)
        # F is skew symmetric
        test.assert_array_almost_equal(self.system.F[0, 0, 0], -self.system.F[0, 0, 0].T)

        # Test matrix multiplication is performed as expected
        self.system.gamma = np.random.randint(10, size=(1, 4, 1, 2, 2))
        self.system.epsilon = np.random.randint(10, size=(1, 4, 1, 2, 2))
        self.system.S_matrix = np.random.randint(10, size=(1, 4, 2, 2))

        self.system.rotate_matrices()

        ans = self.system.S_matrix[0,0]@self.system.gamma[0,0,0]@self.system.S_matrix[0, 0].T
        test.assert_array_almost_equal(self.system.E[0, 0, 0], ans)


    def test_propagate_bead_velocities(self):
        dt = 1
        self.system.compute_V_matrix(self.potential)
        self.system.diagonalise_V()
        self.system.compute_V_prime_matrix(self.potential)
        self.system.compute_adiabatic_derivative()
        self.system.compute_gamma_and_epsilon(dt)
        self.system.rotate_matrices()

        self.system.propagate_bead_velocities(dt)

    def test_calculate_total_electronic_probability(self):
        self.system.calculate_total_electronic_probability()
        test.assert_array_almost_equal(self.system.total_prob, 1)

    def test_calculate_state_probability(self):
        self.system.calculate_state_probability()
        correct = [[0, 1]]
        test.assert_array_almost_equal(self.system.state_prob, correct)

    def test_compute_bead_potential_energy(self):
         self.system.compute_bead_potential_energy(self.potential)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
        NRPMDTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
