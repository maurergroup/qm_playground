import unittest

import numpy as np
import numpy.testing as test

from qmp.potential import Potential
from qmp.systems.mf_rpmd import MF_RPMD


class MF_RPMDTestCase(unittest.TestCase):

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
        self.T = 3000
        self.beta = 1 / (self.T * n_beads)
        n_states = 2
        self.system = MF_RPMD(coords, velocities, masses, n_beads=n_beads,
                              T=self.T,
                              n_states=n_states)

        def f(a):
            return 0.00002*a
        def f2(a):
            return -0.00002*a
        def off(a):
            return np.full_like(a, -0.1)

        self.potential = Potential(cell, f=[f, off, off, f2], n=2)

    def test_set_V_matrix(self):
        self.system.set_V_matrix(self.potential)
        v = self.system.V_matrix[0, 0]
        ans = [[0.0002, -0.1], [-0.1, -0.0002]]
        test.assert_array_almost_equal(v, ans)

    def test_diagonalise_at_current_position(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        t = self.system.S_matrix[0, 0] @ self.system.V_matrix[0, 0] @ self.system.S_matrix[0, 0].T
        test.assert_array_almost_equal(t, self.system.lambdas[0, 0])

    def test_set_exp_lambdas(self):
        self.system.lambdas = np.array([[[[-0.1, 0], [0, 0.1]],
                                         [[-0.1, 0], [0, 0.1]],
                                         [[-0.1, 0], [0, 0.1]],
                                         [[-0.1, 0], [0, 0.1]]]])
        self.system.set_exp_lambdas()
        exp = self.system.exp_lambdas[0, 0, 0, 0]
        ans = np.exp(0.1 * self.system.beta)
        test.assert_array_almost_equal(exp, ans)

    def test_calculate_derivatives(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()

        self.system.calculate_derivatives(self.potential)

        # print(self.system.lam_exp_deriv)

    def test_calculate_D_and_M_matrices(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()
        self.system.calculate_derivatives(self.potential)
        self.system.calculate_D_and_M_matrices()

    def test_calculate_F_matrices(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()
        self.system.calculate_derivatives(self.potential)
        self.system.calculate_D_and_M_matrices()
        self.system.calculate_F_matrices()

    def test_calculate_G_matrices(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()
        self.system.calculate_derivatives(self.potential)
        self.system.calculate_D_and_M_matrices()
        self.system.calculate_F_matrices()
        self.system.calculate_G_matrices()

    def test_calculate_hole_matrices(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()
        self.system.calculate_derivatives(self.potential)
        self.system.calculate_D_and_M_matrices()
        self.system.calculate_F_matrices()
        self.system.calculate_G_matrices()
        self.system.calculate_hole_matrices()

    def test_compute_force(self):
        self.system.set_V_matrix(self.potential)
        self.system.diagonalise_at_current_position()
        self.system.set_exp_lambdas()
        self.system.calculate_derivatives(self.potential)
        self.system.calculate_D_and_M_matrices()
        self.system.calculate_F_matrices()
        self.system.calculate_G_matrices()
        self.system.calculate_hole_matrices()
        f = self.system.compute_force()
        print(f)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
        MF_RPMDTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
