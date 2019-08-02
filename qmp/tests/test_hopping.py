import unittest
from qmp.systems.hopping import Hopping
from qmp.potential.tullymodels import TullySimpleAvoidedCrossing
import numpy as np
import numpy.testing as test


class HoppingTestCase(unittest.TestCase):

    def setUp(self):
        x = np.array([-0.])
        v = np.array([0.01])
        m = np.array([2000.])
        state = 0
        pot = TullySimpleAvoidedCrossing()
        self.hop = Hopping(x, v, m, state, pot)
        self.hop.reset_system()

    def test_construct_V_matrix(self):
        V = self.hop.construct_V_matrix()
        correct = [[0, 0.005], [0.005, 0]]
        test.assert_array_equal(V, correct)

    def test_construct_Nabla_matrix(self):
        D = self.hop.construct_Nabla_matrix()
        correct = [[0.016, 0], [0, -0.016]]
        test.assert_array_equal(D, correct)

    def test_compute_coeffs(self):
        e, c = self.hop.compute_coeffs()
        correct_c = [[-0.70710678, 0.70710678], [0.70710678, 0.70710678]]
        correct_e = [-0.005, 0.005]
        test.assert_array_almost_equal(c, correct_c)
        test.assert_array_almost_equal(e, correct_e)

    def test_compute_force(self):
        f = self.hop.compute_force()
        correct = [0, 0]
        test.assert_array_almost_equal(f, correct)

    def test_compute_hamiltonian(self):
        h = self.hop.compute_hamiltonian()
        correct = [[-5.00000000e-03, -1.10682021e-19],
                   [1.10682021e-19,  5.00000000e-03]]
        test.assert_array_almost_equal(h, correct)

    def test_compute_derivative_coupling(self):
        d = self.hop.compute_derivative_coupling()
        correct = [[0, -1.6],
                   [1.6, 0]]
        test.assert_array_almost_equal(d, correct)

    def test_compute_propagating_hamiltonian(self):
        h = self.hop.compute_propagating_hamiltonian()
        correct = [[-5.00000000e-03+0.j, 0+0.016j],
                   [0-0.016j, 5.00000000e-03+0.j]]
        test.assert_array_almost_equal(h, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            HoppingTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
