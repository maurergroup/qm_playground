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
        self.pot = TullySimpleAvoidedCrossing()
        self.hop = Hopping(x, v, m, state, self.pot)
        self.hop.reset_system(self.pot)

    def test_construct_V_matrix(self):
        V = self.hop.construct_V_matrix(self.pot)
        correct = [[0, 0.005], [0.005, 0]]
        test.assert_array_equal(V, correct)

    def test_construct_Nabla_matrix(self):
        D = self.hop.construct_Nabla_matrix(self.pot)
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

    def test_propagate_density_matrix(self):
        self.hop.propagate_density_matrix(dt=2)
        new = self.hop.density_matrix
        correct = [[0.99897638+0.j, -0.03197603-0.00031988j],
                   [-0.03197603+0.00031988j, 0.00102362+0.j]]
        test.assert_array_almost_equal(new, correct)

    def test_rescale_velocity(self):
        desired_state = 1
        old = self.hop.hamiltonian[0, 0]
        new = self.hop.hamiltonian[1, 1]
        deltaV = new - old

        self.hop.rescale_velocity(deltaV, desired_state)
        test.assert_array_almost_equal(self.hop.v, [0.00948683])

    def test_get_probabilities(self):
        self.hop.density_matrix = np.array([[0.95563873+8.32667268e-17j,
                                            -0.18390605-9.25846251e-02j],
                                            [-0.18390605+9.25846251e-02j,
                                            0.04436127+4.77048956e-18j]])
        self.hop.v = [0.01490894]
        self.hop.derivative_coupling = np.array([[0.00272372, -0.35526184],
                                                 [0.35526184, -0.00272372]])
        result = self.hop.get_probabilities(dt=2)
        correct = 0.00407716
        test.assert_array_almost_equal(result, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            HoppingTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
