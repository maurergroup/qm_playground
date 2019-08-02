import unittest
from qmp.systems.hopping import Hopping
from qmp.potential.tullymodels import TullySimpleAvoidedCrossing
from qmp.integrator.hoppingintegrators import HoppingIntegrator
import numpy as np
import numpy.testing as test


class HoppingIntegratorTestCase(unittest.TestCase):

    def setUp(self):
        x = np.array([-0.])
        v = np.array([0.01])
        m = np.array([2000.])
        state = 0
        pot = TullySimpleAvoidedCrossing()
        self.hop = Hopping(x, v, m, state, pot)
        self.hop.reset_system()

        self.hopper = HoppingIntegrator()
        self.hopper.system = self.hop

    def test_propagate_density_matrix(self):
        dm = self.hopper.system.density_matrix
        new = self.hopper.propagate_density_matrix(dm, dt=2)
        correct = [[0.99897638+0.j, -0.03197603-0.00031988j],
                   [-0.03197603+0.00031988j, 0.00102362+0.j]]
        test.assert_array_almost_equal(new, correct)

    def test_rescale_velocity(self):
        desired_state = 1
        old = self.hop.hamiltonian[0, 0]
        new = self.hop.hamiltonian[1, 1]
        deltaV = new - old

        self.hopper.rescale_velocity(deltaV, desired_state)
        test.assert_array_almost_equal(self.hop.v, [0.00948683])

    def test_get_probabilities(self):
        self.hop.V = np.array([[-0.00723533, 0.00262154],
                              [0.00262154, 0.00723533]])
        self.hop.density_matrix = np.array([[0.95563873+8.32667268e-17j,
                                            -0.18390605-9.25846251e-02j],
                                            [-0.18390605+9.25846251e-02j,
                                            0.04436127+4.77048956e-18j]])
        vel = [0.01490894]
        self.hop.derivative_coupling = np.array([[0.00272372, -0.35526184],
                                                 [0.35526184, -0.00272372]])
        result = self.hopper.get_probabilities(vel)
        correct = [0, 0.00407716]
        test.assert_array_almost_equal(result, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            HoppingIntegratorTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
