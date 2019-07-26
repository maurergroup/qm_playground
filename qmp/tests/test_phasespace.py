import unittest
from qmp.systems.phasespace import PhaseSpace
from qmp.potential import Potential
import numpy as np
import numpy.testing as test


class PhaseSpaceTestCase(unittest.TestCase):

    def setUp(self):
        cell = [[-5, 5], [-5, 5]]
        coords = [[0, 0],
                  [3, 3],
                  [3, 7]]
        velocities = [[1, 1],
                      [-1, 2],
                      [0, 0]]
        masses = [1, 2, 1]
        self.phasespace = PhaseSpace(coords, velocities, masses)

        def f(a, b):
            return a * b
        self.potential = Potential(cell, f=f)

    def test_compute_kinetic_energy(self):
        ke = self.phasespace.compute_kinetic_energy()
        correct = [2, 10, 0]
        test.assert_array_equal(ke, correct)

    def test_compute_potential_energy(self):
        pe = self.phasespace.compute_potential_energy(self.potential)
        correct = [0, 9, 21]
        test.assert_array_equal(pe, correct)

    def test_compute_force(self):
        force = self.phasespace.compute_force(self.potential)
        correct = np.array([[0, -3],
                            [0, -3],
                            [-3, 6]])
        test.assert_array_almost_equal(force, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            PhaseSpaceTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
