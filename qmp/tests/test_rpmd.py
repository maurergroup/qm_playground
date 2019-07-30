import unittest
from qmp.systems.rpmd import RPMD
from qmp.potential import Potential
import numpy as np
import numpy.testing as test


class RPMDTestCase(unittest.TestCase):

    def setUp(self):
        cell = [[0, 20], [0, 20]]
        coords = [[1, 1],
                  [2, 2]]
        velocities = [[0, 0], [1, 1]]
        masses = [1860, 100]
        n_beads = 4
        T = [293.15, 293.15]
        self.rpmd = RPMD(coords, velocities, masses, n_beads, T,
                         init_type='velocity')

        def f(a, b):
            return a * b
        self.potential = Potential(cell, f=f)

    def test_compute_kinetic_energy(self):
        ke = self.rpmd.compute_kinetic_energy()
        correct = np.zeros_like(ke[0])
        test.assert_array_almost_equal(ke[0], correct, decimal=2)
        correct = np.full_like(ke[0], 100)
        test.assert_array_almost_equal(ke[1], correct, decimal=0)

    def test_compute_potential_energy(self):
        pe = self.rpmd.compute_potential_energy(self.potential)
        correct = [[1, 1, 1, 1],
                   [4, 4, 4, 4]]
        test.assert_array_equal(pe, correct)

    def test_compute_bead_potential_energy(self):
        # Not a very good test, the untested part is equal to zero. When all
        # beads have the same position.
        pe = self.rpmd.compute_bead_potential_energy(self.potential)
        correct = [[1, 1, 1, 1],
                   [4, 4, 4, 4]]
        test.assert_array_equal(pe, correct)

    def test_compute_force(self):
        force = self.rpmd.compute_force(self.potential)
        correct = np.array([[[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                            [[-2, -2], [-2, -2], [-2, -2], [-2, -2]]])
        test.assert_array_almost_equal(force, correct)

    def test_compute_bead_force(self):
        # Not a very good test, the untested part is equal to zero. When all
        # beads have the same position.
        force = self.rpmd.compute_bead_force(self.potential)
        correct = np.array([[[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                            [[-2, -2], [-2, -2], [-2, -2], [-2, -2]]])
        test.assert_array_almost_equal(force, correct)

    def test_compute_omega_Rugh(self):
        # Not finished yet.
        self.rpmd.compute_omega_Rugh(self.potential)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RPMDTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
