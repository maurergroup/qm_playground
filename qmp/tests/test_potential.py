import unittest
from qmp import potential
import numpy as np


class PotentialTestCase(unittest.TestCase):

    def setUp(self):
        one_dimension = potential.Potential(f=lambda a: a)
        two_dimension = potential.Potential([[0, 1], [0, 1]], f=lambda a, b:
                a+b)
        self.potentials = [one_dimension, two_dimension]

    def test_evaluate_single_point(self):
        for i, pot in enumerate(self.potentials):
            with self.subTest(dimension=i+1):
                point = np.full(pot.dimension, 5)
                evaluation = pot.evaluate_potential(point)
                self.assertEqual(evaluation, 5*pot.dimension)

    def test_call(self):
        point = [5]
        self.assertEqual(self.potentials[0](point), 5)
        point = [5, 5]
        self.assertTrue((self.potentials[0](point) == [5, 5]).all())

        point = [[5, 5]]
        self.assertEqual(self.potentials[1](point), 10)
        point = [[5, 5], [5, 5]]
        self.assertTrue(
            len(self.potentials[1](point)) == self.potentials[1].dimension)

    def test_deriv_single_point(self):
        point = [5]
        np.testing.assert_allclose(
                self.potentials[0].single_point_deriv(point), [1])
        point = [5, 5]
        np.testing.assert_allclose(
                self.potentials[1].single_point_deriv(point), [1, 1])

    def test_deriv(self):
        point = [[5]]
        np.testing.assert_allclose(
                self.potentials[0].deriv(point), [[1]])
        point = [[5], [5]]
        np.testing.assert_allclose(
                self.potentials[0].deriv(point), [[1], [1]])

        point = [[5, 5]]
        np.testing.assert_allclose(
                self.potentials[1].deriv(point), [[1, 1]])
        point = [[5, 5], [5, 5]]
        np.testing.assert_allclose(
                self.potentials[1].deriv(point), [[1, 1], [1, 1]])

    def test_hess_single_point(self):
        point = [5]
        self.assertAlmostEqual(
                self.potentials[0].single_point_hess(point), 0)
        point = [5, 5]
        self.assertAlmostEqual(
                self.potentials[1].single_point_hess(point), 0)

    def test_hess(self):
        point = [[5]]
        np.testing.assert_almost_equal(
                self.potentials[0].hess(point), [0])
        point = [[5], [5]]
        np.testing.assert_almost_equal(
                self.potentials[0].hess(point), [0, 0])

        point = [[5, 5]]
        np.testing.assert_almost_equal(
                self.potentials[1].hess(point), [0])
        point = [[5, 5], [5, 5]]
        np.testing.assert_almost_equal(
                self.potentials[1].hess(point), [0, 0])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            PotentialTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
