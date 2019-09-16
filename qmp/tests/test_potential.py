import unittest
from qmp import potential
import numpy as np
import numpy.testing as test


class PotentialTestCase(unittest.TestCase):

    def setUp(self):
        self.one_dimension = potential.Potential([[0, 1]], f=lambda a: a)
        self.two_dimension = potential.Potential([[0, 1], [0, 1]],
                                                 f=lambda a, b: a+b)
        self.potentials = [self.one_dimension, self.two_dimension]

    def test_call_one_dimension(self):
        point = 5
        self.assertEqual(self.one_dimension(point), 5)
        point = np.linspace(0, 5, 10)
        test.assert_array_equal(self.one_dimension(point), point)

    def test_call_two_dimension(self):
        self.assertEqual(self.two_dimension(5, 5), 10)
        x = np.linspace(0, 5, 1)
        xx, yy = np.meshgrid(x, x)
        test.assert_array_equal(self.two_dimension(xx, yy), xx+yy)

    def test_deriv(self):
        point = 5
        test.assert_allclose(self.potentials[0].deriv(point), 1)
        point = np.array([[5], [5]])
        test.assert_allclose(self.potentials[0].deriv(point), [[1], [1]])

        point = np.array([[5, 5]])
        test.assert_allclose(self.potentials[1].deriv(point), [[1, 1]])
        point = np.array([[5, 5], [5, 5], [0, 0]])
        test.assert_allclose(self.potentials[1].deriv(point), [[1, 1],
                                                               [1, 1],
                                                               [1, 1]])

    def test_hess(self):
        point = 5
        test.assert_almost_equal(self.potentials[0].hess(point), [0])
        point = np.array([[5], [5]])
        test.assert_almost_equal(self.potentials[0].hess(point), [[0], [0]])

        point = np.array([[5, 5]])
        test.assert_almost_equal(self.potentials[1].hess(point), [0])
        point = np.array([[5, 5], [5, 5], [5, 5]])
        test.assert_almost_equal(self.potentials[1].hess(point), [0, 0, 0])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            PotentialTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
