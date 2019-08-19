import unittest
from qmp.systems.grid import Grid1D
import numpy as np
import numpy.testing as test
from qmp.potential import Potential


class Grid1DTestCase(unittest.TestCase):

    def setUp(self):
        self.mass = 1
        self.start = 0
        self.end = 1
        self.N = 5
        self.nstates = 2

        self.grid = Grid1D(self.mass, self.start, self.end, self.N,
                           states=self.nstates)

    def test_initial_state(self):
        test.assert_equal(self.grid.psi.shape, (self.nstates*self.N,))

    def test_define_laplacian(self):
        correct_shape = (self.N*self.nstates, self.N*self.nstates)
        actual_shape = self.grid.define_laplacian().shape
        test.assert_equal(correct_shape, actual_shape)

    def test_construct_T_matrix(self):
        T = self.grid.construct_T_matrix()
        test.assert_equal(T.shape, self.grid.L.shape)

    def test_construct_V_matrix(self):
        f = [lambda a: a]*3 + [lambda a: np.zeros_like(a)]
        pot = Potential(n=self.nstates, f=f)

        correct_shape = (self.nstates * self.N, self.nstates * self.N)
        V = self.grid.construct_V_matrix(pot)
        test.assert_equal(correct_shape, V.shape)

    def test_set_initial_wvfn(self):
        psi = np.linspace(1, 1, self.N)
        self.grid.set_initial_wvfn(psi)
        correct_psi = np.concatenate((psi, np.zeros_like(psi)))
        test.assert_equal(self.grid.psi, correct_psi)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            Grid1DTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
