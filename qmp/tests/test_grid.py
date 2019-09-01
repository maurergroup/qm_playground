import unittest
from qmp.systems.grid import Grid
import numpy as np
import numpy.testing as test
from qmp.potential import Potential


class GridTestCase(unittest.TestCase):

    def setUp(self):
        self.mass = 1
        self.N = 5

        self.grid2d1n = Grid(self.mass, np.array([[0, 1], [0, 1]]), 5, 1)
        self.grid2d2n = Grid(self.mass, np.array([[0, 1], [0, 1]]), 5, 2)
        self.grid1d2n = Grid(self.mass, np.array([[0, 1]]), 5, 2)
        self.grid1d1n = Grid(self.mass, np.array([[0, 1]]), 5, 1)

        cell = np.array([[0, 1], [0, 1], [0, 1]])
        self.grid3d1n = Grid(self.mass, cell, 5, 1)

    def test_create_mesh(self):
        self.grid1d1n.create_mesh()
        test.assert_array_equal(self.grid1d1n.mesh, [[0, 0.25, 0.5, 0.75, 1]])
        test.assert_array_equal(self.grid1d1n.steps, [0.25])

        self.grid2d1n.create_mesh()
        test.assert_array_equal(np.shape(self.grid2d1n.mesh), (2, 5, 5))
        test.assert_array_equal(self.grid2d1n.steps, [0.25, 0.25])

        self.grid3d1n.create_mesh()
        test.assert_array_equal(np.shape(self.grid3d1n.mesh), (3, 5, 5, 5))
        test.assert_array_equal(self.grid3d1n.steps, [0.25, 0.25, 0.25])

    def test_define_laplacian(self):

        # 1D, 2 level
        self.grid1d2n.define_laplacian()
        test.assert_equal(self.grid1d2n.L.shape, (10, 10))

        # 2D, 2 level
        self.grid2d2n.define_laplacian()
        test.assert_equal(self.grid2d2n.L.shape, (50, 50))

        # 1D, 1 level
        self.grid1d1n.define_laplacian()
        test.assert_equal(self.grid1d1n.L.shape, (5, 5))

        # 2D, 1 level
        self.grid2d1n.define_laplacian()
        test.assert_equal(self.grid2d1n.L.shape, (25, 25))

    def test_construct_V_matrix(self):

        # 1D, 2 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]*4
        potential = Potential(cell, f=f, n=2)
        self.grid1d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d2n.V.shape, (10, 10))

        # 2D, 2 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]*4
        potential = Potential(cell, f=f, n=2)
        self.grid2d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid2d2n.V.shape, (50, 50))

        # 1D, 1 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]
        potential = Potential(cell, f=f, n=1)
        self.grid1d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d1n.V.shape, (5, 5))

        # 2D, 1 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]
        potential = Potential(cell, f=f, n=1)
        self.grid2d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid2d1n.V.shape, (25, 25))

        # 3D, 1 level
        cell = [[0, 1], [0, 1], [0, 1]]
        f = [lambda x, y, z: 2*x + 2*y + 2*z**2]
        potential = Potential(cell, f=f, n=1)
        self.grid3d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid3d1n.V.shape, (125, 125))

    def test_construct_T_matrix(self):
        # 1D, 2 level
        self.grid1d2n.construct_T_matrix()
        test.assert_equal(self.grid1d2n.T.shape, (10, 10))

        # 2D, 2 level
        self.grid2d2n.construct_T_matrix()
        test.assert_equal(self.grid2d2n.T.shape, (50, 50))

        # 1D, 1 level
        self.grid1d1n.construct_T_matrix()
        test.assert_equal(self.grid1d1n.T.shape, (5, 5))

        # 2D, 1 level
        self.grid2d1n.construct_T_matrix()
        test.assert_equal(self.grid2d1n.T.shape, (25, 25))

    def test_construct_hamiltonian(self):
        # 1D, 2 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]*4
        potential = Potential(cell, f=f, n=2)
        self.grid1d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d2n.V.shape, (10, 10))

        # 2D, 2 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]*4
        potential = Potential(cell, f=f, n=2)
        self.grid2d2n.construct_hamiltonian(potential)
        test.assert_equal(self.grid2d2n.H.shape, (50, 50))

        # 1D, 1 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]
        potential = Potential(cell, f=f, n=1)
        self.grid1d1n.construct_hamiltonian(potential)
        test.assert_equal(self.grid1d1n.H.shape, (5, 5))

        # 2D, 1 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]
        potential = Potential(cell, f=f, n=1)
        self.grid2d1n.construct_hamiltonian(potential)
        test.assert_equal(self.grid2d1n.H.shape, (25, 25))

    def test_set_initial_wvfn(self):
        # 1D, 2 level
        psi = np.ones(self.N)
        self.grid1d2n.set_initial_wvfn(psi)
        test.assert_equal(self.grid1d2n.psi.shape, (self.N * 2,))

        # # 2D, 2 level
        xx, yy = np.meshgrid(psi, psi)
        self.grid2d2n.set_initial_wvfn(xx)
        test.assert_equal(self.grid2d2n.psi.shape, (2 * self.N ** 2,))

        # 1D, 1 level
        self.grid1d1n.set_initial_wvfn(psi)
        test.assert_equal(self.grid1d1n.psi.shape, (self.N * 1,))

        # 2D, 1 level
        self.grid2d1n.set_initial_wvfn(xx)
        test.assert_equal(self.grid2d1n.psi.shape, (1 * self.N ** 2,))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            GridTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
