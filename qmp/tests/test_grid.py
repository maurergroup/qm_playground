import unittest
from qmp.systems.grid import Grid
import numpy as np
import numpy.testing as test
from qmp.potential import Potential
from numpy.fft import fftn, ifftn

class GridTestCase(unittest.TestCase):

    def setUp(self):
        self.mass = 100
        self.N = 50
        start = 0
        end = 100
        self.end = end

        self.grid2d1n = Grid(self.mass, np.array([[start, end], [start, end]]),
                             self.N, 1)
        self.grid2d2n = Grid(self.mass, np.array([[start, end], [start, end]]),
                             self.N, 2)
        self.grid1d2n = Grid(self.mass, np.array([[start, end]]), self.N, 2)
        self.grid1d1n = Grid(self.mass, np.array([[start, end]]), self.N, 1)

        cell = np.array([[start, end], [start, end], [start, end]])
        self.grid3d1n = Grid(self.mass, cell, self.N, 1)

    def test_create_mesh(self):
        N = self.N
        steps = self.end / (self.N - 1)
        self.grid1d1n.create_mesh()
        test.assert_array_equal(np.shape(self.grid1d1n.mesh), (1, N))
        test.assert_array_equal(self.grid1d1n.steps, steps)

        self.grid2d1n.create_mesh()
        test.assert_array_equal(np.shape(self.grid2d1n.mesh), (2, N, N))
        test.assert_array_equal(self.grid2d1n.steps, [steps, steps])

        self.grid3d1n.create_mesh()
        test.assert_array_equal(np.shape(self.grid3d1n.mesh), (3, N, N, N))
        test.assert_array_equal(self.grid3d1n.steps, [steps, steps, steps])

    def test_define_laplacian(self):

        N = self.N
        # 1D, 2 level
        self.grid1d2n.define_laplacian()
        test.assert_equal(self.grid1d2n.L.shape, (2*N, 2*N))

        # 2D, 2 level
        self.grid2d2n.define_laplacian()
        test.assert_equal(self.grid2d2n.L.shape, (2*N**2, 2*N**2))

        # 1D, 1 level
        self.grid1d1n.define_laplacian()
        test.assert_equal(self.grid1d1n.L.shape, (N, N))

        # 2D, 1 level
        self.grid2d1n.define_laplacian()
        test.assert_equal(self.grid2d1n.L.shape, (N**2, N**2))

    def test_construct_V_matrix(self):

        N = self.N
        # 1D, 2 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]*4
        potential = Potential(cell, f=f, n=2)
        self.grid1d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d2n.V.shape, (2*N, 2*N))

        # 2D, 2 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]*4
        potential = Potential(cell, f=f, n=2)
        self.grid2d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid2d2n.V.shape, (2*N**2, 2*N**2))

        # 1D, 1 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]
        potential = Potential(cell, f=f, n=1)
        self.grid1d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d1n.V.shape, (N, N))

        # 2D, 1 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]
        potential = Potential(cell, f=f, n=1)
        self.grid2d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid2d1n.V.shape, (N**2, N**2))

        # 3D, 1 level
        cell = [[0, 1], [0, 1], [0, 1]]
        f = [lambda x, y, z: 2*x + 2*y + 2*z**2]
        potential = Potential(cell, f=f, n=1)
        self.grid3d1n.construct_V_matrix(potential)
        test.assert_equal(self.grid3d1n.V.shape, (N**3, N**3))

    def test_construct_coordinate_T(self):
        N = self.N

        # 1D, 2 level
        self.grid1d2n.construct_coordinate_T()
        test.assert_equal(self.grid1d2n.coordinate_T.shape, (2*N, 2*N))

        # 2D, 2 level
        self.grid2d2n.construct_coordinate_T()
        test.assert_equal(self.grid2d2n.coordinate_T.shape, (2*N**2, 2*N**2))

        # 1D, 1 level
        self.grid1d1n.construct_coordinate_T()
        test.assert_equal(self.grid1d1n.coordinate_T.shape, (N, N))

        # 2D, 1 level
        self.grid2d1n.construct_coordinate_T()
        test.assert_equal(self.grid2d1n.coordinate_T.shape, (N**2, N**2))

    def test_construct_hamiltonian(self):
        N = self.N

        # 1D, 2 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]*4
        potential = Potential(cell, f=f, n=2)
        self.grid1d2n.construct_V_matrix(potential)
        test.assert_equal(self.grid1d2n.V.shape, (2*N, 2*N))

        # 2D, 2 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]*4
        potential = Potential(cell, f=f, n=2)
        self.grid2d2n.construct_hamiltonian(potential)
        test.assert_equal(self.grid2d2n.H.shape, (2*N**2, 2*N**2))

        # 1D, 1 level
        cell = [[0, 1]]
        f = [lambda x: 2*x]
        potential = Potential(cell, f=f, n=1)
        self.grid1d1n.construct_hamiltonian(potential)
        test.assert_equal(self.grid1d1n.H.shape, (N, N))

        # 2D, 1 level
        cell = [[0, 1], [0, 1]]
        f = [lambda x, y: 2*x + 2*y]
        potential = Potential(cell, f=f, n=1)
        self.grid2d1n.construct_hamiltonian(potential)
        test.assert_equal(self.grid2d1n.H.shape, (N**2, N**2))

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

    # def test_construct_imaginary_potential(self):
    #     # 1D, 2 level
    #     x = self.grid1d2n.mesh[0]
    #     k = 5
    #     sigma = 1
    #     psi = np.exp(1j * k * x) * np.exp(-(x/sigma)**2)
    #     print(psi)
    #     self.grid1d2n.construct_imaginary_potential()
    #     print(self.grid1d2n.imag)
    #     # test.assert_equal(self.grid1d2n.psi.shape, (self.N * 2,))

    #     # # 2D, 2 level
    #     xx, yy = np.meshgrid(psi, psi)
    #     self.grid2d2n.construct_imaginary_potential()
    #     test.assert_equal(self.grid2d2n.psi.shape, (2 * self.N ** 2,))

    #     # 1D, 1 level
    #     self.grid1d1n.construct_imaginary_potential()
    #     test.assert_equal(self.grid1d1n.psi.shape, (self.N * 1,))

    #     # 2D, 1 level
    #     self.grid2d1n.construct_imaginary_potential()
    #     test.assert_equal(self.grid2d1n.psi.shape, (1 * self.N ** 2,))

    def test_compute_k(self):
        # Currently tests only 1D
        self.grid1d1n.compute_k()
        k = self.grid1d1n.k
        dk = abs(k[0] - k[1])
        correct = 2 * np.pi / (self.grid1d1n.end - self.grid1d1n.start)
        test.assert_almost_equal(correct, dk, decimal=1)

    def test_transform(self):
        psi = np.random.rand(self.grid1d1n.N)
        psi_t = self.grid1d1n.transform(psi, fftn)
        psi_t = self.grid1d1n.transform(psi_t, ifftn)
        test.assert_almost_equal(psi, psi_t)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            GridTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
