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
        T = 293.15
        self.rpmd = RPMD(coords, velocities, masses, n_beads, T)
        # self.rpmd = RPMD(coords, velocities, masses, n_beads, T,
        #                      init_type='position')

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
        self.rpmd.r = np.array([[[1, 1], [1, 1], [1, 1], [1, 1]],
                                [[2, 2], [2, 2], [2, 2], [2, 2]]])
        pe = self.rpmd.compute_potential_energy(self.potential)
        correct = [[1, 1, 1, 1],
                   [4, 4, 4, 4]]
        test.assert_array_equal(pe, correct)

    def test_compute_bead_potential_energy(self):
        # Not a very good test, the untested part is equal to zero. When all
        # beads have the same position.
        self.rpmd.r = np.array([[[1, 1], [1, 1], [1, 1], [1, 1]],
                                [[2, 2], [2, 2], [2, 2], [2, 2]]])
        pe = self.rpmd.compute_bead_potential_energy(self.potential)
        correct = [[1, 1, 1, 1],
                   [4, 4, 4, 4]]
        test.assert_array_equal(pe, correct)

    def test_compute_force(self):
        self.rpmd.r = np.array([[[3, 3], [3, 3], [3, 3], [3, 3]],
                                [[5, 5], [5, 5], [5, 5], [5, 5]]])
        force = self.rpmd.compute_force(self.potential)
        correct = -self.rpmd.r
        test.assert_array_almost_equal(force, correct)

    def test_initialise_transformer(self):
        self.rpmd.n_beads = 12
        n = self.rpmd.n_beads
        self.rpmd.initialise_transformer()
        c = self.rpmd.transformer

        j, k = 5, 8
        c48 = np.sqrt(2/n)*np.sin(2*np.pi*j*k/n)
        j, k = 2, 1
        c11 = np.sqrt(2/n)*np.cos(2*np.pi*j*k/n)
        c30 = np.sqrt(1/n)
        c56 = np.sqrt(1/n)*(-1)**6
        test.assert_array_almost_equal(c[4, 8], c48)
        test.assert_array_almost_equal(c[1, 1], c11)
        test.assert_array_almost_equal(c[3, 0], c30)
        test.assert_array_almost_equal(c[5, 6], c56)

    def test_initialise_normal_frequencies(self):
        self.rpmd.n_beads = 12
        self.rpmd.omega = 1
        self.rpmd.initialise_normal_frequencies()
        omegas = self.rpmd.omega_k

        omega_4 = 2 * self.rpmd.omega * np.sin(4 * np.pi / self.rpmd.n_beads)
        omega_8 = 2 * self.rpmd.omega * np.sin(8 * np.pi / self.rpmd.n_beads)

        test.assert_array_almost_equal(omegas[0], 0)
        test.assert_array_almost_equal(omegas[4], omega_4)
        test.assert_array_almost_equal(omegas[8], omega_8)

    def test_transform_to_normal_modes(self):

        self.rpmd.p = self.rpmd.v * self.rpmd.masses[..., None, None]
        self.rpmd.q = self.rpmd.r
        self.rpmd.initialise_transformer()

        self.rpmd.transform_to_normal_modes()

        normal_p_1_0_0 = np.zeros(2)
        for j in range(self.rpmd.n_beads):
            for k in range(self.rpmd.ndim):
                normal_p_1_0_0[k] += (self.rpmd.transformer[j, 0]
                                      * self.rpmd.p[1, j, k])
        test.assert_array_almost_equal(normal_p_1_0_0[0],
                                       self.rpmd.p_normal[1, 0, 0])

        normal_q_1_0_0 = np.zeros(2)
        for j in range(self.rpmd.n_beads):
            for k in range(self.rpmd.ndim):
                normal_q_1_0_0[k] += (self.rpmd.transformer[j, 0]
                                      * self.rpmd.q[1, j, k])
        test.assert_array_almost_equal(normal_q_1_0_0[0],
                                       self.rpmd.q_normal[1, 0, 0])

        normal_q_1_1_0 = np.zeros(2)
        for j in range(self.rpmd.n_beads):
            for k in range(self.rpmd.ndim):
                normal_q_1_1_0[k] += (self.rpmd.transformer[j, 1]
                                      * self.rpmd.q[1, j, k])
        test.assert_array_almost_equal(normal_q_1_1_0[0],
                                       self.rpmd.q_normal[1, 1, 0])

    def test_reversible_transformation(self):
        self.rpmd.p = self.rpmd.v * self.rpmd.masses[..., None, None]
        self.rpmd.q = self.rpmd.r
        self.rpmd.initialise_transformer()

        p_before = self.rpmd.p
        q_before = self.rpmd.q

        self.rpmd.transform_to_normal_modes()
        self.rpmd.transform_from_normal_modes()

        test.assert_array_almost_equal(p_before, self.rpmd.p)
        test.assert_array_almost_equal(q_before, self.rpmd.q)

    def test_initialise_thermostat(self):
        dt = 1
        self.rpmd.omega_k = np.array([0, 2, 3])
        self.rpmd.initialise_thermostat(dt, tau0=dt)

        gamma_k = 2 * self.rpmd.omega_k
        gamma_k[0] = 1 / dt

        c1 = np.exp(-dt/2 * gamma_k)
        c2 = np.sqrt(1 - c1 ** 2)

        test.assert_array_almost_equal(c1, self.rpmd.c1)
        test.assert_array_almost_equal(c2, self.rpmd.c2)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RPMDTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
