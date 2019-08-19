import unittest
from qmp.systems.grid import Grid1D
import numpy as np
import numpy.testing as test
from qmp.potential import tullymodels
from qmp.integrator.waveintegrators import SOFT_Propagator
from qmp.data_containers import Data
from scipy import signal


class SOFT_PropagatorTestCase(unittest.TestCase):

    def setUp(self):
        self.mass = 1
        self.start = -10
        self.end = 10
        self.N = 256
        self.nstates = 2
        self.size = self.N * self.nstates

        self.system = Grid1D(self.mass, self.start, self.end, self.N,
                             states=self.nstates)
        self.pot = tullymodels.TullySimpleAvoidedCrossing()
        self.dt = 0.5
        self.integrator = SOFT_Propagator(self.dt)
        self.integrator.system = self.system
        self.integrator.V = self.system.construct_V_matrix(self.pot)
        self.integrator.k = self.integrator.compute_k()
        psi = signal.gaussian(self.N, 0.5)
        self.system.set_initial_wvfn(psi)
        self.integrator.propT = self.integrator.expT(self.dt/2)
        self.integrator.propV = self.integrator.expV(self.dt)

    def test_initialise_start(self):
        self.integrator.initialise_start(self.system, self.pot)
        test.assert_equal(self.integrator.V.shape,
                          (self.size, self.size))
        test.assert_equal(self.integrator.k.shape, (self.N,))
        test.assert_equal(np.shape(self.integrator.psi_t),
                          (1, self.size))

    def test_compute_k(self):
        # Currently tests only 1D
        k = self.integrator.compute_k()
        dk = abs(k[0] - k[1])
        test.assert_almost_equal(2*np.pi/(self.end-self.start), dk, decimal=1)

    def test_expT(self):
        expT = self.integrator.expT(self.dt)
        test.assert_equal(expT.shape, (self.size, self.size))

    def test_expV(self):
        expV = self.integrator.expV(self.dt)
        test.assert_equal(expV.shape, (self.size, self.size))

    def test_propagate_psi(self):
        self.integrator.propagate_psi()
        shape_after = self.system.psi.shape
        test.assert_equal(shape_after, (self.size,))

    def test_compute_energies(self):
        self.integrator.psi_t = np.array([self.system.psi, self.system.psi])
        self.integrator.compute_energies()
        test.assert_equal(self.integrator.E_kin_t.shape, (2,))

    def test_store_result(self):
        self.integrator.psi_t = [self.system.psi]
        self.integrator.store_result()
        shape = np.shape(self.integrator.psi_t)
        test.assert_equal(shape, (2, self.size))

    def test_assign_data(self):
        # No assertions but runs without error.
        self.integrator.psi_t = np.array([self.system.psi])
        self.integrator.compute_energies()
        data = Data()
        self.integrator.assign_data(data)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            SOFT_PropagatorTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)