import unittest

import numpy as np
import numpy.testing as test
from scipy import signal

from qmp.data_containers import Data
from qmp.integrator.waveintegrators import SOFT_Propagator
from qmp.potential import tullymodels
from qmp.systems.grid import Grid


class SOFT_PropagatorTestCase(unittest.TestCase):

    def setUp(self):
        self.mass = 1
        self.cell = np.array([[-10, 10]])
        self.N = 256
        self.nstates = 2
        self.size = self.N * self.nstates

        self.system = Grid(self.mass, self.cell, self.N,
                           states=self.nstates)
        self.pot = tullymodels.TullySimpleAvoidedCrossing()
        self.dt = 0.5
        self.integrator = SOFT_Propagator(self.dt)
        self.integrator.potential = self.pot
        self.integrator.system = self.system
        self.integrator.V = self.system.construct_V_matrix(self.pot)
        self.system.compute_k()
        psi = signal.gaussian(self.N, 0.5)
        self.system.set_initial_wvfn(psi)
        self.integrator.propT = self.integrator._expT(self.dt/2)
        self.integrator.propV = self.integrator._expV(self.dt)

    def test_initialise_start(self):
        self.integrator._initialise_start()
        test.assert_equal(self.system.V.shape,
                          (self.size, self.size))
        test.assert_equal(self.system.k.shape, (self.size,))
        test.assert_equal(np.shape(self.integrator.psi_t),
                          (1, self.size))

    def test_expT(self):
        expT = self.integrator._expT(self.dt)
        test.assert_equal(expT.shape, (self.size,))

    def test_expV(self):
        expV = self.integrator._expV(self.dt)
        test.assert_equal(expV.shape, (self.size, self.size))

    def test_propagate_system(self):
        self.integrator._propagate_system()
        shape_after = self.system.psi.shape
        test.assert_equal(shape_after, (self.size,))

    def test_compute_current_energy(self):
        E = self.integrator._compute_current_energy()
        test.assert_equal(type(E), np.float64)

    def test_store_result(self):
        self.integrator.psi_t = [self.system.psi]
        self.integrator.E_t = []
        self.integrator._store_result()
        shape = np.shape(self.integrator.psi_t)
        test.assert_equal(shape, (2, self.size))
        shape = np.shape(self.integrator.E_t)
        test.assert_equal(shape, (1,))

    def test_assign_data(self):
        # No assertions but runs without error.
        self.integrator.status = 'good'
        self.system.absorbed_density = [0.2, 0.8]
        self.integrator.psi_t = np.array([self.system.psi])
        self.integrator.E_t = np.array([5.02])
        self.integrator._compute_current_energy()
        data = Data()
        self.integrator._assign_data(data)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            SOFT_PropagatorTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
