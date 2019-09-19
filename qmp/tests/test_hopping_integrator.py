import unittest
from qmp.systems.hopping import Hopping
from qmp.potential.tullymodels import TullySimpleAvoidedCrossing
from qmp.integrator.hoppingintegrators import HoppingIntegrator
import numpy as np
import numpy.testing as test


class HoppingIntegratorTestCase(unittest.TestCase):

    def setUp(self):
        x = np.array([-0.])
        v = np.array([0.01])
        m = np.array([2000.])
        state = 0
        self.pot = TullySimpleAvoidedCrossing()
        self.hop = Hopping(x, v, m, state, self.pot)
        self.hop.reset_system(self.pot)

        self.hopper = HoppingIntegrator()
        self.hopper.system = self.hop


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            HoppingIntegratorTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
