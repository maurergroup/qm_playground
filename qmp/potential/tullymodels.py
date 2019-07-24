from qmp.potential import Potential
import math
import numpy as np


class TullySimpleAvoidedCrossing(Potential):

    def __init__(self, cell=[[-5, 5]], a=0.01, b=1.6, c=0.005, d=1.0):

        Potential.__init__(self, cell=cell, n=2)

        self.A = a
        self.B = b
        self.C = c
        self.D = d

        self.f = self.get_f()
        self.firstd = self.get_deriv()

    def get_f(self):
        def v11(x):
            return math.copysign(self.A, x) * (1.0 - math.exp(-self.B*abs(x)))

        def v12(x):
            return self.C * math.exp(-self.D * x**2)

        def v22(x):
            return -v11(x)

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return self.A * self.B * math.exp(-self.B * abs(x))

        def v12(x):
            return -2.0 * self.C * self.D * x * math.exp(-self.D*x**2)

        def v22(x):
            return -v11(x)

        return np.array([v11, v12, v12, v22])
