"""Potentials for the 3 Tully models.

By default these are configured for -10 to 10 one dimensional cells with the
default parameters set out by Tully (1990)."""
from qmp.potential import Potential
import numpy as np


class TullySimpleAvoidedCrossing(Potential):

    def __init__(self, cell=[[-10, 10]], a=0.01, b=1.6, c=0.005, d=1.0):

        self.A = a
        self.B = b
        self.C = c
        self.D = d

        f = self.get_f()
        firstd = self.get_deriv()

        Potential.__init__(self, cell=cell, n=2, f=f, firstd=firstd)

    def get_f(self):
        def v11(x):
            return np.copysign(self.A, x) * (1.0 - np.exp(-self.B*abs(x)))

        def v12(x):
            return self.C * np.exp(-self.D * x**2)

        def v22(x):
            return -v11(x)

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return self.A * self.B * np.exp(-self.B * abs(x))

        def v12(x):
            return -2.0 * self.C * self.D * x * np.exp(-self.D*x**2)

        def v22(x):
            return -v11(x)

        return np.array([v11, v12, v12, v22])


class TullyDualAvoidedCrossing(Potential):

    def __init__(self, cell=[[-10, 10]], a=0.1, b=0.28, c=0.015, d=0.06, e=0.05):

        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E_0 = e

        f = self.get_f()
        firstd = self.get_deriv()

        Potential.__init__(self, cell=cell, n=2, f=f, firstd=firstd)

    def get_f(self):
        def v11(x):
            return 0 * x

        def v12(x):
            return self.C * np.exp(-self.D * x**2)

        def v22(x):
            return -self.A * np.exp(-self.B * x**2) + self.E_0

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return 0 * x

        def v12(x):
            return -2.0 * self.C * self.D * x * np.exp(-self.D * x**2)

        def v22(x):
            return 2.0 * self.A * self.B * x * np.exp(-self.B * x**2)

        return np.array([v11, v12, v12, v22])


class TullyExtendedCoupling(Potential):
    def __init__(self, cell=[[-10, 10]], a=0.0006, b=0.10, c=0.90):

        self.A = a
        self.B = b
        self.C = c

        f = self.get_f()
        firstd = self.get_deriv()

        Potential.__init__(self, cell=cell, n=2, f=f, firstd=firstd)

    def get_f(self):
        def v11(x):
            return np.full_like(x, self.A, dtype=float)

        def v12(x):

            def positive(x):
                return self.B * (2 - np.exp(-self.C * x))

            def negative(x):
                return self.B * np.exp(self.C * x)

            return np.piecewise(x, [x < 0., x >= 0.], [negative,
                                                       positive])

        def v22(x):
            return -v11(x)

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return 0 * x

        def v12(x):
            return self.B * self.C * np.exp(-self.C * np.abs(x))

        def v22(x):
            return 0 * x

        return np.array([v11, v12, v12, v22])
