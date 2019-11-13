from .potential import Potential
import numpy as np


class SpinBoson(Potential):

    def __init__(self, cell=[[-10, 10]], gamma=1, delta=1, m=1, omega=1):

        self.gamma = gamma
        self.delta = delta
        self.m = m
        self.omega = omega

        f = self.get_f()
        firstd = self.get_deriv()

        super().__init__(cell=cell, n=2, f=f, firstd=firstd)

    def get_f(self):
        def v11(x):
            return np.sqrt(2) * self.gamma * x + harmonic(x)

        def v12(x):
            return self.delta / 2 * np.ones_like(x)

        def v22(x):
            return -np.sqrt(2) * self.gamma * x + harmonic(x)

        def harmonic(x):
            return 0.5 * self.m * self.omega**2 * x**2

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return np.sqrt(2)*self.gamma*np.ones_like(x) + harmonic_deriv(x)

        def v12(x):
            return np.zeros_like(x)

        def v22(x):
            return -v11(x)

        def harmonic_deriv(x):
            return self.m * self.omega**2 * x

        return np.array([v11, v12, v12, v22])
