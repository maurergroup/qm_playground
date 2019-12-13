import numpy as np

from .potential import Potential


class SpinBoson(Potential):

    def __init__(self, cell=[[-10, 10]], gamma=1, delta=1, m=1, omega=1, alpha=0):

        self.gamma = gamma
        self.delta = delta
        self.m = m
        self.omega = omega
        self.alpha = alpha

        f = self.get_f()
        firstd = self.get_deriv()

        super().__init__(cell=cell, n=2, f=f, firstd=firstd)

    def get_f(self):
        def v11(x):
            return self.alpha + self.gamma * x + harmonic(x)

        def v12(x):
            return self.delta * np.ones_like(x)

        def v22(x):
            return -self.alpha - self.gamma * x + harmonic(x)

        def harmonic(x):
            return 0.5 * self.m * self.omega**2 * x**2

        return np.array([v11, v12, v12, v22])

    def get_deriv(self):
        def v11(x):
            return self.gamma*np.ones_like(x) + harmonic_deriv(x)

        def v12(x):
            return np.zeros_like(x)

        def v22(x):
            return -self.gamma*np.ones_like(x) + harmonic_deriv(x)

        def harmonic_deriv(x):
            return self.m * self.omega**2 * x

        return np.array([v11, v12, v12, v22])

    def compute_cell_potential(self, density=100):
        x = np.linspace(self.cell[0][0], self.cell[0][1], density)
        v11 = self(x, i=0, j=0)
        v22 = self(x, i=1, j=1)
        v12 = self(x, i=1, j=0)

        return v11, v12, v22
