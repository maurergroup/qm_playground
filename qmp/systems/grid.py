import numpy as np
import scipy.sparse as sp


class Grid:

    def __init__(self, mass=1800, cell=[[-10, 10]], N=256, states=1, **kwargs):

        self.mass = mass
        self.N = N
        self.cell = cell
        self.nstates = states
        self.start = cell[:, 0]
        self.end = cell[:, 1]
        self.ndim = len(self.start)
        self.solved = False

        self.delta = kwargs.get('delta', 0.2)

        self.create_mesh()

        # Initial wavefunction all zeros
        size = (self.nstates * self.N ** self.ndim)
        self.psi = np.zeros(size, dtype=complex)
        self.imag = np.zeros_like(self.psi)
        self.rho = np.zeros(size)

        self.absorbed_density = np.zeros((self.nstates * 2))

    def create_mesh(self):

        axes = [np.linspace(self.start[i], self.end[i], self.N, retstep=True)
                for i in range(len(self.start))]

        axes, steps = list(zip(*axes))
        self.steps = np.array(steps)
        self.mesh = np.meshgrid(*axes)

    def construct_hamiltonian(self, potential):
        self.construct_V_matrix(potential)
        self.construct_T_matrix()
        self.H = self.V + self.T

    def construct_V_matrix(self, potential):

        try:
            v11 = self.compute_diagonal_potential(potential, i=0, j=0)
            v12 = self.compute_diagonal_potential(potential, i=0, j=1)
            v22 = self.compute_diagonal_potential(potential, i=1, j=1)
            self.V = sp.bmat([[v11, v12], [v12, v22]])

            V = self.V.A
            self.U = np.zeros((self.N, 2, 2))
            for i in range(self.N):
                mat = [[V[i, i], V[i, i+self.N]],
                       [V[i+self.N, i], V[i+self.N, i+self.N]]]
                w, v = np.linalg.eigh(mat)
                self.U[i] = v

        except IndexError:
            self.V = v11

    def compute_diagonal_potential(self, potential, i, j):
        vflat = potential(*self.mesh, i=i, j=j).flatten()
        return sp.diags(vflat)

    def construct_T_matrix(self):
        self.define_laplacian()
        self.T = - self.L / (2 * self.mass)

    def define_laplacian(self):
        L = self.get_1D_laplacian()

        if self.ndim == 2:
            L = sp.kronsum(L, L, format='lil')

        L[0, -1] = 1.
        L[-1, 0] = 1.
        self.L = L / (self.steps ** 2).prod()
        self.L = sp.block_diag([self.L] * self.nstates)

    def get_1D_laplacian(self):
        off_diagonal = np.ones(self.N-1)
        diagonal = np.full(self.N, -2)
        L = sp.diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1])
        return L.tolil()

    def construct_imaginary_potential(self):
        """
        Calculates imaginary potential as described by Manolopoulos
        in JCP 117, 9552 (2002).
        Currently only implemented for 1D.
        Clipping added at the end to allow for Kosloff implementation instead
        of adding to the hamiltonian as an imaginary potential, this allows for
        easier recording of the transmission/reflection probabilities.
        """

        def imaginary_potential(r):

            def y(x):
                return a*x - b*x**3 + 4/(c-x)**2 - 4/(c+x)**2

            def result(x):
                return self.E_min * y(x)

            c = 2.62206
            a = 1 - 16/c**3
            b = 1 - 17/c**3

            k_min = np.sqrt(2 * self.mass * self.E_min)
            r2 = self.end[0]
            r1 = r2 - c / (2 * self.delta * k_min)
            x = 2 * self.delta * k_min * (r - r1)

            after_start = r1 < r
            before_end = r2 > r
            is_inside_region = np.logical_and(after_start, before_end)
            end_zone = np.piecewise(x.astype(dtype=np.complex),
                                    is_inside_region,
                                    [result])

            start_zone = end_zone[::-1]

            return np.clip(start_zone + end_zone, 0, 1)

        self.imag = np.tile(imaginary_potential(self.mesh[0]), self.nstates)

    def set_initial_wvfn(self, psi):
        self.psi[:self.N**self.ndim] = np.array(psi).flatten()

        self.total_initial_density = np.sum(self.compute_probability_density())
        self.compute_initial_kinetic_energy()
        if self.ndim == 1:
            self.construct_imaginary_potential()

    def compute_probability_density(self):
        return np.real(self.psi.conj() * self.psi)

    def compute_initial_kinetic_energy(self):
        self.construct_T_matrix()
        self.E_min = np.real(self.psi.conj().dot(self.T.dot(self.psi))) / 10

    def absorb_boundary(self, dt):
        """ Applies absorbing boundaries as described by Kosloff and Kosloff
        in JCP 63, 363-367 (1986).
        """
        density_before = self.compute_adiabatic_density()
        self.psi = (1 - self.imag * dt) * self.psi
        density_after = self.compute_adiabatic_density()
        absorbed_density = density_before - density_after

        self.store_absorption_increase(absorbed_density)

    def absorb_all(self):
        """ After the simulation has finished, "absorb" the remaining density
        to give a result for transmission and reflection probabilities.
        """
        current_density = self.compute_adiabatic_density()
        self.store_absorption_increase(current_density)
        self.normalise_probabilities()

    def store_absorption_increase(self, density):
        splits = np.array(np.split(density, self.nstates * 2))
        probabilities = np.sum(splits, axis=1).real
        self.absorbed_density += probabilities

    def normalise_probabilities(self):
        norm = np.sum(self.absorbed_density)
        self.absorbed_density /= norm

    def get_adiabatic_wavefunction(self):
        psi = np.zeros_like(self.psi)
        for i in range(self.N):
            psi_current = np.array([self.psi[i],
                                   self.psi[i+self.N]])
            psi[i], psi[i+self.N] = self.U[i].dot(psi_current)
        return psi

    def compute_adiabatic_density(self):
        psi = self.get_adiabatic_wavefunction()
        return np.real(psi.conj() * psi)
