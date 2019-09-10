import numpy as np
import scipy.sparse as sp
from numpy.fft import fftn
from numpy.fft import ifftn


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
        self.rho = np.zeros(size)

        self.exit_flux = np.zeros((self.nstates, 2))

    def create_mesh(self):

        axes = [np.linspace(self.start[i], self.end[i], self.N, retstep=True)
                for i in range(len(self.start))]

        axes, steps = list(zip(*axes))
        self.steps = np.array(steps)
        self.mesh = np.meshgrid(*axes)

    def construct_hamiltonian(self, potential):
        self.construct_V_matrix(potential)
        self.construct_coordinate_T()
        self.H = self.V + self.coordinate_T

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

    def construct_coordinate_T(self):
        self.define_laplacian()
        self.coordinate_T = - self.L / (2 * self.mass)

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
        """
        self.compute_absorption_parameters()

        c = 2.62206
        r2 = self.end[0]
        r1 = r2 - c / (2 * self.delta * self.k_min)

        r = self.mesh[0]
        if self.ndim == 2:
            r = r[0]
        imag = Grid.imaginary_potential(r, r1, r2, self.delta,
                                        self.k_min, self.E_min)

        if self.ndim == 2:
            stacked = np.vstack([imag] * self.N)
            transpose = stacked.T
            new = stacked + transpose
            imag = new.flatten()
        self.imag = np.tile(imag, self.nstates)

    def compute_absorption_parameters(self):
        self.compute_k()
        self.E_min = self.compute_kinetic_energy() / 3
        self.k_min = np.sqrt(2 * self.mass * self.E_min)

    def set_initial_wvfn(self, psi, n=1):
        if n == 1:
            self.psi[:self.N**self.ndim] = np.array(psi).flatten()
        elif n == 2:
            self.psi[self.N**self.ndim:] = np.array(psi).flatten()
        self.total_initial_density = np.sum(self.compute_diabatic_density())

    def compute_diabatic_density(self):
        return np.real(self.psi.conj() * self.psi)

    def detect_all(self):
        """ After the simulation has finished, detect the remaining density
        to give a result for transmission and reflection probabilities.
        """
        current_density = self.compute_adiabatic_density()
        self.detect_flux(current_density)

    def detect_flux(self, density):
        splits = np.array(np.split(density, 2*self.nstates))
        probabilities = np.sum(splits, axis=1).reshape(self.nstates, 2)
        self.exit_flux += probabilities

    def normalise_probabilities(self):
        norm = np.sum(self.exit_flux)
        self.exit_flux /= norm

    def get_adiabatic_wavefunction(self):
        psi = np.zeros_like(self.psi)
        for i in range(self.N):
            psi_current = np.array([self.psi[i],
                                   self.psi[i+self.N]])
            psi[i], psi[i+self.N] = self.U[i].dot(psi_current)
        return psi

    def compute_adiabatic_density(self):
        psi = self.psi
        if self.nstates == 2:
            psi = self.get_adiabatic_wavefunction()
        return np.real(psi.conj() * psi)

    def compute_k(self):
        from numpy.fft import fftfreq as FTp

        k = 2 * np.pi * FTp(self.N, self.steps[0])
        if self.ndim == 2:
            """ TODO Pretty sure this is wrong. """
            k = k ** 2
            k = np.kron(np.ones(self.N), k) + np.kron(k, np.ones(self.N))
        elif self.ndim > 2:
            raise NotImplementedError('Only 1D and 2D systems implemented')
        self.k = np.tile(k, self.nstates)
        self.momentum_T = self.k**2 / (2*self.mass)

    def compute_kinetic_energy(self):
        T_dot_psi = self.momentum_T * self.transform(self.psi, fftn)
        E_kin = np.real(self.psi.conj().dot(self.transform(T_dot_psi, ifftn)))
        return E_kin

    def compute_potential_energy(self):
        E_pot = np.real(self.psi.conj().dot(self.V.dot(self.psi)))
        return E_pot

    def transform(self, psi, transform):
        size = self.nstates * self.N ** self.ndim
        split = np.array(np.split(psi, self.nstates))
        axes = [-1]
        if self.ndim == 2:
            split = split.reshape((self.nstates, self.N, self.N))
            axes = [-2, -1]
        psi_transformed = transform(split, axes=axes).reshape(size)
        return psi_transformed

    @staticmethod
    def imaginary_potential(r, r1, r2, delta, k_min, E_min):

        c = 2.62206
        a = 1 - 16/c**3
        b = (1 - 17/c**3) / c**2

        def y(x):
            return a*x - b*x**3 + 4/(c-x)**2 - 4/(c+x)**2

        x = 2 * delta * k_min * (r - r1)

        after_start = r1 < r
        before_end = r2 > r
        is_inside_region = np.logical_and(after_start, before_end)
        function = np.piecewise(x.astype(dtype=np.complex),
                                is_inside_region,
                                [y])
        function[-1] = 1000
        end_zone = -1j * E_min * function

        start_zone = end_zone[::-1]

        return start_zone + end_zone
