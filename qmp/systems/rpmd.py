import numpy as np
from qmp.systems.phasespace import PhaseSpace
from qmp.tools.dyn_tools import get_v_maxwell
import scipy.sparse as sp
import scipy.linalg as la
from numpy.fft import fft, ifft, fftfreq


class RPMD(PhaseSpace):

    def __init__(self, coordinates, velocities, masses, n_beads=4,
                 T=298, init_type='velocity'):
        """Initialise the ring polymer system.

        Parameters
        ----------
        coordinates : 2-D array
            Coordinates representing the ring polymer centroid.
        velocities : 2-D array
            Velocities of the ring polymer centroids.
        masses : 1-D array
            Masses for each particle.
        n_beads : int
            Number of beads in each ring polymer.
        T : float or int
            Temperature of the ring polymer.
        """

        PhaseSpace.__init__(self, coordinates, velocities, masses)

        if (n_beads == 0) or (type(n_beads) != int):
            print('0 and lists are not allowed for number of beads,'
                  + 'using n_beads = 4 per default')
            self.n_beads = 4
        else:
            self.n_beads = n_beads

        self.temp = T / 3.15777504e5
        self.beta = 1 / (self.temp * self.n_beads)
        self.omega = 1 / (self.beta)
        self.omega = self.temp * self.n_beads
        self.init_type = init_type

        self.initialise_beads()

        self.initialise_normal_frequencies()
        self.initialise_transformer()

    def initialise_beads(self):
        xi = 2.*np.pi/self.n_beads
        rotMat = np.array([[np.cos(xi), np.sin(xi)],
                          [-np.sin(xi), np.cos(xi)]])
        arand = np.random.random(1)*2.*np.pi
        rM = np.array([[np.cos(arand), np.sin(arand)],
                      [-np.sin(arand), np.cos(arand)]])

        self.r_beads = np.zeros((self.n_particles, self.n_beads, self.ndim))
        self.v_beads = np.zeros((self.n_particles, self.n_beads, self.ndim))

        if self.init_type == 'velocity':
            self.velocity_init(rotMat, rM)
        elif self.init_type == 'position':
            self.position_init(rotMat, rM)
        else:
            raise ValueError('Init type not recognised')

        self.r = self.r_beads
        self.v = self.v_beads

    def velocity_init(self, rotMat, rM):
        for i_par in range(self.n_particles):
            if self.ndim == 1:
                i_start = self.n_beads % 2
                # for odd number of beads:
                # (n_beads-1)/2 pairs, v=0 for last bead ~> start at i_start=1
                for i_bead in range(i_start, self.n_beads, 2):
                    v_mb = get_v_maxwell(self.masses[i_par],
                                         self.temp)
                    # assign v_p + v_mb to bead1 of pair
                    self.v_beads[i_par, i_bead] = self.v[i_par] + v_mb
                    # assign v_p - v_mb to bead2 of pair
                    self.v_beads[i_par, i_bead+1] = self.v[i_par] - v_mb
            else:
                v_abs = get_v_maxwell(self.masses[i_par], self.temp)
                v_vec = np.dot(np.array([0., v_abs]), rM).T
                for i_bead in range(self.n_beads):
                    self.v_beads[i_par, i_bead] = self.v[i_par] + v_vec
                    v_vec = np.dot(v_vec, rotMat)

            for i_bead in range(self.n_beads):
                self.r_beads[i_par, i_bead] = self.r[i_par]

    def position_init(self, rotMat, rM):
        for i_par in range(self.n_particles):
            r_abs = (np.pi)*np.sqrt(
                self.n_beads/(2.*self.masses[i_par]*self.temp))/400.
            if self.ndim == 1:
                self.r_beads[i_par, 0] = self.r[i_par] - r_abs
                for i_bead in range(1, self.n_beads):
                    self.r_beads[i_par,
                                 i_bead] = (self.r_beads[i_par, i_bead-1]
                                            + (2.*r_abs/(self.n_beads-1)))
            else:
                r_vec = np.dot(np.array([0., r_abs]), rM).T
                self.r_beads[i_par, 0] = self.r[i_par] + r_vec
                for i_bead in range(1, self.n_beads):
                    r_vec = np.dot(r_vec, rotMat)
                    self.r_beads[i_par, i_bead] = self.r[i_par] + r_vec

            for i_bead in range(self.n_beads):
                self.v_beads[i_par, i_bead] = self.v[i_par]

    def compute_kinetic_energy(self):
        return 0.5 * np.einsum('i,ijk->ij', self.masses, self.v ** 2)

    def compute_bead_potential_energy(self, potential):
        if self.ndim == 1:
            return potential(self.r[:, :, 0])
        elif self.ndim == 2:
            return potential(self.r[:, :, 0], self.r[:, :, 1])

    def compute_potential_energy(self, potential):
        """Compute the potential energy for each bead."""
        M = np.eye(self.n_beads) - np.diag(np.ones(self.n_beads-1), 1)
        M[-1, 0] = -1.

        bead = self.compute_bead_potential_energy(potential)

        a = np.sum(np.einsum('ij,xjk->xik', M, self.r)**2, 2)
        b = 0.5 * self.masses[:, np.newaxis] * self.omega ** 2
        spring = a * b
        return bead + spring

    def compute_force(self, potential):
        """Compute the force from the potential on the bead."""
        force = np.empty_like(self.r)
        for i, particle in enumerate(self.r):
            force[i] = -1 * potential.deriv(particle)
        return force

    def compute_acceleration(self, potential):
        F = self.compute_force(potential)
        self.acceleration = F / self.masses[:, np.newaxis, np.newaxis]

    def compute_omega_Rugh(self, potential):
        """Definition of dynamical temperature according to Rugh.

        = (n_beads/hbar)*kB*T
        = (n_beads/hbar)*( (|V'(r)|^2 + |v|^2) / (V"(r) + ndim/m) )
        """
        A = la.norm(PhaseSpace.compute_force(self, potential))
        a = self.compute_hessian(potential)
        v = la.norm(self.v)

        self.omega = (self.n_beads)*((A**2 + v**2)/(a+(self.ndim/self.masses)))

    def compute_hessian(self, potential):
        r = np.mean(self.r, 1)
        return potential.hess(r)

    def propagate_positions(self, dt):

        self.transform_to_normal_modes()
        self.propagate_free_polymer()
        self.transform_from_normal_modes()

    def initialise_transformer(self):
        c = np.zeros((self.n_beads, self.n_beads))
        j = np.arange(1, self.n_beads+1)

        c[:, 0] = 1
        lower_range = np.arange(1, self.n_beads//2)
        upper_range = np.arange(self.n_beads//2+1, self.n_beads)

        for k in lower_range:
            c[:, k] = np.sqrt(2)*np.cos(2*np.pi*j*k/self.n_beads)
        for k in upper_range:
            c[:, k] = np.sqrt(2)*np.sin(2*np.pi*j*k/self.n_beads)

        if self.n_beads % 2 == 0:
            c[:, self.n_beads//2] = (-1)**j

        self.transformer = c / np.sqrt(self.n_beads)

    def initialise_normal_frequencies(self):
        k = np.arange(self.n_beads)
        self.omega_k = 2. * self.omega * np.sin(k*np.pi/self.n_beads)

    def initialise_propagators(self, dt):
        square = (self.omega_k * dt / 2)**2
        self.cayley_11 = (1 - square) / (1 + square)
        self.cayley_12 = -(4 * square / dt) / (1 + square) * self.masses[..., None]
        self.cayley_21 = dt / (1 + square) / self.masses[..., None]

    def transform_to_normal_modes(self):
        self.p_normal = np.zeros_like(self.v)
        self.q_normal = np.zeros_like(self.r)

        self.p_normal = self.transformer.T @ self.v * self.masses[..., None, None]
        self.q_normal = self.transformer.T @ self.r

    def transform_from_normal_modes(self):
        self.v = np.zeros_like(self.p_normal)
        self.r = np.zeros_like(self.q_normal)

        self.v = self.transformer @ self.p_normal / self.masses[..., None, None]
        self.r = self.transformer @ self.q_normal

    def propagate_free_polymer(self):
        old_p = self.p_normal
        old_q = self.q_normal
        self.p_normal = self.cayley_11[..., None] * old_p + old_q * self.cayley_12[..., None]
        self.q_normal = self.cayley_11[..., None] * old_q + old_p * self.cayley_21[..., None]
