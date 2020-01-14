import pickle

import numpy as np
import scipy.linalg as la

from qmp.systems.phasespace import PhaseSpace
from qmp.tools.dyn_tools import atomic_to_kelvin


class RingPolymer(PhaseSpace):

    def __init__(self, coordinates, velocities, masses, n_beads=4, T=298):
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

        self.temp = T / atomic_to_kelvin
        self.beta = 1 / (self.temp * self.n_beads)
        self.omega = self.temp * self.n_beads

        self.initialise_beads()
        self.initialise_normal_frequencies()
        self.initialise_transformer()

    def initialise_beads(self):

        self.r_beads = np.zeros((self.n_particles, self.n_beads, self.ndim))
        self.v_beads = np.zeros((self.n_particles, self.n_beads, self.ndim))

        for i_par in range(self.n_particles):
            self.r_beads[i_par] = self.r[i_par]
            self.v_beads[i_par] = self.v[i_par]

        self.r = self.r_beads
        self.v = self.v_beads

    def compute_kinetic_energy(self):
        return 0.5 * np.einsum('i,ijk->ij', self.masses, self.v ** 2)

    def _compute_bead_potential_energy(self, potential):
        if self.ndim == 1:
            pot = potential(self.r[:, :, 0])
        elif self.ndim == 2:
            pot = potential(self.r[:, :, 0], self.r[:, :, 1])
        return pot

    def compute_potential_energy(self, potential):
        """Compute the potential energy for each bead."""
        bead = self._compute_bead_potential_energy(potential)
        spring = self.compute_spring_potential()
        return bead + spring

    def compute_spring_potential(self):
        if self.n_beads == 1:
            spring = 0
        else:
            M = np.eye(self.n_beads) - np.diag(np.ones(self.n_beads-1), 1)
            M[-1, 0] = -1.

            a = np.sum(np.einsum('ij,xjk->xik', M, self.r)**2, 2)
            b = 0.5 * self.masses[:, np.newaxis] * self.omega ** 2
            spring = a * b
        return spring

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

    def propagate_positions(self):

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
        self.p_normal = self.transformer.T @ self.v * self.masses[..., None, None]
        self.q_normal = self.transformer.T @ self.r

    def transform_from_normal_modes(self):
        self.v = self.transformer @ self.p_normal / self.masses[..., None, None]
        self.r = self.transformer @ self.q_normal

    def propagate_free_polymer(self):
        old_p = self.p_normal
        old_q = self.q_normal
        self.p_normal = self.cayley_11[..., None] * old_p + old_q * self.cayley_12[..., None]
        self.q_normal = self.cayley_11[..., None] * old_q + old_p * self.cayley_21[..., None]

    def initialise_thermostat(self, dt, tau0=1, ignore_centroid=False):
        self.gamma_k = 2 * self.omega_k
        self.gamma_k[0] = 1 / tau0

        self.c1 = np.exp(-dt * self.gamma_k / 2)
        self.c2 = np.sqrt(1 - self.c1 ** 2)
        if ignore_centroid:
            self.c1[0] = 0
            self.c2[0] = 0

    def apply_thermostat(self):
        randoms = np.random.normal(size=self.r.shape)
        self.transform_to_normal_modes()
        self.p_normal = self.c1[None, :, None] * self.p_normal
        self.p_normal += (np.sqrt(self.masses[:, None, None] / self.beta)
                          * self.c2[None, :, None] * randoms)
        self.transform_from_normal_modes()

    def set_position_from_trajectory(self, file, equilibration_end):
        file = open(file, 'rb')
        trajectory = pickle.load(file)[0]
        positions = trajectory['rb_t']
        velocities = trajectory['vb_t']

        choice = np.random.randint(low=equilibration_end, high=len(positions))

        self.r = positions[choice]
        self.v = velocities[choice]
