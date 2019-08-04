from qmp.integrator.integrator import Integrator
import numpy as np


class HoppingIntegrator(Integrator):

    def __init__(self, dt=2):
        Integrator.__init__(self, dt)

    def propagate_density_matrix(self, density_matrix, dt):
        """ Propagates the density matrix by dt.
        This is taken from smparker's FSSH, I don't know what's going on.
        My attempt was the commented RK4 method that does not appear to
        correctly treat the imaginary part.
        """

        V = self.system.compute_propagating_hamiltonian()

        diags, coeff = np.linalg.eigh(V)
        prop = np.diag(np.exp(-1j * diags * dt))

        U = np.linalg.multi_dot([coeff, prop, coeff.T.conj()])
        return np.dot(U, np.dot(density_matrix, U.T.conj()))

        # This bit doesn't quite work.
        # def density_matrix_dot(A):
        #     summation = np.einsum('lj,kl', A, V) - np.einsum('kl,lj', A, V)
        #     return -1j * summation

        # k1 = dt * density_matrix_dot(density_matrix)
        # k2 = dt * density_matrix_dot(density_matrix + k1/2)
        # k3 = dt * density_matrix_dot(density_matrix + k2/2)
        # k4 = dt * density_matrix_dot(density_matrix + k3)

        # return density_matrix + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    def advance_position(self, dt):
        self.system.r += self.system.v * dt

    def advance_velocity(self, dt):
        acceleration = (self.system.force[self.system.current_state] /
                        self.system.masses)
        self.system.last_velocity = self.system.v
        self.system.v += acceleration * dt

    def rescale_velocity(self, deltaV, desired_state):
        """ Rescales velocity.
        Rescaling is carried out in the direction of the derivative coupling
        after a successful hop.
        """
        d = self.system.derivative_coupling[desired_state,
                                            self.system.current_state]

        direction = d / np.sqrt(np.dot(d, d))
        Md = self.system.masses * direction
        a = np.dot(Md, Md)
        b = 2.0 * np.dot(self.system.masses * self.system.v, Md)
        c = -2.0 * self.system.masses * -deltaV
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.system.v += np.real(scal) * direction

    def get_probabilities(self, velocity):
        """ Calculates the hopping probability.
        Returns an array where prob[i] is the probability of hopping from the
        current state to state i.
        """
        A = self.system.density_matrix
        R = velocity
        d = self.system.derivative_coupling
        n = self.system.current_state

        prob = 2 * np.real(A[n] * R * d[n] / A[n, n]) * self.dt

        prob[n] = 0.0
        prob = prob.clip(0.0, 1.0)

        return prob

    def check_possible_hop(self, g):
        """ Determines whether a hop should occur.
        A random number is compared with the hopping probability, if the
        probability is higher than the random number, a hop occurs. This
        currently only works for a two level system.
        """
        zeta = np.random.uniform()
        for i, prob in enumerate(g):
            if prob > zeta:
                return True, i
        return False, -1

    def attempt_hop(self, desired_state, velocity):
        """ Carries out a hop if the particle has sufficient kinetic energy.
        If the energy is sufficient the velocity is rescaled accordingly,
        otherwise the velocity is reversed.
        """
        hamiltonian = self.system.hamiltonian
        old = hamiltonian[self.system.current_state,
                          self.system.current_state]
        new = hamiltonian[desired_state, desired_state]
        deltaV = new - old
        kinetic = 0.5 * self.system.masses * velocity ** 2

        if kinetic >= deltaV:
            self.rescale_velocity(deltaV, desired_state)
            self.system.current_state = desired_state
        else:
            # Chem. Phys. 349, 334 (2008) suggests reversing the velocity
            # during a frustrated hop but I see not mention of this elsewhere,
            # I'll leave in here in case someone else wants to try it.
            # self.system.v = -1 * self.system.v
            self.system.v = self.system.v

    def execute_hopping(self):
        """ Carries out surface hopping.
        """

        velocity = 0.5 * (self.system.v + self.system.last_velocity)
        g = self.get_probabilities(velocity)
        can_hop, desired_state = self.check_possible_hop(g)

        if can_hop:
            self.attempt_hop(desired_state, velocity)

    def check_finished(self, steps):
        """ Checks criteria for a successful trajectory.
        If the particle has exited the cell the result is recorded and the loop
        exited. Otherwise the loop continues. If the maximum number of steps is
        exceeded the loop is exited but the result is not recorded.
        """
        exit = False
        outcome = np.zeros((2, 2))
        if self.current_step >= steps:
            print('Max steps exceeded, trajectory discarded.')
            self.ntraj -= 1
            exit = True
        elif self.system.r < self.system.potential.cell[0][0]:
            outcome[0, self.system.current_state] = 1
            exit = True
        elif self.system.r > self.system.potential.cell[0][1]:
            outcome[1, self.system.current_state] = 1
            exit = True
        return exit, outcome

    def store_result(self):
        self.r_t[self.current_step] = self.system.r
        self.v_t[self.current_step] = self.system.v

    def run_single_trajectory(self, steps=1e5, dt=20.0):

        self.dt = dt
        self.current_step = 0
        self.r_t[0] = self.system.r
        self.v_t[0] = self.system.v
        acceleration = (self.system.force[self.system.current_state]
                        / self.system.masses)

        dv = 0.5 * acceleration * dt
        self.system.last_velocity = self.system.v - dv
        self.system.v = self.system.v + dv

        self.system.density_matrix = self.propagate_density_matrix(
                                         self.system.density_matrix,
                                         dt*0.5)

        while(True):

            self.current_step += 1

            self.advance_position(dt)

            self.system.update_electronics()

            self.advance_velocity(dt)

            self.system.density_matrix = self.propagate_density_matrix(
                                             self.system.density_matrix,
                                             dt)

            self.execute_hopping()

            exit, outcome = self.check_finished(steps)

            self.store_result()

            if exit:
                self.r_t[self.current_step:] = self.system.r
                self.v_t[self.current_step:] = self.system.v
                break

        return outcome

    def initialise_zero_arrays(self, steps):
        """ Creates empty arrays to house data.
        Currently this is basically unused but could be modified later for
        visualisation purposes.
        """
        self.r_t = np.zeros((steps+1, self.system.ndim))
        self.v_t = np.zeros((steps+1, self.system.ndim))

    def assign_data(self, data, result):
        data.reflect_lower = result[0, 0]
        data.reflect_upper = result[0, 1]
        data.transmit_lower = result[1, 0]
        data.transmit_upper = result[1, 1]

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)
        self.ntraj = kwargs.get('ntraj', 2000)

        self.system = system

        self.initialise_zero_arrays(steps)

        result = np.zeros((2, 2))
        momentum = self.system.initial_v * self.system.masses
        print(f'Running {self.ntraj} surface hopping trajectories'
              + f' for momentum = {momentum}')
        for i in range(self.ntraj):

            self.system.reset_system()

            result += self.run_single_trajectory(steps=steps, dt=dt)

        result = result / self.ntraj
        self.assign_data(data, result)
        print(f'{self.ntraj} successful trajectories completed.')
