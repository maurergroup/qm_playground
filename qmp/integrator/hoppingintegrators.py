from qmp.integrator.integrator import Integrator
import numpy as np
import copy


class HoppingIntegrator(Integrator):

    def __init__(self, dt=20):
        Integrator.__init__(self, dt)

    def propagate_density_matrix(self, density_matrix, dt):

        V = self.system.compute_propagating_hamiltonian()

        def density_matrix_dot(A):
            summation = np.einsum('lj,kl', A, V) - np.einsum('kl,lj', A, V)
            return 1.0 / 1.0j * summation

        k1 = dt * density_matrix_dot(density_matrix)
        k2 = dt * density_matrix_dot(density_matrix + k1/2)
        k3 = dt * density_matrix_dot(density_matrix + k2/2)
        k4 = dt * density_matrix_dot(density_matrix + k3)

        return density_matrix + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    def advance_position(self, dt):
        self.system.r += self.system.v * dt

    def advance_velocity(self, dt):
        acceleration = (self.system.force[self.system.current_state] /
                        self.system.masses)
        self.system.last_velocity = self.system.v
        self.system.v += acceleration * dt

    def rescale_velocity(self, deltaV, desired_state):
        d = self.system.derivative_coupling[desired_state,
                                            self.system.current_state]
        a = 0.5 * d ** 2 / self.system.masses
        b = np.dot(self.system.v, d)

        roots = np.roots([a, b, -deltaV])
        gamma = min(roots, key=lambda x: abs(x))
        self.system.v -= np.real(gamma) * d / self.system.masses

        # This is the same I think
        # direction = d / np.sqrt(np.dot(d, d))
        # Md = self.system.mass * direction
        # a = np.dot(Md, Md)
        # b = 2.0 * np.dot(self.system.mass * self.velocity, Md)
        # c = -2.0 * self.system.mass * -deltaV
        # roots = np.roots([a, b, c])
        # scal = min(roots, key=lambda x: abs(x))
        # self.velocity += scal * direction

    def execute_hopping(self):

        def get_probabilities(velocity):
            A = self.system.density_matrix.conj().T
            R = velocity
            d = self.system.derivative_coupling
            V = self.system.V

            imag = np.imag(A * V)
            real = np.real(A * R * d)
            B = 2*imag - 2*real

            n = self.system.current_state
            B = B[:, n]
            g = self.dt * B / self.system.density_matrix[n, n]
            g[n] = 0.0
            g.clip(0.0, 1.0)

            return g

        def check_possible_hop(g):
            zeta = np.random.uniform()
            for i, prob in enumerate(g):
                if prob > zeta:
                    return True, i
            return False, -1

        def attempt_hop(desired_state, velocity):
            hamiltonian = self.system.hamiltonian
            old = hamiltonian[self.system.current_state, self.system.current_state]
            new = hamiltonian[desired_state, desired_state]
            deltaV = new - old
            kinetic = 0.5 * self.system.masses * velocity ** 2

            if kinetic >= deltaV:
                self.rescale_velocity(deltaV, desired_state)
                self.system.current_state = desired_state
            else:
                self.system.v = -1.0 * self.system.v

        velocity = 0.5 * (self.system.v + self.system.last_velocity)
        g = get_probabilities(velocity)
        can_hop, desired_state = check_possible_hop(g)

        if can_hop:
            attempt_hop(desired_state, velocity)

    def check_finished(self, steps):
        exit = False
        outcome = np.zeros((2, 2))
        if self.current_step >= steps:
            print('Max steps exceeded, trajectory discarded.')
            exit = True
        elif self.system.r < self.system.potential.cell[0][0]:
            # print('reflect, state: ' + str(self.current_state))
            outcome[0, self.system.current_state] = 1
            exit = True
        elif self.system.r > self.system.potential.cell[0][1]:
            # print('transmit, state: ' + str(self.current_state))
            outcome[1, self.system.current_state] = 1
            exit = True
        return exit, outcome

    def store_result(self, count):
        self.r_t[count, self.current_step] = self.system.r
        self.v_t[count, self.current_step] = self.system.v

    def run_single_trajectory(self, ntraj, steps=1e5, dt=20.0):

        self.dt = dt
        self.current_step = 0
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

            self.store_result(ntraj)

            if exit:
                self.r_t[ntraj, self.current_step:] = self.system.r
                self.v_t[ntraj, self.current_step:] = self.system.v
                break

        return outcome

    def initialise_zero_arrays(self, ntraj, steps, N, ndim, data):
        self.r_t = np.zeros((ntraj, steps+1, N, ndim))
        self.v_t = np.zeros((ntraj, steps+1, N, ndim))
        self.r_t[:, 0] = self.system.r
        self.v_t[:, 0] = self.system.v

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)
        ntraj = kwargs.get('ntraj', 2000)

        self.system = system
        ndim = self.system.ndim
        N = self.system.n_particles

        self.initialise_zero_arrays(ntraj, steps, N, ndim, data)

        result = np.zeros((2, 2))
        momentum = self.system.initial_v * self.system.masses
        print(f'Running {ntraj} surface hopping trajectories for momentum = {momentum}')
        for i in range(ntraj):

            self.system.reset_system()

            result += self.run_single_trajectory(i, steps=steps, dt=dt)

        result = result / ntraj
        data.reflect_lower = result[0, 0]
        data.reflect_upper = result[0, 1]
        data.transmit_lower = result[1, 0]
        data.transmit_upper = result[1, 1]
        data.r_t = self.r_t
        data.v_t = self.v_t
