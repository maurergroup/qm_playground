from qmp.integrator.integrator import Integrator
import numpy as np
import copy


class HoppingIntegrator(Integrator):

    def __init__(self, **kwargs):
        Integrator.__init__(self, **kwargs)

        self.basis = self.data.hop.basis
        self.initial_position = copy.copy(self.basis.x)
        self.initial_velocity = copy.copy(self.basis.velocity)
        self.initial_state = copy.copy(self.basis.current_state)

        self.position = self.basis.x
        self.velocity = self.basis.velocity
        self.current_state = self.basis.current_state

    def propagate_density_matrix(self, density_matrix, dt):

        V = self.basis.compute_propagating_hamiltonian()

        def density_matrix_dot(A):
            summation = np.einsum('lj,kl', A, V) - np.einsum('kl,lj', A, V)
            return 1.0 / 1.0j * summation

        k1 = dt * density_matrix_dot(density_matrix)
        k2 = dt * density_matrix_dot(density_matrix + k1/2)
        k3 = dt * density_matrix_dot(density_matrix + k2/2)
        k4 = dt * density_matrix_dot(density_matrix + k3)

        return density_matrix + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    def advance_position(self, dt):
        self.position += self.velocity * dt

    def advance_velocity(self, dt):
        acceleration = self.basis.force[self.current_state] / self.basis.mass
        self.last_velocity = self.velocity
        self.velocity += acceleration * dt

    def rescale_velocity(self, deltaV, desired_state):
        d = self.basis.derivative_coupling[desired_state, self.current_state]
        a = 0.5 * d ** 2 / self.basis.mass
        b = np.dot(self.velocity, d)

        roots = np.roots([a, b, -deltaV])
        gamma = min(roots, key=lambda x: abs(x))
        self.velocity -= gamma * d / self.basis.mass

        # This is the same I think
        # direction = d / np.sqrt(np.dot(d, d))
        # Md = self.basis.mass * direction
        # a = np.dot(Md, Md)
        # b = 2.0 * np.dot(self.basis.mass * self.velocity, Md)
        # c = -2.0 * self.basis.mass * -deltaV
        # roots = np.roots([a, b, c])
        # scal = min(roots, key=lambda x: abs(x))
        # self.velocity += scal * direction

    def execute_hopping(self):

        def get_probabilities(velocity):
            A = self.density_matrix.conj().T
            R = velocity
            d = self.basis.derivative_coupling
            V = self.basis.V

            imag = np.imag(A * V)
            real = np.real(A * R * d)
            B = 2*imag - 2*real

            n = self.current_state
            B = B[:, n]
            g = self.dt * B / self.density_matrix[n, n]
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
            hamiltonian = self.basis.hamiltonian
            old = hamiltonian[self.current_state, self.current_state]
            new = hamiltonian[desired_state, desired_state]
            deltaV = new - old
            kinetic = 0.5 * self.basis.mass * velocity ** 2

            if kinetic >= deltaV:
                self.rescale_velocity(deltaV, desired_state)
                self.current_state = desired_state
            else:
                self.velocity = -1.0 * self.velocity

        velocity = 0.5 * (self.velocity + self.last_velocity)
        g = get_probabilities(velocity)
        can_hop, desired_state = check_possible_hop(g)

        if can_hop:
            attempt_hop(desired_state, velocity)

    def check_finished(self, steps):
        exit = False
        outcome = np.zeros((2, 2))
        if self.current_step > steps:
            print('Steps exceeded')
            exit = True
        elif self.position < self.data.cell[0][0]:
            # print('reflect, state: ' + str(self.current_state))
            outcome[0, self.current_state] = 1
            exit = True
        elif self.position > self.data.cell[0][1]:
            # print('transmit, state: ' + str(self.current_state))
            outcome[1, self.current_state] = 1
            exit = True
        return exit, outcome

    def run_single_trajectory(self, steps=1e5, dt=20.0):

        self.dt = dt
        self.current_step = 0
        acceleration = self.basis.force[self.current_state] / self.basis.mass

        dv = 0.5 * acceleration * dt
        self.last_velocity = self.velocity - dv
        self.velocity = self.velocity + dv

        self.density_matrix = self.propagate_density_matrix(
                                    self.basis.density_matrix,
                                    dt*0.5)

        while(True):

            self.advance_position(dt)

            self.basis.update_electronics(self.position)

            self.advance_velocity(dt)

            self.density_matrix = self.propagate_density_matrix(
                                        self.basis.density_matrix,
                                        dt)

            self.execute_hopping()

            self.current_step += 1

            exit, outcome = self.check_finished(steps)

            if exit:
                return outcome

    def run(self, steps=1e5, dt=20.0):

        result = np.zeros((2, 2))
        ntraj = 2000
        for i in range(2000):

            self.basis.reset_system(self.initial_position,
                                    self.initial_velocity, self.initial_state)
            self.position = self.basis.x
            self.velocity = self.basis.velocity
            self.current_state = self.basis.current_state

            result += self.run_single_trajectory(steps=steps, dt=dt)

        self.data.result = result / ntraj
