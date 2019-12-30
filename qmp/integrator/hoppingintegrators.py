import numpy as np

from .integrator import Integrator


class FSSH(Integrator):
    """Single trajectory surface hopping.

    This class must be used in conjunction with a managing class, otherwise the
    outcome attribute of the data object has not been initilised.
    """

    def _initialise_start(self):
        """Reset system and prepare logging variables."""
        self.r_t = [self.system.r]
        self.v_t = [self.system.v]
        self.state_t = [self.system.state]

        self.outcome = np.zeros((self.system.nstates, 2))
        self.current_acc = self.system.compute_acceleration(self.potential)

    def _propagate_system(self):
        """Propagate the system by a single timestep.

        This function carries out the shortened form of the velocity verlet
        algorithm. It requires that the systems taking advantage of this
        integrator implement the three functions used within it.
        """
        self.system.propagate_velocities(self.dt*0.5)
        self.system.propagate_positions(self.dt)
        self.system.compute_acceleration(self.potential)
        self.system.propagate_velocities(self.dt*0.5)

        self.system.propagate_density_matrix(self.dt)
        self.system.execute_hopping(self.dt)

    def _is_finished(self):
        """ Checks criteria for a successful trajectory.

        Check if the particle has left the cell, if so, the outcome is
        recorded.
        """
        quit = False
        if self.system.has_reflected():
            self.outcome[self.system.state, 0] = 1
            quit = True
        elif self.system.has_transmitted():
            self.outcome[self.system.state, 1] = 1
            quit = True
        return quit

    def _assign_data(self, data):
        """Add to the cumulative outcome."""
        data.outcome = self.outcome
        data.state_t = self.state_t

    def _store_result(self):
        """Store trajectory data but this is currently not used."""
        self.r_t.append(self.system.r)
        self.v_t.append(self.system.v)
        self.state_t.append(self.system.state)


class RingPolymerFSSH(FSSH):

    def _initialise_start(self):
        super()._initialise_start()
        self.system.initialise_propagators(self.dt)

    def _propagate_system(self):
        """Propagate the system by a single timestep.

        This function carries out the shortened form of the velocity verlet
        algorithm. It requires that the systems taking advantage of this
        integrator implement the three functions used within it.
        """
        self.system.propagate_velocities(self.dt*0.5)
        self.system.propagate_positions()
        self.system.compute_acceleration(self.potential)
        self.system.propagate_velocities(self.dt*0.5)

        self.system.propagate_density_matrix(self.dt)
        self.system.execute_hopping(self.dt)
