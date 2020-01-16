import copy

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

        self.outcome = np.zeros((self.system.n_states, 2))
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
        data.state_t = np.array(self.state_t)
        data.r_t = np.array(self.r_t)
        data.v_t = np.array(self.v_t)
        self._save_potential(data)

    def _store_result(self):
        """Store trajectory data but this is currently not used."""
        self.r_t.append(self.system.r)
        self.v_t.append(self.system.v)
        self.state_t.append(self.system.state)

    def _save_potential(self, data):
        data.v11, data.v12, data.v22 = self.potential.compute_cell_potential(density=1000)

class RingPolymerFSSH(FSSH):

    def _initialise_start(self):
        super()._initialise_start()
        self.system.initialise_propagators(self.dt)

        self.state_occ_t = [copy.copy(self.system.state_occupations)]
        self.E_pot = [self.system.compute_potential_energy(self.potential)]
        self.E_kin = [self.system.compute_kinetic_energy()]

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

    def _assign_data(self, data):
        """Add to the cumulative outcome."""
        super()._assign_data(data)
        data.state_occ_t = np.array(self.state_occ_t)
        data.E_kin_t = np.mean(np.array(self.E_kin), axis=2)
        data.E_pot_t = np.mean(np.array(self.E_pot), axis=2)
        data.E_kink_t = np.zeros_like(data.E_pot_t)
        data.E_t = data.E_kin_t + data.E_pot_t

    def _store_result(self):
        """Store trajectory data but this is currently not used."""
        super()._store_result()
        self.state_occ_t.append(copy.copy(self.system.state_occupations))
        self.E_pot.append(self.system.compute_potential_energy(self.potential))
        self.E_kin.append(self.system.compute_kinetic_energy())

    def _is_finished(self):
        quit = False
        if np.mean(self.system.r) < -15:
            self.outcome[self.system.state, 0] = 1
            quit = True
        elif np.mean(self.system.r) > 15:
            self.outcome[self.system.state, 1] = 1
            quit = True
        return quit

class RingPolymerEquilibriumHopping(RingPolymerFSSH):

    def _initialise_start(self):
        super()._initialise_start()
        self.E_kink = [self.system.compute_kink_potential()]

    def _propagate_system(self):
        """Propagate the system by a single timestep.

        This function carries out the shortened form of the velocity verlet
        algorithm. It requires that the systems taking advantage of this
        integrator implement the three functions used within it.
        """
        self.system.execute_hopping(self.dt*0.5)

        self.system.propagate_velocities(self.dt*0.5)
        self.system.propagate_positions()
        self.system.compute_acceleration(self.potential)
        self.system.propagate_velocities(self.dt*0.5)

        self.system.execute_hopping(self.dt*0.5)

    def _store_result(self):
        """Store trajectory data but this is currently not used."""
        super()._store_result()
        self.E_kink.append(self.system.compute_kink_potential())

    def _assign_data(self, data):
        """Add to the cumulative outcome."""
        super()._assign_data(data)
        data.state_occ_t = np.array(self.state_occ_t)
        data.E_kin_t = np.mean(np.array(self.E_kin), axis=2)
        data.E_pot_t = np.mean(np.array(self.E_pot), axis=2)
        data.E_kink_t = np.array(self.E_kink)
        data.E_t = data.E_kin_t + data.E_pot_t + data.E_kink_t
