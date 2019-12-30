import numpy as np

from .ring_polymer import RingPolymer


class RingPolymerHopping(RingPolymer):

    def __init__(self, coordinates, velocities, masses, initial_state,
                 start_file=None, equilibration_end=None,
                 n_beads=4, T=298,
                 n_states=2):

        super().__init__(coordinates, velocities, masses, n_beads, T)

        self.n_states = n_states
        self.state = initial_state

        if start_file is not None and equilibration_end is not None:
            self.set_position_from_trajectory(start_file, equilibration_end)
