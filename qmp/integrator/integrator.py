from abc import ABC, abstractmethod

import progressbar


class Integrator(ABC):

    def __init__(self, dt=1, output_interval=1):
        """Class is initialised with a timestep as a single argument."""
        self.dt = dt
        self.output_freq = output_interval
        self.system = None
        self.potential = None

    def run(self, system, steps, potential, data, **kwargs):
        """This function is called by the model to run the integration process.

        This function handles the operation of the integrator. The integrator
        is first set up and then the system is integrated for the specified
        number of steps. Finally, the data object is assigned.

        Parameters
        ----------
        system : qmp.systems
        steps : int
        potential : qmp.potential.potential.Potential
        data : qmp.data_containers.Data
        kwargs : {'dt', 'output_freq'}
        """

        self.system = system
        self.potential = potential

        self._read_kwargs(kwargs)
        self._initialise_start()
        self._integrate(steps)
        self._assign_data(data)

    def _read_kwargs(self, kwargs):
        """Allowed keyword arguments are read here.

        Parameters
        ----------
        kwargs : {'dt', 'output_freq'}
        """
        self.dt = kwargs.get('dt', self.dt)
        self.output_freq = kwargs.get('output_freq', 2)

    @abstractmethod
    def _initialise_start(self):
        """Prepare any logging variables and calculate intial values."""

    def _integrate(self, steps):
        """Carry out main integration loop.

        Parameters
        ----------
        steps : int
            The number of steps.
        """
        print('Integrating...')

        for i in progressbar.progressbar(range(steps)):
            try:
                self._perform_timestep(i)
            except SimulationTerminated:
                break

        print('INTEGRATED')

    def _perform_timestep(self, iteration):

        self._propagate_system()

        if (iteration+1) % self.output_freq == 0:
            self._store_result()

    @abstractmethod
    def _propagate_system(self):
        """Propagate the system by a single timestep."""

    @abstractmethod
    def _store_result(self):
        """Store the results of the current step."""

    @abstractmethod
    def _assign_data(self, data):
        """Assign the data at the end of the simulation."""


class SimulationTerminated(Exception): pass
