#    qmp.integrator.integrator
#
#    qm_playground - python package for dynamics simulations
#    Copyright (C) 2016  Reinhard J. Maurer
#
#    This file is part of qm_playground.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#
"""
integrator.py
"""


class Integrator:
    """
    Base class for all integrators. All integrators
    need to be initialized with basis, potential and data
    object.

    Every integrator needs the following:
     __init__,
     run,
     get_forces,
     integrate

    """

    def __init__(self, dt):

        self.dt = dt

    def run(self, steps):
        """
        Placeholder for run function of subclasses.
        """

        raise NotImplementedError('run needs to be implemented by \
                the subclasses!')
        pass

    def get_forces(self):
        """
        Placeholder for get_forces function of subclasses.
        """

        raise NotImplementedError('get_forces needs to be implemented by \
                the subclasses!')
        pass

    def integrate(self):
        """
        Placeholder for integrate function of subclasses.
        """

        raise NotImplementedError('integrate needs to be implemented by \
                the subclasses!')
        pass
