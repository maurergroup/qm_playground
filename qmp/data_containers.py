#    qmp.data_containers.py
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
data_containers.py

collection of different data containers for different
job classes such as wavefunction, particle, and RPMD necklace
"""


class Data(dict):
    """
    Base class for all data containers
    """
    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
