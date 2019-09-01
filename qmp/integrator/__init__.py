#    qmp.integrator.__init__
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
Integrators
"""

from .waveintegrators import PrimitivePropagator, EigenPropagator
from .waveintegrators import SOFT_Propagator
from .trajintegrators import VelocityVerlet, Langevin
from .rpmdintegrators import RPMD_VelocityVerlet
from .hoppingintegrators import HoppingIntegrator

# integrator_type = {
#     'primitive': wave.PrimitivePropagator,
#     'eigen': wave.EigenPropagator,
#     'SOFT': wave.SOFT_Propagator,
#     # 'SOFT_scatter': wave.SOFT_Scattering,
#     # 'SOFT_averages': wave.SOFT_AverageProperties,
#     'velocity_verlet': traj.VelocityVerlet,
#     'langevin': traj.Langevin,
#     'RPMD_VelocityVerlet': rpmd.RPMD_VelocityVerlet,
#     'RPMD_averages': rpmd.RPMD_EquilibriumProperties,
#     'RPMD_scatter': rpmd.RPMD_Scattering,
#     'hopping': HoppingIntegrator
#     }
