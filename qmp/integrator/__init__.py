#qmp.integrator.__init__
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

from qmp.integrator.waveintegrators import *
from qmp.integrator.trajintegrators import *
from qmp.integrator.rpmdintegrators import *

integrator_type = {
    'primitive' : prim_propagator,
    'primprop' : prim_propagator,
    'eigen' : eigen_propagator,
    'eigenprop' :eigen_propagator,
    'SOFT' : SOFT_propagation,
    'splitopprop' : SOFT_propagation,
    'split_operator' : SOFT_propagation,
    'SOFT_scatter': SOFT_scattering,
    'SOFT_averages' : SOFT_average_properties,
    'SOFT_avg_properties' : SOFT_average_properties,
    'SOFT_avg_props' : SOFT_average_properties,
    'velocity_verlet' : velocity_verlet_integrator,
    'vel_verlet' : velocity_verlet_integrator,
    'langevin' : langevin_integrator,
    'VelocityVerlet_RPMD' : RPMD_VelocityVerlet,
    'RPMD_VelocityVerlet' : RPMD_VelocityVerlet,
    'RPMD_averages' : RPMD_equilibrium_properties,
    'RPMD_equilibrium_properties' : RPMD_equilibrium_properties,
    'RPMD_eq_props' : RPMD_equilibrium_properties,
    'RPMD_scatter' : RPMD_scattering,
    }

def integrator_init(data, potential):

    param = data.parameters
    integrator = integrator_type[param['integrator']](\
            data=data, potential=potential)

    return integrator

def integrator_help():
        string = 'Integrator types include: \n'
        for k in integrator_type.keys():
            string += key+'\n'
        return string

