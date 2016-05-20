#setup.py
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


try:
    from setuptools import setup
except:
    from distutils.core import setup

#ext_modules = [
    #Extension('internal', [ 'interna/internal.f90'],swig_opts=['-ipo -O3 -prec-div -axW -static']#,
##        libraries = ['imf', 'svml', 'ifcore'],
        #)
    #]

packages = ['qmp',
            'qmp.basis',
            'qmp.integrator',
            'qmp.solver',
            'qmp.tools',
            'qmp.potential',
            ]

config = {
    'description': 'playground code for solving simple 1D/2D QM problems',
    'author': 'Reinhard J. Maurer',
    'url': 'URL',
    'download_url': 'download it.',
    'author_email': 'reinhard.maurer@yale.edu',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': packages,
    'name': 'qm_playground'
}

setup(**config)

