#    qmp.tools.dyn_tools
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
This module contains a variety of utility functions that can be used to assist
the generation of initial conditions for a simulation.
"""
import numpy as np
from scipy.stats import maxwell

kB = 1.380649e-23
atomic_to_kelvin = 3.15777504e5


def project_wvfn(wvfn, evecs):
    """Project wave packet onto eigenvectors and return vector of coefficients.

    Parameters
    ----------
    evecs : array_like
        Matrix containing the eigenvectors of the Hamiltonian.
    wvfn : array_like
        Wavepacket to be projected on eigenstates defined on the same grid as
        the eigenvectors.

    Returns
    -------
    array_like
        A vector containing the normalised coefficients corresponding to the
        eigenvectors.
    """
    c = np.dot(wvfn.flatten(), evecs)
    norm = np.sqrt(np.conjugate(c).dot(c))
    return c/norm


def create_gaussian(x, x0=0., p0=0., sigma=1.):
    """Create a 1D gaussian wavepacket.

    Parameters
    ----------
    x : array_like
        Grid for wavepacket.
    x0 : float or int, optional
        Expectation value of position.
    p0 : float or int, optional
        Expectation value of momentum.
    sigma: float or int, optional
        Variance of gaussian.

    Returns
    -------
    array_like
        A 1D gaussian.
    """
    wave = np.exp(-((x-x0)**2/sigma**2/4.)
                  + 1j*p0*(x-x0))/(np.sqrt(np.sqrt(2.*np.pi)*sigma))
    return wave


def create_real_gaussian(x, x0=0., sigma=1.):
    """Creates a real gaussian wave.

    Parameters
    ----------
    x : array_like
        1D grid for gaussian.
    x0 : float or int, optional
        Center/expectation value of gaussian (default 0.)
    sigma : float or int, optional
        Variance of gaussian (default 1.)
    """
    wave = np.exp(-((x-x0)**2/sigma**2/4.))/(np.sqrt(np.sqrt(2.*np.pi)*sigma))
    return wave


def create_gaussian2D(xgrid, ygrid, x0=[0., 0.], p0=[0., 0.], sigma=[1., 1.]):
    """Creates 2D gaussian wave.

    Parameters
    ----------
    xgrid : array_like
        x grid wave will be constructed on
    ygrid : array_like
        y grid wave will be constructed on
    x0 : array_like, optional
        (initial) center/expectation value of wave (default [0.,0.])
    p0 : array_like, optional
        initial momentum of wave (default [0.,0.])
    sigma : array_like, optional
        variance of gaussian in x and y direction (default [1.,1.])
    """
    if (type(sigma) == float) or (type(sigma) == int):
        sigma = [sigma, sigma]

    wave = np.exp(-(1/2.)*(((xgrid-x0[0])/sigma[0])**2
                  + ((ygrid-x0[1])/sigma[1])**2)
                  + 1j*(p0[0]*(xgrid-x0[0])
                  + p0[1]*(ygrid-x0[1])))
    return wave


def create_real_gaussian2D(xgrid, ygrid, x0=[0., 0.], sigma=[1., 1.]):
    """Creates 2D gaussian wave

    Parameters
    ----------
    xgrid : array_like
        x grid
    ygrid : array_like
        y grid
    x0 : array_like, optional
        (initial) center/expectation value of wave (default [0.,0.])
    sigma : array_like, optional
        variance of gaussian in x and y direction (default [1.,1.])
    """
    if (type(sigma) == float) or (type(sigma) == int):
        sigma = [sigma, sigma]

    wave = np.exp(-(1/2.)*(((xgrid-x0[0])/sigma[0])**2
                           + ((ygrid-x0[1])/sigma[1])**2))
    return wave


def create_2D_NVEdistribution(e, m, n):
    """Create 2D NVE phasespace distribution.

    Create an initial phase space distribution with n points
    at a given energy e for particles with mass m according to a
    potential energy function f.
    """
    from numpy.random import random
    r0 = np.zeros([n, 2])
    v0 = np.zeros([n, 2])
    v2 = 2.*e/m
    for i in range(n):
        vx, vy = random(2)*2.0-1.0  # random numbers between -1 and 1
        v2tmp = vx**2+vy**2
        vx = vx*np.sqrt(v2/v2tmp)
        vy = vy*np.sqrt(v2/v2tmp)
        v0[i, :] = [vx, vy]
    return r0, v0


def create_thermostat(name='no_thermostat', **kwargs):
    """Returns thermostat as defined by kwargs."""
    from scipy.stats import norm

    def andersen_ts(v, m, dt, ndim):
        cfreq = kwargs.get('cfreq', 0.001)
        T_set = kwargs.get('T_set', 293.)

        p_rand = np.random.random(v.shape[0])
        mask = np.array([(p_rand < cfreq*dt)]*ndim).T
        #if p_rand < cfreq*dt:
        s = np.sqrt(kB*T_set/m)
        p_rand = np.random.random(v.shape)
        #v = norm.ppf(p_rand,scale=s)
        #dt = 0.

        v_rand = norm.ppf(p_rand,scale=s)
        v = v_rand*mask + v*(1.-mask)
        dt = dt*mask[:,0]

        return v, dt

    def no_ts(v, m, dt, ndim):
        return v, 0

    if name == 'no_thermostat':
        return no_ts
    elif (name == 'Andersen') or (name == 'andersen'):
        return andersen_ts
    else:
        raise KeyError("Thermostat '"+name+"' is not implemented yet or misspelled. Available: 'Andersen', 'no_thermostat'")


def EOM_morse_analyt(a, D, m, t, pos, Temp=293.15):
    """Not really sure what this is for."""

    if kB*Temp >= D:
        raise ValueError('System not bound at given Temperature')

    xi = np.arccos(np.sqrt(kB*Temp/D))
    om_0 = a*np.sqrt(2.*D/m)
    return pos + (np.log( (1.-np.cos(xi)*np.cos(om_0*np.sin(xi)*t))/(np.sin(xi)**2) ))/a


def get_v_init(pot, r_p=[1.], m_p=1., E=1., v_dir=[1.]):
    """Create velocity for particle in order to match total energy.

    Parameters
    ----------
    r_p
        the particles position
    m_p
        mass of particle
    E
        the particles initial energy
    v_dir
        direction of particles motion
    """
    from scipy.linalg import norm

    pot_p = np.real(pot(*r_p))
    v_dir = np.array(v_dir)
    v_dir /= norm(v_dir)
    vel_p = v_dir*np.sqrt(2.*(E-pot_p)/m_p)
    return vel_p


def get_v_maxwell(m, T):
    """Draw velocity from Maxwell-Boltzmann distribution with mean 0."""
    s = np.sqrt(T/m)
    x_rand = np.random.random(1)
    return maxwell.ppf(x_rand, loc=0., scale=s)


def kronecker(i, j):
    if i == j:
        return 1
    else:
        return 0
