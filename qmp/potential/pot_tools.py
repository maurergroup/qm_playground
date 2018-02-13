#qmp.potential.pot_tools
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
contains predefined model potentials
"""

import numpy as np
from qmp.integrator.dyn_tools import create_real_gaussian, create_real_gaussian2D

def create_potential(cell, name='free', **kwargs):
        """
        returns potential function

        parameters:
        ===========
        name:    name of potential(
            'free',
            'box',
            'double_box',
            'wall',
            'harmonic',
            'morse',
            'mexican_hat',
            'gaussian',
            'double_well')
        cell:    simulation box
        kwargs:  parameters of specific potential (see definitions)
        """

        #print( "Using model potential '"+str(name)+"'")
        #print( 'Parameters:')
        #for key, value in kwargs.items():
        #    print( '    '+key+':  '+str(value))
        #if kwargs == {}:
        #    print( 'No specific parameters given. Using defaults.')
        #print( '')
        
        if name == 'free':
            ## free particle 
            def f_free(x):
                return np.zeros_like(x)
            
            return f_free
        
        elif name == 'wall':
            ## wall
            wall_p = kwargs.get('wall_pos', np.sum(cell)/2.)
            wall_w = kwargs.get('wall_width', 2.)/2.
            wall_h = kwargs.get('wall_height', 2.)
            if (wall_p-wall_w < cell[0][0]) or \
               (wall_p+wall_w > cell[0][1]):
                raise ValueError('Adjust position!')
            def f_wall(x):
                x = np.array([x]).flatten()
                for i, xx in enumerate(x):
                    if xx >= wall_p-wall_w and xx < wall_p+wall_w:
                        x[i] = wall_h
                    else:
                        x[i] = 0.
                return x
            
            return f_wall
        
        elif name == 'box':
            ## particle in a box 
            box_p = kwargs.get('box_pos', np.sum(cell)/2.)
            box_w = kwargs.get('box_width', 4.)/2.
            box_h = kwargs.get('box_height', 1000000.)
            if (box_p-box_w < cell[0][0]) or \
               (box_p+box_w > cell[0][1]):
                raise ValueError('Adjust position!')
            def f_box(x):
                x = np.array([x]).flatten()
                for i,xx in enumerate(x):
                    if xx<= box_p+box_w and xx>box_p-box_w:
                        x[i]= 0.0
                    else:
                        x[i]= box_h
                return x

            return f_box
        
        elif name == 'double_box':
            ## particle in two square wells 
            box1_p = kwargs.get('box1_pos', np.sum(cell)/4.)
            box1_w = kwargs.get('box1_width', 4.)/2.
            box2_p = kwargs.get('box2_pos', 3.*np.sum(cell)/4.)
            box2_w = kwargs.get('box2_width', 4.)/2.
            outer_h = kwargs.get('outer_height', 1000000.)
            inner_h = kwargs.get('inner_height', 1000000.)
            if (box2_p < box1_p) or \
               (box1_p+box1_w >= box2_p-box2_w) or \
               (box1_p-box1_w <= cell[0][0]) or \
               (box2_p+box2_w >= cell[0][1]):
                raise ValueError('Adjust parameters!')
            def f_double_box(x):
                x = np.array([x]).flatten()
                for i,xx in enumerate(x):
                    if (xx<=box1_p-box1_w and xx>box2_p+box2_w):
                        x[i]= outer_h
                    elif (xx<=box2_p-box2_w and xx>box1_p+box1_w):
                        x[i]= inner_h
                    else:
                        x[i]= 0.
                return x
            return f_double_box
        
        elif name == 'harmonic':
            ## harmonic potential
            harm_om = kwargs.get('harmonic_omega', .5)
            harm_p = kwargs.get('harmonic_pos', np.sum(cell)/2.)
            if (harm_p < cell[0][0]) or \
               (harm_p > cell[0][1]):
                raise ValueError('Adjust potential minimum!')
            def f_harm(x):
                x = np.array([x]).flatten()
                for i, xx in enumerate(x):
                    x[i] = harm_om*(xx-harm_p)**2
                return x

            return f_harm
        
        elif name == 'morse':
            ## morse potential
            morse_a = kwargs.get('morse_a', 0.5)
            morse_D = kwargs.get('morse_D', 5.0)
            morse_p = kwargs.get('morse_pos', np.sum(cell)/3.)
            if (morse_p < cell[0][0]) or \
               (morse_p > cell[0][1]):
                raise ValueError('Adjust potential minimum!')
            def f_morse(x):
                x = np.array([x]).flatten()
                for i, xx in enumerate(x):
                    x[i] = morse_D*(1-np.exp(-morse_a*(xx-morse_p)))**2
                return x

            return f_morse
        
        elif name == 'mexican_hat':
            ## 1D "mexican hat"
            mex_p = kwargs.get('mexican_pos', np.sum(cell)/2.)
            mex_si = kwargs.get('mexican_sigma', 1.)
            mex_sc = kwargs.get('mexican_scale', 5.)
            if (mex_p < cell[0][0]) or \
               (mex_p > cell[0][1]):
                raise ValueError('Adjust position!')
            def f_mexican(x):
                pref = mex_sc/(np.sqrt(3*mex_si)*np.pi**(1./4.))
                brak = 1.-((x-mex_p)/mex_si)**2
                f = pref*(brak*np.exp(-(1./2.)*((x-mex_p)/mex_si)**2))
                f += 2.*pref*np.exp(-3./2.)
                return f

            return f_mexican
        
        elif name == 'gaussian':
            ## gaussian
            gauss_p = kwargs.get('gauss_pos', np.sum(cell)/2.)
            gauss_s = kwargs.get('gauss_sigma', 5.)
            if (gauss_p < cell[0][0]) or \
               (gauss_p > cell[0][1]):
                raise ValueError('Adjust potential maximum!')
            def f_gauss(x):
                return create_real_gaussian(x, sigma=gauss_s, x0=gauss_p)

            return f_gauss
        
        elif name == 'double_well':
            ## double well
            dwell_h = kwargs.get('double_well_barrier', 2.)
            dwell_w = kwargs.get('double_well_width', 2.)
            dwell_c = kwargs.get('double_well_center', np.mean(cell))
            dwell_a = kwargs.get('double_well_asymmetry', 0.)
                
            def f_dwell(x):
                return dwell_h/(dwell_w**4)*((x-dwell_c)**2 - dwell_w**2)**2 + dwell_a*(x-dwell_c)**3
            
            return f_dwell

        else:
            raise NotImplementedError("Name '"+name+"' could not be resolved\n\
Available potentials: 'free', 'box', 'double_box', \n\
'harmonic', 'morse', 'mexican_hat', 'gaussian', and 'double_well'")


def create_potential2D(cell, name='free', **kwargs):
        """
        returns potential function

        parameters:
        ===========
        name:    name of potential(
            'free',
            'box',
            'double_box',
            'elbow',
            'wall',
            'harmonic',
            'mexican_hat',
            'gaussian')
        cell:    simulation box
        kwargs:  parameters of specific potential (see definitions)
        """

        print( "Using model potential '"+str(name)+"'")
        print( 'Parameters:')
        for key, value in kwargs.iteritems():
            print( '    '+key+':  '+str(value))
        if kwargs == {}:
            print( 'No specific parameters given. Using defaults.')
        print( '')

        ## free particle
        def f_free(x,y):
            return np.zeros_like(x)

        ## 2D box
        def f_box(x,y):
            box_p = kwargs.get('box_pos', np.mean(cell,0))
            box_wx = kwargs.get('box_widthx', np.mean(cell,0)[0]/2.)/2.
            box_wy = kwargs.get('box_widthy', np.mean(cell,0)[1]/2.)/2.
            box_h = kwargs.get('box_height', 10000000000.)
            
            m = (x < box_p[0]+box_wx)*(x > box_p[0]-box_wx)
            m *= (y<box_p[1]+box_wy)*(y>box_p[1]-box_wy)
            return m*box_h
            
        ## double box
        def f_double_box(x,y):
            box1_p = kwargs.get('box1_pos', np.mean(cell,0)/3.)
            box1_wx = kwargs.get('box1_widthx', np.mean(cell,0)[0]/2.)/2.
            box1_wy = kwargs.get('box1_widthy', np.mean(cell,0)[1]/2.)/2.
            box2_p = kwargs.get('box2_pos', 2.*np.mean(cell,0)/3.)
            box2_wx = kwargs.get('box2_widthx', np.mean(cell,0)[0]/2.)/2.
            box2_wy = kwargs.get('box2_widthy', np.mean(cell,0)[1]/2.)/2.
            db_h = kwargs.get('double_box_height', 1000000.)
            
            m1 = (x < box1_p[0]+box1_wx)*(x > box1_p[0]-box1_wx)
            m1 *= (y<box1_p[1]+box1_wy)*(y>box1_p[1]-box1_wy)
            m2 = (x < box2_p[0]+box2_wx)*(x > box2_p[0]-box2_wx)
            m2 *= (y<box2_p[1]+box2_wy)*(y>box2_p[1]-box2_wy)
            return -(m1+m2)*db_h
        
        
        ## elbow potential
        def f_elbow(x,y):
            elbow_sc = kwargs.get('elbow_scale', 2.)
            elbow_p1 = kwargs.get('elbow_pos1', [11,4.])
            elbow_p2 = kwargs.get('elbow_pos2', [4.,31./3.])
            elbow_si1 = kwargs.get('elbow_sigma1', [9./2.,1.])
            elbow_si2 = kwargs.get('elbow_sigma2', [3./2.,11./2.])
            
            z2 = np.exp( -(1./2.)*(((x-y-0.1)/2.)**2 + ((x-y-0.1)/2.)**2))
            z = 100.*(-create_real_gaussian2D(x,y,x0=elbow_p1,sigma=elbow_si1)-create_real_gaussian2D(x,y,x0=elbow_p2,sigma=elbow_si2))+(50./3.)*z2
            return np.real(elbow_sc*z)
        

        ## harmonic potential
        def f_harm(x,y):
            omx = kwargs.get('harmonic_omega_x', 1./2.)
            omy = kwargs.get('harmonic_omega_y', 1./2.)
            harm_p = kwargs.get('harmonic_pos', np.mean(cell,0))
            if (harm_p[0] < cell[0][0]) or \
               (harm_p[0] > cell[1][0]) or \
               (harm_p[1] < cell[0][1]) or \
               (harm_p[1] > cell[1][1]):
                raise ValueError('Please define positions within cell')
            
            return omx*((x-harm_p[0])**2) + omy*((y-harm_p[1])**2)
        

        ## wall
        def f_wall(x,y):
            wall_dir = kwargs.get('wall_dir', 0)
            wall_p = kwargs.get('wall_pos', np.mean(cell,wall_dir))
            wall_h = kwargs.get('wall_height', 100000000.)
            
            if wall_dir == 0:
                m = (x > wall_p)
            elif wall_dir == 1:
                m = (y > wall_p)
            else:
                raise ValueError("Please define 'wall_dir' as 0 or 1 (corresponds to x or y direction, respectively)")
            
            if (wall_dir == 0) and \
               ((wall_p < cell[0][0]) or (wall_p > cell[1][0])):
                raise ValueError('Please define position within cell')
            elif (wall_dir == 1) and \
               ((wall_p < cell[0][1]) or (wall_p > cell[1][1])):
                raise ValueError('Please define position within cell')
        
            return m*wall_h
        

        ## mexican hat
        def f_mexican(x,y):
            mex_si = kwargs.get('mexican_sigma', 1.)
            mex_sc = kwargs.get('mexican_scale', 20.)
            mex_p = kwargs.get('mexican_pos', np.mean(cell,0))
            if (mex_p[0] < cell[0][0]) or \
               (mex_p[0] > cell[1][0]) or \
               (mex_p[1] < cell[0][1]) or \
               (mex_p[1] > cell[1][1]):
                raise ValueError('Please define positions within cell')
            
            pref = mex_sc/(np.pi*mex_si**4)
            brak = 1.-(((x-mex_p[0])**2+(y-mex_p[1])**2)/(2*mex_si**2))
            f = pref*brak*np.exp(-(((x-mex_p[0])**2+(y-mex_p[1])**2)/(2.*mex_si**2)))
            return f - min(f.flatten())
        

        ## gaussian
        def f_gauss(x,y):
            gauss_s = kwargs.get('gaussian_sigma', [1.,1.])
            gauss_p = kwargs.get('gaussian_pos', np.mean(cell,0))
            if (gauss_p[0] < cell[0][0]) or \
               (gauss_p[0] > cell[1][0]) or \
               (gauss_p[1] < cell[0][1]) or \
               (gauss_p[1] > cell[1][1]):
                raise ValueError('Please define positions within cell')
            
            return np.real(create_real_gaussian2D(x, y, sigma=gauss_s, x0=gauss_p))
        
            

        if name == 'free':
            return f_free
        elif name == 'wall':
            return f_wall
        elif name == 'box':
            return f_box
        elif name == 'double_box':
            return f_double_box
        elif name == 'harmonic':
            return f_harm
        #elif name == 'morse':
        #    return f_morse
        elif name == 'elbow':
            return f_elbow
        elif name == 'mexican_hat':
            return f_mexican
        elif name == 'gaussian':
            return f_gauss
        else:
            raise NotImplementedError("Name '"+name+"' could not be resolved\n\
Available potentials: 'free', 'box', 'double_box', 'elbow',\n\
'harmonic', 'mexican_hat', and 'gaussian'")
