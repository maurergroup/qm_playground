"""
contains predefined model potentials
"""

import numpy as np
from qmp.integrator.dyn_tools import create_gaussian

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
            'gauss')
        cell:    simulation box
        kwargs:  parameters of specific potential (see definitions)
	"""

	print "Using model potential '"+str(name)+"'"
	print 'Parameters:'
	for key, value in kwargs.iteritems():
            print '    '+key+':  '+str(value)
	if kwargs == {}:
	    print 'No specific parameters given. Using defaults.'
	print ''

	## free particle 
	def f_free(x):
	    return np.zeros_like(x)

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

	## gaussian
	gauss_p = kwargs.get('gauss_pos', np.sum(cell)/2.)
	gauss_s = kwargs.get('gauss_sigma', 5.)
	if (gauss_p < cell[0][0]) or \
	   (gauss_p > cell[0][1]):
	    raise ValueError('Adjust potential maximum!')
	def f_gauss(x):
	    return create_gaussian(x, sigma=gauss_s, x0=gauss_p)


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
	elif name == 'morse':
	    return f_morse
	elif name == 'mexican_hat':
	    return f_mexican
	elif name == 'gaussian':
	    return f_gauss
	else:
	    raise NotImplementedError("Name '"+name+"' could not be resolved\n\
Available potentials: 'free', 'box', 'double_box', \n\
'harmonic', 'morse', 'mexican_hat', and 'gaussian'")



