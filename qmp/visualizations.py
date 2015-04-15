"""
1D/2D visualization tools
"""

import numpy as np


def wave_movie1D(basis, psi_arr, pot, dt=1., E_arr=None, rho_tot_arr=None, E_kin_arr=None, E_pot_arr=None):
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation

	fig = plt.figure()
	if E_arr is None:
	    E_arr = np.zeros(psi_arr.shape[0])
	    ls_E = ''
	    lab_E = 'NO ENERGY GIVEN'
	else:
	    ls_E = '-'
	    lab_E = '$E(t)$ $[a.u.]$'

	if rho_tot_arr is None:
	    rho_tot_arr = np.zeros(psi_arr.shape[0])
	    ls_rho = ''
	    lab_rho = 'NO TOTAL DENSITY GIVEN'
	else:
	    ls_rho = '-'
	    lab_rho = r'$\Vert\rho_t\Vert^2$ $[a.u.]$'


	ax = plt.subplot2grid((4,2), (1,0), rowspan=3, colspan=2)
	ax1 = plt.subplot2grid((4,2), (0,0))
	ax2 = plt.subplot2grid((4,2), (0,1))
	for tl in ax.get_yticklabels():
	    tl.set_color('b')
	ax0 = ax.twinx()
	for tl in ax0.get_yticklabels():
	    tl.set_color('r')
	
	wave_plot, = ax.plot(basis, psi_arr[0,:]*np.conjugate(psi_arr[0,:]), label=r'$\rho_t(x)$')

	def _init_():
	    ax0.plot(basis, pot, ls=':', c='r', label='$V(x)$')
	    ax0.legend(loc=1)
	    ax1.plot(np.linspace(0., len(E_arr)*dt, len(E_arr)), E_arr, c='b', ls=ls_E, label=lab_E)
	    if E_kin_arr is not None:
		ax1.plot(np.linspace(0., len(E_arr)*dt, len(E_arr)), E_kin_arr, c='g', label='$E_{kin}(t)$ $[a.u.]$')

	    if E_pot_arr is not None:
		ax1.plot(np.linspace(0., len(E_arr)*dt, len(E_arr)), E_pot_arr, c='r', label='$E_{pot}(t)$ $[a.u.]$')
	
	    ax1.legend(loc='best')
	    ax1.set_xlim([0., len(E_arr)*dt])
	    ax1.set_xlabel('$t$ $[a.u.]$')
	    ax1.xaxis.tick_top()
	    ax1.set_xticks(ax1.get_xticks()[1:])
	    ax1.xaxis.set_label_position('top')
	    if (E_kin_arr is None) and (E_pot_arr is None):
		ax1.set_ylim(min(E_arr)-0.01, max(E_arr)+0.01)
	
	    ax2.plot(np.linspace(0., len(rho_tot_arr)*dt, len(rho_tot_arr)), rho_tot_arr, c='b', ls=ls_rho, label=lab_rho)
	    ax2.legend(loc='best')
	    ax2.set_ylim(min(rho_tot_arr)-0.01, max(rho_tot_arr)+0.01)
	    ax2.set_xlim(0., len(rho_tot_arr)*dt)
	    ax2.set_xlabel('$t$ $[a.u.]$')
	    ax2.xaxis.tick_top()
	    ax2.set_xticks(ax2.get_xticks()[1:])
	    ax2.xaxis.set_label_position('top')
	    ax2.yaxis.tick_right()
	    ax.set_xlabel('$x$ $[a.u.]$')
	    ax.set_ylim(-0.025*max((psi_arr*np.conjugate(psi_arr)).flatten()), max((psi_arr*np.conjugate(psi_arr)).flatten()))
	    ax.legend(loc=2)
	    return wave_plot,

	def animate(i):
	    wave_plot.set_ydata(psi_arr[i,:]*np.conjugate(psi_arr[i,:]))  # update data
	    return wave_plot,

	ani = animation.FuncAnimation(fig, animate, np.arange(0, len(psi_arr)), init_func=_init_, \
		                      interval=50, blit=False)

	plt.show()


def wave_slideshow1D(basis, psi_arr, pot):

	from matplotlib import pyplot as plt
	from matplotlib.widgets import Slider, Button, RadioButtons

	## generate figure
	fig, ax = plt.subplots()
	ax0 = ax.twinx()
	plt.subplots_adjust(bottom=0.2)
	l, = ax.plot(basis,psi_arr[0,:]*np.conjugate(psi_arr[0,:]))
	ax.set_ylim(-0.025*max((psi_arr*np.conjugate(psi_arr)).flatten()), 0.9*max((psi_arr*np.conjugate(psi_arr)).flatten()))
	k, = ax0.plot(basis,pot, c='r', ls=':')

	## buttons
	class Index:
	    ind = 0
	    def next(self, event):
		self.ind += 1
		if self.ind == psi_arr.shape[0]:
		    self.ind = 0
		l.set_ydata(psi_arr[self.ind,:]*np.conjugate(psi_arr[self.ind,:]))
		plt.draw()

	    def prev(self, event):
		self.ind -= 1
		if self.ind == -1:
		    self.ind = psi_arr.shape[0]-1
		l.set_ydata(psi_arr[self.ind,:]*np.conjugate(psi_arr[self.ind,:]))
		plt.draw()

	callback = Index()
	pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
	pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
	button_next = Button(pos_next_button, 'Next Wvfn')
	button_next.on_clicked(callback.next)
	button_prev = Button(pos_prev_button, 'Prev Wvfn')
	button_prev.on_clicked(callback.prev)

	plt.show()


def wave_movie2D(xgrid, ygrid, psi_arr, pot=0.):
	from matplotlib import pyplot as plt
	from matplotlib import cm
	import matplotlib.animation as animation
	from mpl_toolkits.mplot3d import Axes3D
	theta = 25.
	if not (np.any(pot) != 0.):
		pot = np.ones_like(xgrid)
		ls_pot = ''
	else:
		ls_pot = '-'
	
	#generate figure
	fig = plt.figure()
	plt.subplots_adjust(bottom=0.2)
	ax = fig.gca(projection='3d')
	
	N = xgrid.shape[0]
	rho_max = max(np.conjugate(psi_arr.flatten())*psi_arr.flatten())
	pot_max = max(pot.flatten())
	
	frame = None
	for i in xrange(psi_arr.shape[0]):
	    oldframe = frame
	
	    frame = ax.plot_surface(xgrid, ygrid, np.reshape(psi_arr[i,:]*np.conjugate(psi_arr[i,:]), (N,N)), \
				    alpha=0.75, antialiased=False, cmap = cm.coolwarm, lw=0.)
	    ax.set_zlim(-0.025*rho_max,rho_max+0.1*rho_max)
	    ax.contour(xgrid, ygrid, pot, zdir='z', offset=ax.get_zlim()[0], ls=ls_pot, cmap=cm.spectral)
	    ax.contour(xgrid, ygrid, pot/pot_max*rho_max, zdir='x', offset=min(xgrid.flatten()), ls=ls_pot, cmap=cm.spectral)
	    ax.contour(xgrid, ygrid, pot/pot_max*rho_max, zdir='y', offset=max(ygrid.flatten()), ls=ls_pot, cmap=cm.spectral)

	
	    if oldframe is not None:
		ax.collections.remove(oldframe)
		
	    plt.pause(0.0005)
	    


def wave_slideshow2D(xgrid, ygrid, psi_arr, pot=0.):

	from matplotlib import pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.widgets import Slider, Button, RadioButtons

	if not (np.any(pot) != 0.):
		pot = np.ones_like(xgrid)
		ls_pot = ''
	else:
		ls_pot = '-'
	
	N = xgrid.shape[0]
	
	psi_max = max(psi_arr.flatten())
	psi_min = min(psi_arr.flatten())
	pot_max = max(pot.flatten())
	
	## generate figure
	fig = plt.figure()
	plt.subplots_adjust(bottom=0.2)
	ax = fig.gca(projection='3d')
	ax.view_init(elev=theta , azim=-45.)
	l = ax.plot_surface(xgrid, ygrid, np.reshape(psi_arr[:,0], (N,N)), lw=0., cmap=cm.coolwarm)
	ax.set_zlim([psi_min-0.1*psi_min, psi_max+0.1*psi_max])
	ax.contour(xgrid, ygrid, pot, zdir='z', offset=ax.get_zlim()[0], ls=ls_pot, cmap=cm.spectral)
        ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='x', offset=min(xgrid.flatten()), ls=ls_pot, cmap=cm.spectral)
        ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='y', offset=max(ygrid.flatten()), ls=ls_pot, cmap=cm.spectral)

	
	
	## buttons
	class Index:
	    ind = 0
	    def next(self, event):
	        self.ind += 1
		if self.ind == psi_arr.shape[1]:
		    self.ind = 0
		ax.clear()
		l = ax.plot_surface(xgrid, ygrid, np.reshape(psi_arr[:,self.ind], (N,N)), lw=0., cmap=cm.coolwarm)
		ax.set_zlim([psi_min-0.1*psi_min, psi_max+0.1*psi_max])
		ax.contour(xgrid, ygrid, pot, zdir='z', offset=ax.get_zlim()[0], ls=ls_pot, cmap=cm.spectral)
		ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='x', offset=min(xgrid.flatten()), ls=ls_pot, cmap=cm.spectral)
		ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='y', offset=max(ygrid.flatten()), ls=ls_pot, cmap=cm.spectral)
		plt.draw()
	
	    def prev(self, event):
		self.ind -= 1
		if self.ind == -1:
		    self.ind = psi_arr.shape[1]-1
		ax.clear()
		l = ax.plot_surface(xgrid, ygrid, np.reshape(psi_arr[:,self.ind], (N,N)), lw=0., cmap=cm.coolwarm)
		ax.set_zlim([psi_min-0.1*psi_min, psi_max+0.1*psi_max])
		ax.contour(xgrid, ygrid, pot, zdir='z', offset=ax.get_zlim()[0], ls=ls_pot, cmap=cm.spectral)
		ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='x', offset=min(xgrid.flatten()), ls=ls_pot, cmap=cm.spectral)
		ax.contour(xgrid, ygrid, pot/pot_max*psi_max, zdir='y', offset=max(ygrid.flatten()), ls=ls_pot, cmap=cm.spectral)
		plt.draw()
	
	callback = Index()
	pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
	pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
	button_next = Button(pos_next_button, 'Next Wvfn')
	button_next.on_clicked(callback.next)
	button_prev = Button(pos_prev_button, 'Prev Wvfn')
	button_prev.on_clicked(callback.prev)
	
	plt.show()


