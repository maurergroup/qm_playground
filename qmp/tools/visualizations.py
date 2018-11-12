#qmp.tools.visualizations
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
1D/2D visualization tools
"""

import numpy as np
from copy import copy
def wave_movie1D(basis, psi_arr, pot, dt=1., E_arr=None, rho_tot_arr=None, E_kin_arr=None, E_pot_arr=None,save=False,filename='wave_dyn1D',pot_lims=None):
    if save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=8, metadata=dict(title='wave scattering',artist='Matplotlib'), bitrate=128)
    else:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    
    fig = plt.figure(figsize=(18.5,10.5))
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
    if pot_lims is not None:
        ax0.set_ylim(pot_lims[0],pot_lims[1])
    for tl in ax0.get_yticklabels():
        tl.set_color('r')
    
    wave_plot, = ax.plot(basis, psi_arr[0,:], label=r'$\rho_t(x)$')
    
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
        psi_min = min(psi_arr.flatten())
        psi_max = max(psi_arr.flatten())
        ax.set_ylim(psi_min-0.01*abs(psi_max), psi_max+0.1*abs(psi_max))
        ax.legend(loc=2)
        return wave_plot,
    
    def animate(i):
        wave_plot.set_ydata(psi_arr[i,:])  # update data
        return wave_plot,
    
    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(psi_arr)), init_func=_init_, \
                                  interval=50, blit=False)
    
    if save:
        ani.save(filename+'.mp4',writer=writer)
    else:
        plt.show()


def wave_slideshow1D(basis, psi_arr, pot):
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    ## generate figure
    fig, ax = plt.subplots()
    ax0 = ax.twinx()
    plt.subplots_adjust(bottom=0.2)
    l, = ax.plot(basis,psi_arr[:,0]*np.conjugate(psi_arr[:,0]))
    ax.set_ylim(-0.025*max((psi_arr*np.conjugate(psi_arr)).flatten()), 0.9*max((psi_arr*np.conjugate(psi_arr)).flatten()))
    k, = ax0.plot(basis,pot, c='r', ls=':')
    
    ## buttons
    class Index:
        def __init__(self):
            self.ind = 0
        def next(self, event):
            self.ind += 1
            if self.ind == psi_arr.shape[1]:
                self.ind = 0
            l.set_ydata(psi_arr[:,self.ind]*np.conjugate(psi_arr[:,self.ind]))
            plt.draw()
        
        def prev(self, event):
            self.ind -= 1
            if self.ind == -1:
                self.ind = psi_arr.shape[1]-1
            l.set_ydata(psi_arr[:,self.ind]*np.conjugate(psi_arr[:,self.ind]))
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
    theta = 25.
    
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
        def __init__(self):

            self.ind = 0
        
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


def contour_movie2D(xgrid, ygrid, pot, pos_arr, steps, npar=1, interval = 50, trace=False):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm
    markers = ['o','v','^','<','>','8','s','p','*','h','H','+','D','d']
    ml = len(markers)
    
    fig = plt.figure()
    ax0 = plt.gca()
    dummy = ax0.contourf(xgrid, ygrid, pot, 48)
    lvl = dummy.levels
    lvl = np.sort(np.append(np.array(lvl[2:]),np.linspace(lvl[0],lvl[1],4)))

    if trace == False:

        if npar == 1:
            pos_plot, = ax0.plot(pos_arr[0,0], pos_arr[0,1], label=r'$\vec{x}(t)$', \
                              ls='', marker='o', mec='k',mfc='k', ms=9)

            def _init_():
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                cbar = plt.colorbar(pot_plot)
                cbar.ax.set_ylabel('$potential$ $energy$ $[a.u.]$')
                return pos_plot,
        
            def animate(i):
                pos_plot.set_data(pos_arr[i,0], pos_arr[i,1])
                return pos_plot,
        
            ani = animation.FuncAnimation(fig, animate, np.arange(0, steps), init_func=_init_, \
                                   interval=interval, blit=False)
        
            plt.show()
        else:
            plots = []
            for p in range(npar):
                plots.append(ax0.plot(pos_arr[p,0,0], pos_arr[p,0,1], \
                        label='r{0:03d}(t)'.format(p), \
                        ls='', marker=markers[p%ml], mec='k',mfc='k', ms=9)[0])
                ttl = ax0.text(0.5,1.005,'', transform=ax0.transAxes)
            def _init_():
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                cbar = plt.colorbar(pot_plot)
                cbar.ax.set_ylabel('$potential$ $energy$ $[a.u.]$')
                ttl.set_text('step=0')
                return tuple(plots), ttl
    
            def animate(i):
                result = []
                for p in range(npar):
                    plots[p].set_data(pos_arr[p,i,0],pos_arr[p,i,1])
                ttl.set_text('step='+str(i))
                return tuple(plots), ttl
    
            ani = animation.FuncAnimation(fig, animate, np.arange(0, steps), init_func=_init_, \
                               interval=interval, blit=False)
    
            plt.show()
    else:
        if npar ==1:
            pos_plot, = ax0.plot(pos_arr[0,0,0], pos_arr[0,0,1], label=r'$\vec{x}(t)$', \
                     ls='', marker='o', mec='k',mfc='k', ms=9,zorder=10)
            trace_plot, = ax0.plot(pos_arr[0,0,0], pos_arr[0,0,1], ls=':',lw=2,c='0.9',zorder=1)
    
            def _init_():
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                cbar = plt.colorbar(pot_plot)
                cbar.ax.set_ylabel('$potential$ $energy$ $[a.u.]$')
                return pos_plot, trace_plot,
    
            def animate(i):
                trace_plot.set_data(pos_arr[0,:i+1,0], pos_arr[0,:i+1,1])
                pos_plot.set_data(pos_arr[0,i,0], pos_arr[0,i,1])
                return pos_plot, trace_plot,
    
            ani = animation.FuncAnimation(fig, animate, np.arange(0, steps), init_func=_init_, \
                               interval=interval, blit=False)
    
            plt.show()
        else:
            plots = []
            traces = []
            for p in range(npar):
                plots.append(ax0.plot(pos_arr[p,0,0], pos_arr[p,0,1], \
                            label='r{0:03d}(t)'.format(p), \
                            ls='', marker=markers[p%ml], mec='k',mfc='k', ms=9)[0])
                traces.append(ax0.plot(pos_arr[p,0,0], pos_arr[p,0,1], ls=':',lw=2,c='0.9',zorder=1)[0])
                ttl = ax0.text(0.5,1.005,'', transform=ax0.transAxes)
            def _init_():
                pot_plot = ax0.contourf(xgrid, ygrid, pot, lvl, ls=None,alpha=.75,cmap=cm.jet)
                cbar = plt.colorbar(pot_plot)
                cbar.ax.set_ylabel('$potential$ $energy$ $[a.u.]$')
                ttl.set_text('step=0')
                return tuple(plots), tuple(traces),ttl,
    
            def animate(i):
                result = []
                for p in range(npar):
                    plots[p].set_data(pos_arr[p,i,0],pos_arr[p,i,1])
                    traces[p].set_data(pos_arr[p,:i+1,0],pos_arr[p,:i+1,1])
                ttl.set_text('step='+str(i))
                return tuple(plots), tuple(traces),ttl,
            ani = animation.FuncAnimation(fig, animate, np.arange(0, steps), init_func=_init_, \
                               interval=interval, blit=False)
    
            plt.show()
    
def propability_distribution1D(model, plot_pot=True, show_all_particles=False, show_plot=False, \
                               scale_potential=1., figname="propability_distribution1d.pdf"):
    import matplotlib.pyplot as plt
    from qmp.model import Model as mod_ref
    from matplotlib.lines import Line2D
    
    
    assert type(model) == mod_ref, "Please, provide qmp model."
    sim_mode = model.parameters['mode']
    dummy_str = "hasattr(model.data."+sim_mode+", 'prop_tot')"
    has_prop_tot =  eval(dummy_str)
    err_msg = "Please, specify a model that provides propability distribution"
    assert has_prop_tot, err_msg
    if show_all_particles:
        dummy_str = "hasattr(model.data."+sim_mode+", 'prop_vals')"
        has_prop_vals =  eval(dummy_str)
        err_msg = "Please, specify a model that provides propability distributions"
        assert has_prop_vals, err_msg
    
    Ptot = eval("model.data."+sim_mode+".prop_tot")
    bins = eval("model.data."+sim_mode+".prop_bins")
    assert len(bins) == 1, "Is this one-dimensional?"
    bins = bins[0]
    if plot_pot:
        xgrid = np.arange(bins[0], bins[-1]+0.1, 0.1)
        pot = model.pot(xgrid)
    
    if show_all_particles: Pvals = eval("model.data."+sim_mode+".prop_vals")
    fig = plt.figure()
    ax = fig.gca()
    labs, lines = [], []
    if plot_pot:
        ax2 = ax.twinx()
        pot_label = 'potential'
        ax2.plot(xgrid, pot, c='r', lw=1.75, ls='--', label=pot_label)
        ax2.set_ylim(min(pot), min(pot)+1.2*max(Ptot)*scale_potential)
        ax2.set_ylabel('potential energy [Ha]')
        labs.append(pot_label)
        lines.append(Line2D([0.], [1.], c='r', lw=1.75, ls='--'))
    
    if show_all_particles:
        for P_i in Pvals: ax.plot(bins[:-1], P_i/(len(Pvals)), alpha=0.5, lw=1.5)
        lines.append(Line2D([0.], [1.], c='k', alpha=0.5))
        labs.append("individual particle density")
    
    prop_label = 'total propability density'
    ax.plot(bins[:-1], Ptot, c='b', lw=1.75, label=prop_label)
    labs.append(prop_label)
    lines.append(Line2D([0.], [1.],  c='b', lw=1.75))
    
    ax.set_xlabel('x [a.u.]')
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0., 1.2*max(Ptot))
    ax2.legend(lines, labs, loc='best')
    ax.set_title("Propability distribution from QMP simulation")
    fig.savefig(figname)
    plt.show() if show_plot else plt.close()
    

def propability_distribution2D(model, plot_pot=True, dx_pot=0.1, dy_pot=0.1, show_plot=False, \
                               scale_potential=1., nlines_pot=8, add_contour_labels=False, \
                               only_particle=None, figname="propability_distribution2d.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec as gsp
    from matplotlib import colors as cm
    from qmp.model import Model as mod_ref
    
    
    assert type(model) == mod_ref, "Please, provide qmp model."
    sim_mode = model.parameters['mode']
    has_property = "hasattr(model.data."+sim_mode+", 'prop_tot')"
    err_msg = "Please, specify a model that provides propability distribution"
    assert eval(has_property), err_msg
    
    if only_particle is None:
        Ptot = eval("model.data."+sim_mode+".prop_tot")
    else:
        err_msg = "Please, specify integer <i> for particle to be plottet (only_particle=<i>)"
        assert type(only_particle) == int, err_msg
        err_msg = "Please, specify a model that provides propability distributions"
        has_property = "hasattr(model.data."+sim_mode+", 'prop_vals')"
        assert eval(has_property), err_msg
        try:
            npar = model.basis.npar
            check = True
        except:
            check = False
        
        assert check, "Your model does not provide information on the number of particles. Stopping."
        err_msg = "Specified index for particle to be plotted is out of range.\nOnly "
        err_msg += str(npar)+" particles simulated. (particle indices start at 0)"
        assert only_particle<=npar-1, err_msg
        Ptot = eval("model.data."+sim_mode+".prop_vals["+str(only_particle)+"]")
    
    bins = eval("model.data."+sim_mode+".prop_bins")
    assert len(bins) == 2, "Is this two-dimensional?"
    x, y = np.meshgrid(bins[0][:-1],bins[1][:-1])
    gs = gsp.GridSpec(2,2, height_ratios=[0.05,1.], width_ratios=[1.,0.05])
    gs.update(left=0.075, right=0.9, bottom=0.08, top=0.9, wspace=0.02, hspace=0.03)
    ax = plt.subplot(gs[1,0])
    con_p = ax.contourf(x, y, Ptot-1e-12, cmap='hot_r', linestyles=None, extend='neither', \
                        vmin=0., vmax=np.max(abs(Ptot)))
    for c in con_p.collections: c.set_linestyle('solid')
    cbax = plt.subplot(gs[1,1])
    norm = cm.Normalize(vmin=0., vmax=con_p.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=con_p.cmap)
    sm.set_array([])
    cb = plt.colorbar(cax=cbax, mappable=sm, ticks=con_p.levels, orientation='vertical', ticklocation='right')
    cb.set_label('Propability density', labelpad=10)
    if plot_pot:
        xax = np.arange(bins[0][0], bins[0][-1]+dx_pot, dx_pot)
        yax = np.arange(bins[1][0], bins[1][-1]+dy_pot, dy_pot)
        xgrid, ygrid = np.meshgrid(xax, yax)
        V_xy = model.pot(xgrid, ygrid)
        con_v = ax.contour(xgrid, ygrid, V_xy, nlines_pot, cmap='winter', linewidths=2., extend='neither')
        if add_contour_labels: ax.clabel(con_v, inline=1, fontsize=10)
        cbax = plt.subplot(gs[0,0])
        norm = cm.Normalize(vmin=con_v.cvalues.min(), vmax=con_v.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=con_v.cmap)
        sm.set_array([])
        cb = plt.colorbar(cax=cbax, mappable=sm, ticks=con_v.levels, orientation='horizontal', ticklocation='top')
        cb.set_label('Potential isolines [Ha]', labelpad=10)
    
    ax.set_xlabel('x [a.u.]')
    ax.set_ylabel('y [a.u.]')
    
    plt.savefig(figname)
    plt.show() if show_plot else plt.close()
    

