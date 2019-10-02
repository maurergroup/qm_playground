qm_playground
============

qm_playground is a Python package, 
that enables simulation of simple 1D and 2D quantum mechanical 
bound state and scattering problems using a number of different 
adiabatic and nonadiabatic techniques.

Features
--------

* Analytical potentials in 1D and 2D
* Wavepacket propagation 
* Classical molecular dynamics
* Ring Polymer molecular dynamics
* Nonadiabatic wavepacket propagation
* Interactive visualization

User Guide
----------
Execution of a qm_playground simulation involves initialising a model,
calling the model's `run()` or `solve()` methods, then visualising the
result.

##### Initialising the model
Depending on the type of simulation being performed this step can vary
slightly. Creating an instance of the `Model` class requires it be provided
with all the necessary components for the simulation. The essential components
are a system, a potential, a mode. For a time-dependent simulation one must
also provide an integrator.

A brief look at the examples should make clear how this works.

##### Running the simulation
Once the model is set up, this is the easy part. The `run()` method propagates
the system through time and writes the results to the `model.data` attribute.
The `solve()` method is used only for the solution of the time-independent
Schrodinger equation for wave models and similarly writes the output to
`model.data`.

After the simulation has finished, the contents of `model.data` are written to
a pickled output file with the `.end` extension. The name of this file defaults
to `simulation.end` but can be changed by providing the model with a `name`
keyword argument.

##### Visualisation
Provided in the
visualisation directory is a file called `plot.py`. This can be run from the
command line as `bokeh serve --show plot.py`. A browser window will open that
allows the user to provide the path to their `.end` file. By clicking the
button a visualisation of the trajectory will appear. This works currently for
wave, traj and rpmd modes only currently.

Another option is to use the matplotlib based visualisations available in
`qmp.tools.visualizations`.

Licensing
---------

This code is licensed under the GNU General Public License, version 3.
