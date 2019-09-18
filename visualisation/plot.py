"""
Visualisation server for qm_playground simulations.

Run with 'bokeh server --show plot.py'
Once running, enter the file path into the text box and click the button.
Hopefully a relevant plot should appear.

The construct_plot function handles the creation of the plot which amounts to
instantiating one of Plot classes.

This framework should be easily extendable to allow for more visualisations in
the future, just add another option to the if, else statement and copy one of
the current working Plot classes.
"""
from bokeh.models import TextInput, Button
from bokeh.plotting import curdoc
from bokeh.layouts import column, grid
import pickle
from wave_plot import WavePlot
from wave_plot2D import WavePlot2D
from traj_plot2D import TrajPlot2D
from traj_plot1D import TrajPlot1D
from rpmd_plot2D import RPMDPlot2D
from rpmd_plot1D import RPMDPlot1D


class FileReader():
    """Reads file and creates plot.

    Arguably a poorly named class.
    """

    def __init__(self):
        """Create text field and empty plotting grid."""

        self.file_input = TextInput(placeholder='Enter file path')
        self.layout = grid([], ncols=2)
        self.i = 0
        self.j = 0

    def add_plot_area(self):
        """Add the empty grid to the page."""
        curdoc().add_root(self.layout)

    def construct_plot(self):
        """Construct the plot.

        Instantiates a Plot object that has a get_layout method. This should
        return the bokeh layout object to be displayed on the screen.

        Currently maintains a goofy n by 2 grid of plots. Might want to change
        this.
        """
        self.load_data()

        if self.data.mode == 'wave':
            ndim = len(self.data.cell)
            if ndim == 1:
                self.plot = WavePlot(self.data)
            elif ndim == 2:
                self.plot = WavePlot2D(self.data)

        elif self.data.mode == 'traj':
            ndim = len(self.data.potential.shape)
            if ndim == 1:
                self.plot = TrajPlot1D(self.data)
            elif ndim == 2:
                self.plot = TrajPlot2D(self.data)

        elif self.data.mode == 'rpmd':
            ndim = len(self.data.potential.shape)
            if ndim == 1:
                self.plot = RPMDPlot1D(self.data)
            elif ndim == 2:
                self.plot = RPMDPlot2D(self.data)

        self.layout.children.append((self.plot.get_layout(), self.i, self.j))
        if self.j == 1:
            self.i += 1
            self.j = 0
        else:
            self.j += 1

    def load_data(self):
        """Load the pickled data from the file specified in the text field."""
        file = open(self.file_input.value, 'rb')
        self.data = pickle.load(file)


reader = FileReader()

button = Button(label="Plot data!")
button.on_click(reader.construct_plot)

format = column(reader.file_input, button)
curdoc().add_root(format)
reader.add_plot_area()
