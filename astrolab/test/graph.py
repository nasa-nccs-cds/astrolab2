import time
import xarray as xa
from astrolab.gui.application import Astrolab
from astrolab.data.manager import DataManager
from astrolab.graph.gpu import gpActivationFlow

app = Astrolab.instance()
app.configure()
n_neighbors = 5
t0 = time.time()

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
# table_cols = project_dataset.attrs['colnames']

graph_data: xa.DataArray = project_dataset["reduction"]
activation_flow = gpActivationFlow( graph_data, n_neighbors )

