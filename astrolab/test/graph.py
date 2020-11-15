import cudf, cuml, cupy, cupyx
from cuml.neighbors import NearestNeighbors
import time
import xarray as xa
import pandas as pd
from astrolab.gui.application import Astrolab
from astrolab.data.manager import DataManager
from astrolab.graph.gpu import gpActivationFlow

app = Astrolab.instance()
app.configure()
n_neighbors = 5

t0 = time.time()
nrows = 10

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("test")
table_cols = project_dataset.attrs['colnames']

graph_data: xa.DataArray = project_dataset["reduction"]

activation_flow = gpActivationFlow( graph_data, n_neighbors )

