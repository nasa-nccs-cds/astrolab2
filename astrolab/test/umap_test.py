import time
import xarray as xa
from astrolab.gui.application import Astrolab
from astrolab.data.manager import DataManager
from astrolab.reduction.base import UMAP
import cudf, cuml, cupy, cupyx

app = Astrolab.instance()
app.configure("spectraclass")
n_neighbors = 15
project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
umap_data: xa.DataArray = project_dataset["reduction"].compute()

umap = UMAP.instance()
embedding = umap.transform( umap_data )
print( f"Completed embedding, result shape = {embedding.shape}" )




