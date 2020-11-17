import time
import xarray as xa
from astrolab.gui.application import Astrolab
from astrolab.data.manager import DataManager
import cudf, cuml, cupy, cupyx

app = Astrolab.instance()
app.configure("spectraclass")
n_neighbors = 5
project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
umap_data: xa.DataArray = project_dataset["reduction"].compute()

t0 = time.time()
print( f"Computing embedding, input shape = {umap_data.shape}" )
input_data = cudf.DataFrame({ icol : umap_data[:,icol] for icol in range(umap_data.shape[1]) } )
reducer = cuml.UMAP( n_neighbors=15, n_components=3, n_epochs=500, min_dist=0.1, output_type="numpy" )
embedding = reducer.fit_transform( input_data )
print( f"Completed embedding in time {time.time()-t0} sec, embedding shape = {embedding.shape}" )




