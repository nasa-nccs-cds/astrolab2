import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict

class ActivationFlow(object):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        self.nneighbors = n_neighbors
        self.nodes: xa.DataArray = None
        self.reset = True
        self.setNodeData( nodes_data, **kwargs )

    def clear(self):
        self.reset = True

    def setNodeData(self, nodes_data: xa.DataArray, **kwargs ):
        raise NotImplementedError()

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        raise NotImplementedError()


