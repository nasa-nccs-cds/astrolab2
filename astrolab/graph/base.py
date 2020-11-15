import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict

class ActivationFlow(object):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        self.nneighbors = n_neighbors
        self.reset = True
        self.setNodeData( nodes_data, **kwargs )

    def clear(self):
        self.reset = True

    def setNodeData(self, nodes_data: xa.DataArray, **kwargs ):
        raise NotImplementedError()

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        raise NotImplementedError()

    @classmethod
    def instance(cls, point_data: xa.DataArray, nneighbors: int, **kwargs ):
        from astrolab.data.manager import DataManager
        if DataManager.proc_type == "cpu":
            from .cpu import cpActivationFlow
            return cpActivationFlow( point_data, nneighbors, **kwargs )
        elif DataManager.proc_type == "gpu":
            from .gpu import gpActivationFlow
            return gpActivationFlow( point_data, nneighbors, **kwargs )




