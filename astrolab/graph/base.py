import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict

class ActivationFlow(object):
    _instance: "ActivationFlow" = None

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
    def instance(cls, point_data: xa.DataArray, nneighbors: int, **kwargs ) -> "ActivationFlow":
        from astrolab.data.manager import DataManager
        if cls._instance is None:
            if DataManager.proc_type == "cpu":
                from .cpu import cpActivationFlow
                cls._instance = cpActivationFlow( point_data, nneighbors, **kwargs )
            elif DataManager.proc_type == "gpu":
                from .gpu import gpActivationFlow
                cls._instance =  gpActivationFlow( point_data, nneighbors, **kwargs )
            else:
                print( f"Error, unknown proc_type: {DataManager.proc_type}")
        return cls._instance




