import numpy as np
import numpy.ma as ma
from .base import ActivationFlow
import xarray as xa
import numba
from typing import List, Union, Tuple, Optional, Dict
import os, time, threading, traceback

class gpActivationFlow(ActivationFlow):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        ActivationFlow.__init__( nodes_data, n_neighbors, **kwargs )
        self.I: np.ndarray = None
        self.D: np.ndarray = None
        self.P: np.ndarray = None
        self.C: np.ndarray = None
        self.setNodeData( nodes_data, **kwargs )

    def setNodeData(self, nodes_data: xa.DataArray, **kwargs ):
        if self.reset or (self.nodes is None):
            if (nodes_data.size > 0):
                t0 = time.time()
                self.nodes = nodes_data
                self.nnd = self.getNNGraph( nodes_data, self.nneighbors,  **kwargs )
                self.I = self.nnd.neighbor_graph[0]
                self.D = self.nnd.neighbor_graph[1]
                dt = (time.time()-t0)
                print( f"Computed NN Graph with {self.nnd.n_neighbors} neighbors and {nodes_data.shape[0]} verts in {dt} sec ({dt/60} min)")
            else:
                print( "No data available for this block")

    @classmethod
    def getNNGraph(cls, nodes: xa.DataArray, n_neighbors: int, **kwargs ):
        n_trees = kwargs.get('ntree', 5 + int(round((nodes.shape[0]) ** 0.5 / 20.0)))
        n_iters = kwargs.get('niter', max(5, 2 * int(round(np.log2(nodes.shape[0])))))
        nnd = NNDescent(nodes.values, n_trees=n_trees, n_iters=n_iters, n_neighbors=n_neighbors, max_candidates=60, verbose=True)
        return nnd

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        sample_mask = sample_data == 0
        if self.C is None or self.reset:
            self.C = np.array( sample_data, dtype=np.int32 )
        else:
            self.C = np.where( sample_mask, self.C, sample_data )
        label_count = np.count_nonzero(self.C)
        if label_count == 0:
            print( "Workflow violation: Must label some points before this algorithm can be applied"  )
            return None
        if (self.P is None) or self.reset:   self.P = np.full( self.C.shape, float('inf'), dtype=np.float32 )
        self.P = np.where( sample_mask, self.P, 0.0 )
        print(f"Beginning graph flow iterations, #C = {label_count}")
        t0 = time.time()
        converged = False
        for iter in range(nIter):
            try:
                iterate_spread_labels( self.I, self.D, self.C, self.P )
                new_label_count = np.count_nonzero(self.C)
                if new_label_count == label_count:
                    print( "Converged!" )
                    converged = True
                    break
                else:
                    label_count = new_label_count
 #                   print(f"\n -->> Iter{iter + 1}: #C = {label_count}\n")
            except Exception as err:
                print(f"Error in graph flow iteration {iter}:")
                traceback.print_exc(50)
                break

        t1 = time.time()
        print(f"Completed graph flow {nIter} iterations in {(t1 - t0)} sec, #marked = {np.count_nonzero(self.C)}")
        self.reset = False
        return converged


