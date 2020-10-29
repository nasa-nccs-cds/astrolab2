import numpy as np
import ipywidgets as widgets
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, glob
import ipywidgets as ip
from functools import partial
from collections import OrderedDict
from astrolab.reduction.embedding import ReductionManager
from pathlib import Path
import xarray as xa
import traitlets as tl
import traitlets.config as tlc
from astrolab.model.base import AstroSingleton

class DataManager(tlc.SingletonConfigurable,AstroSingleton):
    reduce_method = tl.Unicode("Autoencoder").tag(config=True)
    reduce_nepochs = tl.Int( 2 ).tag(config=True)
    cache_dir = tl.Unicode("~/Development/Cache").tag(config=True)
    data_dir = tl.Unicode("~/Development/Data").tag(config=True)
    dataset = tl.Unicode("").tag(config=True)
    model_dims = tl.Int(16).tag(config=True)
    subsample = tl.Int( 5 ).tag(config=True)

    def __init__(self, mode: str = None, **kwargs):
        super(DataManager, self).__init__(**kwargs)
        from astrolab.gui.application import Astrolab
        self.datasets = {}
        self._mode = mode if mode is not None else Astrolab.instance().mode
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress = None
        self._dset_selection: ip.Select = None
        self.dataset = "NONE"

    @property
    def mode(self):
        return self._mode

    @classmethod
    def getXarray( cls, id: str, xcoords: Dict, subsample: int, xdims:OrderedDict, **kwargs ) -> xa.DataArray:
        np_data: np.ndarray = DataManager.instance().getInputFileData( id, subsample, tuple(xdims.keys()) )
        dims, coords = [], {}
        for iS in np_data.shape:
            coord_name = xdims[iS]
            dims.append( coord_name )
            coords[ coord_name ] = xcoords[ coord_name ]
        attrs = { **kwargs, 'name': id }
        return xa.DataArray( np_data, dims=dims, coords=coords, name=id, attrs=attrs )

    def get_input_mdata(self):
        if self._mode == "swift":
            return dict(embedding='scaled_specs', directory=["target_names", "obsids"], plot=dict(y="specs", x='spectra_x_axis'))
        elif self._mode == "tess":
            return dict(embedding='scaled_lcs', directory=['tics', "camera", "chip", "dec", 'ra', 'tmag'], plot=dict(y="lcs", x='times'))
        else:
            raise Exception( f"Unknown data mode: {self._mode}, should be 'tess' or 'swift")

    def prepare_inputs( self, *args ):
        self._progress.value = 0.02
        self.model_dims = self._model_dims_selector.value
        self.subsample = self._subsample_selector.value
        file_name = f"raw" if self.reduce_method == "None" else f"{self.reduce_method}-{self.model_dims}"
        if self.subsample > 1: file_name = f"{file_name}-ss{self.subsample}"
        output_file = os.path.join( self.datasetDir, file_name + ".nc" )

        input_vars = self.get_input_mdata()
        np_embedding = self.getInputFileData( input_vars['embedding'], self.subsample )
        dims = np_embedding.shape
        mdata_vars = list(input_vars['directory'])
        xcoords = OrderedDict( samples = np.arange( dims[0] ), bands = np.arange(dims[1]) )
        xdims = OrderedDict( { dims[0]: 'samples', dims[1]: 'bands' } )
        data_vars = dict( embedding = xa.DataArray( np_embedding, dims=xcoords.keys(), coords=xcoords, name=input_vars['embedding'] ) )
        data_vars.update( { vid: self.getXarray( vid, xcoords, self.subsample, xdims ) for vid in mdata_vars } )
        pspec = input_vars['plot']
        data_vars.update( { f'plot-{vid}': self.getXarray( pspec[vid], xcoords, self.subsample, xdims, norm=pspec.get('norm','')) for vid in [ 'x', 'y' ] } )
        self._progress.value = 0.1
        if self.reduce_method != "None":
           reduced_spectra = ReductionManager.instance().reduce( data_vars['embedding'], self.reduce_method, self.model_dims, self.reduce_nepochs )
           coords = dict( samples=xcoords['samples'], model=np.arange( self.model_dims ) )
           data_vars['reduction'] =  xa.DataArray( reduced_spectra, dims=['samples','model'], coords=coords )
           self._progress.value = 0.8

        dataset = xa.Dataset( data_vars, coords=xcoords, attrs = {'type':'spectra'} )
        dataset.attrs["colnames"] = mdata_vars
        print( f"Writing output to {output_file}" )
        dataset.to_netcdf( output_file, format='NETCDF4', engine='netcdf4' )
        self.updateDatasetList()
        self._progress.value = 1.0

    def updateDatasetList(self):
        self._dset_selection.options = self.getDatasetList()

    def select_dataset(self, *args ):
        from astrolab.gui.application import Astrolab
        if self.dataset != self._dset_selection.value:
            print(f"Loading dataset '{self._dset_selection.value}', current dataset = '{self.dataset}'")
            self.dataset = self._dset_selection.value
            Astrolab.instance(self._mode).refresh()

    def getSelectionPanel(self) -> ip.HBox:
        dsets: List[str] = self.getDatasetList()
        self._dset_selection: ip.Select = ip.Select( options = dsets, description='Datasets:',disabled=False )
        if len( dsets ) > 0: self._dset_selection.value = dsets[0]
        load: ip.Button = ip.Button( description="Load")
        load.on_click( self.select_dataset )
        filePanel: ip.HBox = ip.HBox( [self._dset_selection, load ], layout=ip.Layout( width="100%", height="100%" ), border= '2px solid firebrick' )
        return filePanel

    def getCreationPanel(self) -> ip.HBox:
        load: ip.Button = ip.Button( description="Create", layout=ip.Layout( flex='1 1 auto' ) )
        self._model_dims_selector: ip.SelectionSlider = ip.SelectionSlider( options=range(3,50), description='Model Dimension:', value=self.model_dims, layout=ip.Layout( width="auto" ),
                                                   continuous_update=True, orientation='horizontal', readout=True, disabled=False  )

        self._subsample_selector: ip.SelectionSlider = ip.SelectionSlider( options=range(1,50), description='Subsample:', value=self.subsample, layout=ip.Layout( width="auto" ),
                                                   continuous_update=True, orientation='horizontal', readout=True, disabled=False  )

        load.on_click( self.prepare_inputs )
        self._progress = ip.FloatProgress( value=0.0, min=0, max=1.0, step=0.01, description='Progress:', bar_style='info', orientation='horizontal', layout=ip.Layout( flex='1 1 auto' ) )
        button_hbox: ip.HBox = ip.HBox( [ load,self._progress ], layout=ip.Layout( width="100%", height="auto" ) )
        creationPanel: ip.VBox = ip.VBox( [ self._model_dims_selector,self._subsample_selector, button_hbox ], layout=ip.Layout( width="100%", height="100%" ), border= '2px solid firebrick' )
        return creationPanel

    def gui( self, **kwargs ) -> widgets.Tab():
        wTab = widgets.Tab( layout = ip.Layout( width='auto', height='auto' ) )
        selectPanel = self.getSelectionPanel()
        creationPanel = self.getCreationPanel()
        wTab.children = [ creationPanel, selectPanel ]
        wTab.set_title( 0, "Create New")
        wTab.set_title( 1, "Select Existing")
        return wTab

    def getInputFileData(self, input_file_id: str, subsample: int = 1, dims: Tuple[int] = None ):
        input_file_path = os.path.expanduser( os.path.join( self.data_dir, "astrolab", self._mode, f"{input_file_id}.pkl") )
        try:
            if os.path.isfile(input_file_path):
                print(f"Reading unstructured {input_file_id} data from file {input_file_path}")
                with open(input_file_path, 'rb') as f:
                    result = pickle.load(f)
                    if isinstance( result, np.ndarray ):
                        if dims is not None and (result.shape[0] == dims[1]) and result.ndim == 1: return result
                        return result[::subsample]
                    elif isinstance( result, list ):
                        if dims is not None and ( len(result) == dims[1] ): return result
                        subsampled = [ result[i] for i in range( 0, len(result), subsample ) ]
                        if isinstance( result[0], np.ndarray ):  return np.vstack( subsampled )
                        else:                                    return np.array( subsampled )
            else:
                print( f"Error, the input path '{input_file_path}' is not a file.")
        except Exception as err:
            print(f" Can't read data[{input_file_id}] file {input_file_path}: {err}")

    def loadDataset( self, dsid: str, *args, **kwargs ) -> xa.Dataset:
        if dsid is None: return None
        if dsid not in self.datasets:
            data_file = os.path.join( self.datasetDir, dsid + ".nc" )
            dataset: xa.Dataset = xa.open_dataset( data_file )
            print( f"Opened Dataset {dsid} from file {data_file}")
            dataset.attrs['dsid'] = dsid
            dataset.attrs['type'] = 'spectra'
            self.datasets[dsid] = dataset
        return self.datasets[dsid]

    def getDatasetList(self):
        dset_glob = os.path.expanduser(f"{self.datasetDir}/*.nc")
        print( f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter(os.path.isfile, glob.glob( dset_glob ) ) )
        files.sort( key=lambda x: os.path.getmtime(x), reverse=True )
        return [ Path(f).stem for f in files ]

    def loadCurrentProject(self) -> xa.Dataset:
        self.select_dataset()
        return self.loadDataset( self.dataset )

    @property
    def datasetDir(self):
        dsdir = os.path.join( os.path.expanduser( self.cache_dir ), "astrolab", self._mode )
        os.makedirs( dsdir, exist_ok=True )
        return dsdir