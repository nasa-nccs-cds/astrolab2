from typing import List, Union, Tuple, Optional, Dict, Callable
from IPython.core.debugger import set_trace
from collections import OrderedDict
import os, ipywidgets as ipw
import traitlets.config as tlc
import traitlets as tl
from astrolab.model.base import AstroSingleton

class Astrolab( tlc.SingletonConfigurable, AstroSingleton ):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    name = tl.Unicode('astrolab').tag(config=True,sync=True)
    config_file = tl.Unicode().tag(config=True,sync=True)
    custom_theme = False

    @tl.default('config_file')
    def _default_config_file(self):
        return os.path.join( os.path.expanduser("~"), "." + self.name, "configuration.py" )

    def __init__(self,  **kwargs ):
        from astrolab.data.manager import DataManager
        super(Astrolab, self).__init__( **kwargs )
        self._mode = DataManager.instance().mode

    @property
    def mode(self):
        return self._mode

    def configure(self):
        from traitlets.config.loader import load_pyconfig_files
        if os.path.isfile( self.config_file ):
            print(f"Loading config file: {self.config_file}")
            (dir, fname) = os.path.split( self.config_file )
            config = load_pyconfig_files( [ fname ], dir )
            for clss in self.config_classes:
                clss.instance().update_config(config)

    def save_config(self):
        conf_txt = AstroSingleton.generate_config_file()
        cfg_dir = os.path.dirname(os.path.realpath( self.config_file ) )
        os.makedirs( cfg_dir, exist_ok=True )
        with open( self.config_file, "w" ) as cfile_handle:
            print( f"Writing config file: {self.config_file}")
            cfile_handle.write( conf_txt )

    def getControlPanel( self ) -> ipw.DOMWidget:
        from astrolab.gui.control import ActionsPanel
        file: ipw.HBox = self.getFilePanel()
        actions: ipw.HBox = ActionsPanel.instance().gui()
        gui = ipw.VBox([file, actions], layout = ipw.Layout( width="100%" )  )
        return gui

    def process_menubar_action(self, mname, dname, op, b ):
        print(f" process_menubar_action.on_value_change: {mname}.{dname} -> {op}")

    @classmethod
    def set_astrolab_theme(cls):
        from IPython.display import display, HTML
        if cls.custom_theme:
            theme_file = os.path.join( cls.HOME, "themes", "astrolab.css" )
            with open( theme_file ) as f:
                css = f.read().replace(';', ' !important;')
            display(HTML('<style type="text/css">%s</style>Customized changes loaded.' % css))

    def gui( self, embed: bool = False ):
        from astrolab.gui.graph import GraphManager
        from astrolab.gui.points import PointCloudManager
        from astrolab.gui.table import TableManager
        from astrolab.gui.control import ActionsPanel

        self.set_astrolab_theme()
        self.configure()
        css_border = '1px solid blue'

        tableManager = TableManager.instance()
        graphManager = GraphManager.instance()
        pointCloudManager = PointCloudManager.instance()

        table = tableManager.gui()
        graph = graphManager.gui()
        points = pointCloudManager.instance().gui()

        tableManager.add_selection_listerner(graphManager.on_selection)
        tableManager.add_selection_listerner(pointCloudManager.on_selection)
        actionsPanel = ActionsPanel.instance().gui()

        control = ipw.VBox([actionsPanel, table], layout=ipw.Layout( flex='0 0 600px', border=css_border) )
        plot = ipw.VBox([points, graph], layout=ipw.Layout( flex='1 1 auto', border=css_border) )
        gui = ipw.HBox([control, plot])
        self.save_config()
        if embed: ActionsPanel.instance().embed()
        return gui

    def refresh(self):
        from astrolab.gui.graph import GraphManager
        from astrolab.gui.points import PointCloudManager
        from astrolab.gui.table import TableManager
        from astrolab.model.labels import LabelsManager
        TableManager.instance().refresh()
        GraphManager.instance().refresh()
        PointCloudManager.instance().refresh()
        LabelsManager.instance().refresh()
        self.save_config()
        print( "Refreshed Application")

    def __delete__(self, instance):
        self.save_config()





