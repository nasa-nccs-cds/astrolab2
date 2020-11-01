import traitlets.config as tlc
from collections import OrderedDict
import numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any
import traitlets as tl

class Marker:
    def __init__(self,  pids: np.ndarray, cid: int ):
        self.cid = cid
        self.pids = np.unique( pids )

    def isTransient(self):
        return self.cid == 0

    def isEmpty(self):
        return self.pids.size == 0

    def deletePid( self, pid: int ) -> bool:
        try:
            self.pids = self.pids[ self.pids != pid ]
            return True
        except: return False

    def deletePids( self, dpids: np.ndarray ) -> bool:
        try:
            self.pids = np.setdiff1d( self.pids, dpids )
            return True
        except: return False

class AstroSingleton:
    config_classes = []

    def __init__(self, **kwargs ):
        self.config_classes.append( self.__class__ )

    @classmethod
    def generate_config_file( cls ):
        trait_values: Dict = {}
        for clss in cls.config_classes:
            cfg_traits = clss.class_traits(config=True).keys()
            instance: tlc.SingletonConfigurable = clss.instance()
            cname = f"c.{clss.__name__}."
            cfg_trait_values = { cname+key: value for key,value in instance.trait_values().items() if key in cfg_traits }
            trait_values.update( cfg_trait_values )
        lines = ['']
        for name, value in trait_values.items():
            lines.append( f"{name} = {value}")
        return '\n'.join(lines)

    @classmethod
    def _classes_inc_parents(cls):
        """Iterate through configurable classes, including configurable parents """
        seen = set()
        for c in cls.config_classes:
            # We want to sort parents before children, so we reverse the MRO
            for parent in reversed(c.mro()):
                if issubclass(parent, tlc.Configurable) and (parent not in seen):
                    seen.add(parent)
                    yield parent

    @classmethod
    def _classes_with_config_traits(cls):
        """ Yields only classes with configurable traits, and their subclasses.  """
        cls_to_config = OrderedDict( (cls, bool(cls.class_own_traits(config=True))) for cls in cls._classes_inc_parents())

        def is_any_parent_included(cls):
            return any(b in cls_to_config and cls_to_config[b] for b in cls.__bases__)

        ## Mark "empty" classes for inclusion if their parents own-traits, and loop until no more classes gets marked.
        while True:
            to_incl_orig = cls_to_config.copy()
            cls_to_config = OrderedDict( (cls, inc_yes or is_any_parent_included(cls)) for cls, inc_yes in cls_to_config.items())
            if cls_to_config == to_incl_orig:
                break
        for cl, inc_yes in cls_to_config.items():
            if inc_yes:
                yield cl