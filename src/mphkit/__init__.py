"""Helpers on top of MPh for building COMSOL geometries and selections."""

from . import sel
from ._measure import bounding_box, measure
from ._props import set_ as set  # not in __all__: keeps builtin set
from .errors import LicenseError
from .geometry import (array, block, chamfer, circle, component_of,
                       coordinate_system, cylinder, delete, difference,
                       extrude, feature, fillet, geometry, import_,
                       intersection, interval, line_segment, mirror, move,
                       partition, point, polygon, rectangle, revolve,
                       rigid_transform, rotate, sphere, square, union,
                       workplane)

__version__ = '0.1.0.dev0'

__all__ = ['LicenseError', 'array', 'block', 'bounding_box', 'chamfer',
           'circle', 'component_of', 'coordinate_system', 'cylinder',
           'delete', 'difference', 'extrude', 'feature', 'fillet',
           'geometry', 'import_', 'intersection', 'interval',
           'line_segment', 'measure', 'mirror', 'move', 'partition', 'point',
           'polygon', 'rectangle', 'revolve', 'rigid_transform', 'rotate',
           'sel', 'sphere', 'square', 'union', 'workplane']
