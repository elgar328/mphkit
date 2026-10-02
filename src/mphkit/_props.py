"""
Setting properties of any COMSOL object. Public as `mk.set`.

Calling a Java `set()` directly fails for plain Python ints ("Ambiguous
overloads"), numpy arrays and lists that mix numbers and expressions;
`mk.set` converts them the same way the other helpers do.
"""
from __future__ import annotations

from mph.node import Node, join

from . import _catalog, _comsol
from ._comsol import WorkPlaneNode


def set_(target, /, **properties):
    """
    Sets properties of an MPh node or a Java object, and returns it.

    For example on a mesh size feature,
    `mk.set(size, hmax='L/10', hgrad=1.45)`. The target may be any MPh
    node, or a Java object MPh does not reach, such as a probe, a material
    function or `physics.java.prop('ShapeProperty')`. Keyword arguments
    are COMSOL property names, set in the given order, which matters:
    `custom=False` after `hmax` restores the predefined sizes (setting
    `hmax` alone turns `custom` on). `None` values are skipped.

    Ints, numpy arrays and lists mixing numbers and expressions are
    converted for COMSOL; lists of numbers become string arrays. Unknown
    names raise `ValueError`, with a suggestion where one is close and,
    on features of geometries, physics, meshes and studies and on
    materials, the call that lists the properties; objects
    with no property list (variables) or an empty one (a new material
    property group) do not check names. Invalid choices list the allowed
    values. If one property fails, the ones before it stay set.

    A geometry feature node also takes input selections (`input`,
    `input2`) as in `feature()`; build the geometry again afterwards. To
    turn on a result selection use `sel.result()` rather than `selresult`,
    because it checks for label clashes. On variables
    (`component.variable()`) any value is taken as an expression, and
    `True` becomes 1.
    """
    owner = container = listing = None
    if isinstance(target, Node):
        if _catalog.lists_properties(target):
            listing = 'mk.properties(node, search=...) lists them'
        if (len(target.path) >= 4 and target.path[0] == 'geometries'
                and not isinstance(target, WorkPlaneNode)):
            # A plain node cannot resolve features inside a work plane.
            target = WorkPlaneNode(target.model, join(target.path))
        java = _comsol.java_of(target)
        if len(target.path) >= 3 and target.path[0] == 'geometries':
            owner = _comsol.parent_of(target)
            container = _comsol.feature_container(owner)
    elif hasattr(target, 'set'):
        java = target
        for key, value in properties.items():
            if value is not None and _comsol.is_selection_input(java, key):
                raise TypeError(
                    f'"{key}" is an input selection; use '
                    f'java.selection("{key}"), or pass a geometry feature '
                    'as an MPh node.')
    else:
        raise TypeError(f'Cannot set properties of {target!r}; expected an '
                        'MPh node or a COMSOL Java object.')
    _comsol.set_properties(java, properties, owner, container, listing)
    return target
