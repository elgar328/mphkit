"""Geometry creation, geometry features, and coordinate systems."""
from __future__ import annotations

import numpy
from mph.model import Model
from mph.node import Node
from mph.node import escape

from . import _comsol
from ._comsol import WorkPlaneNode


def geometry(model: Model, dim: int = 3, *, length_unit: str | None = None,
             name: str | None = None) -> Node:
    """
    Creates a new component with a geometry of dimension `dim`.

    The component is created explicitly, so this also works in models that
    already have components. `length_unit` is, for example, `'mm'`; plain
    numbers given to other helpers are then interpreted in that unit.
    Returns the geometry node.
    """
    java = model.java
    taken = _comsol.labels(java.geom())
    if name is not None:
        _comsol.pick_label(name, name, taken)
    ctag = str(java.component().uniquetag('comp'))
    component = java.component().create(ctag, True)
    gtag = str(java.geom().uniquetag('geom'))
    geom = component.geom().create(gtag, dim)
    label = _comsol.pick_label(name, str(geom.label()), taken)
    geom.label(label)
    if length_unit is not None:
        geom.lengthUnit(length_unit)
    node = model/'geometries'/escape(label)
    _comsol.check_tag(node, gtag)
    return node


def component_of(geom: Node) -> Node:
    """
    Returns the component node that contains the geometry (does not create).

    Useful for the parts of a model that live under the component, e.g.
    `mk.component_of(geom).java.coordSystem()`. For coordinate systems use
    `coordinate_system()`.
    """
    java = _comsol.component_of(geom)
    node = geom.model/'components'/escape(str(java.label()))
    _comsol.check_tag(node, str(java.tag()))
    return node


def coordinate_system(geom: Node, type: str, /, *, selection: Node | None = None,
                      name: str | None = None, **properties) -> Node:
    """
    Creates a coordinate system of `type` for the geometry.

    For example a perfectly matched layer:
    `coordinate_system(geom, 'PML', selection=pml, stretchingType='rational')`.
    Other types include `'InfiniteElement'`, `'AbsorbingLayer'` and
    `'Scaling'`, which take a domain `selection`, and `'Rotated'`,
    `'Cylindrical'`, `'Spherical'` or `'Boundary'`, which do not. Keyword
    arguments are COMSOL property names. Returns the node under
    `model/'coordinates'`.

    Plain MPh works as well when the geometry tag comes first,
    `(model/'coordinates').create(geom.tag(), 'PML')`. Without it, COMSOL
    leaves a broken node that makes every later physics node fail.
    """
    model = geom.model
    container = _comsol.component_of(geom).coordSystem()
    everything = model.java.coordSystem()
    tag, label = _comsol.create_java(container, 'coordinates', type, name,
                                     _comsol.labels(everything),
                                     tags=everything, args=(geom.tag(),))
    node = model/'coordinates'/escape(label)
    try:
        _comsol.check_tag(node, tag)
        java = container.get(tag)
        _comsol.set_properties(java, properties)
        if selection is not None:
            try:
                target = java.selection()
            except Exception:
                raise ValueError(f'A "{type}" coordinate system has no '
                                 'selection.') from None
            chosen = _comsol.check_selection(geom, selection)
            if list(chosen.dimension()) != list(target.dimension()):
                raise ValueError(f'Selection "{selection}" is not at the '
                                 f'level a "{type}" coordinate system '
                                 'takes.')
            node.select(selection)
    except Exception:
        container.remove(tag)
        raise
    return node


def feature(parent: Node, type: str, /, *, name: str | None = None,
            **properties) -> Node:
    """
    Creates a geometry feature of any COMSOL `type` and sets its properties.

    `parent` is a geometry or a work plane. Keyword arguments are COMSOL
    property names. Input selections (such as `input` and `input2` of a
    Difference) accept geometry nodes, names, tags, or lists of those, or
    one selection: made with `sel.*(..., where='geometry')` (also at the
    `'object'` level), made in the work plane itself (`sel.box(plane,
    ...)`), or `sel.cumulative()`. Lists that
    mix numbers and expressions are converted for COMSOL. Arguments that
    are `None` are skipped.

    Every helper that creates a feature passes extra keyword arguments on
    to COMSOL, e.g. `contributeto=` to add the result to a cumulative
    selection (`sel.cumulative`).

    If setting a property fails, the new feature is removed again and the
    error is raised. Returns the feature node.
    """
    container = _comsol.feature_container(parent)
    workplane = _comsol.is_workplane(parent.java)
    taken = _comsol.labels(container)
    if (type.endswith('Selection')
            or properties.get('selresult') in (True, 'on')):
        taken |= _comsol.selection_labels(parent.model)
    tag, label = _comsol.create_java(container, 'geometries', type, name,
                                     taken)
    cls = WorkPlaneNode if workplane or type == 'WorkPlane' else Node
    node = _comsol.child_node(parent, label, cls)
    try:
        _comsol.check_tag(node, tag)
        java = container.get(tag)
        _comsol.set_properties(java, properties, parent, container)
    except Exception:
        container.remove(tag)
        raise
    return node


#############################
# Primitives and operations #
#############################

def block(geom: Node, /, size=None, pos=None, *, name: str | None = None,
          **properties) -> Node:
    """
    Creates a Block. `base='center'` centers it on `pos`.

    Layers, e.g. a perfectly matched layer, are shells on chosen faces:
    `layername=['pml'], layer=[5], layertop=True, layerbottom=False`. The
    faces are `layerleft`/`layerright` (−x/+x), `layerfront`/`layerback`
    (−y/+y) and `layerbottom`/`layertop` (−z/+z); only `layerbottom` is on
    by default. Several layers stack from the outer face inward in the
    order of `layername`. Select them with `sel.layer()`.
    """
    return feature(geom, 'Block', name=name, size=size, pos=pos,
                   **properties)


def cylinder(geom: Node, /, r=None, h=None, pos=None, *, name: str | None = None,
             **properties) -> Node:
    """
    Creates a Cylinder with radius `r` and height `h`.

    The cylinder stands on `pos` along the z axis; COMSOL's `axistype` and
    `axis` properties give another direction, e.g. `axistype='x'`.
    """
    return feature(geom, 'Cylinder', name=name, r=r, h=h, pos=pos,
                   **properties)


def sphere(geom: Node, /, r=None, pos=None, *, name: str | None = None,
           **properties) -> Node:
    """Creates a Sphere with radius `r` centered at `pos`."""
    return feature(geom, 'Sphere', name=name, r=r, pos=pos, **properties)


def point(geom: Node, /, p, *, name: str | None = None, **properties) -> Node:
    """Creates a Point at coordinates `p`."""
    return feature(geom, 'Point', name=name, p=p, **properties)


def union(geom: Node, /, input, *, name: str | None = None, **properties) -> Node:
    """
    Creates a Union of the `input` objects. Works in a work plane as well.

    Objects that touch or overlap keep the boundaries between them: they
    stay separate domains, and an overlap becomes a domain of its own (two
    overlapping blocks give three). This is COMSOL's default with or
    without a union (`intbnd` is on, and a geometry ends with a Form
    Union). Pass `intbnd=False` so that touching or overlapping inputs
    become one domain, e.g. a boss standing on a plate. This removes every
    interior boundary of the inputs, block layers and partition cuts
    included; outer faces may stay split where the inputs met, and objects
    that do not touch stay separate.
    """
    return feature(geom, 'Union', name=name, input=input, **properties)


def difference(geom: Node, /, input, input2, *, name: str | None = None,
               **properties) -> Node:
    """
    Creates a Difference: `input` objects minus `input2` objects.

    Touching or overlapping `input` objects stay separate domains unless
    `intbnd=False`, see `union()`.
    """
    return feature(geom, 'Difference', name=name, input=input,
                   input2=input2, **properties)


def rigid_transform(geom: Node, /, input, *, name: str | None = None,
                    **properties) -> Node:
    """
    Creates a Rigid Transform (move and rotate) of the `input` objects.

    For example `displ=(dx, dy, dz)`, `specify='eulerang'`,
    `eulerang=(180, 90, 0)`.
    """
    return feature(geom, 'RigidTransform', name=name, input=input,
                   **properties)


def intersection(parent: Node, /, input, *, name: str | None = None,
                 **properties) -> Node:
    """
    Creates an Intersection: the part the `input` objects share.

    `intbnd=False` removes interior boundaries the inputs carry, such as
    block layers; see `union()`.
    """
    return feature(parent, 'Intersection', name=name, input=input,
                   **properties)


def delete(parent: Node, /, input, *, name: str | None = None,
           **properties) -> Node:
    """
    Creates a Delete of the `input` objects.

    `input` may also be one selection: at the `'object'` level or a
    cumulative selection, whole objects are deleted; a domain or boundary
    selection made with `sel.*(..., where='geometry')` deletes just those.
    In a work plane, use a selection made in that plane.
    """
    return feature(parent, 'Delete', name=name, input=input, **properties)


##############
# Transforms #
##############

def array(parent: Node, /, input, *, size, displ, name: str | None = None,
          **properties) -> Node:
    """
    Creates an Array of copies of the `input` objects.

    `size` is the number of copies per axis, e.g. `size=(5, 5, 1)`, or a
    single number (or expression) for a linear array along `displ`.
    `displ` is the spacing, e.g. `displ=(10, 10, 0)`. Both are
    keyword-only so they cannot be swapped by accident. `size` becomes
    COMSOL's `fullsize` or, for a linear array, `linearsize` with
    `type='linear'`. Works in a work plane as well.
    """
    if isinstance(size, bool):
        raise TypeError(f'size must be a number or one per axis, not {size!r}.')
    _comsol.check_vector(parent, 'displ', displ)
    if isinstance(size, (list, tuple, numpy.ndarray)):
        _comsol.check_vector(parent, 'size', size)
        properties['fullsize'] = size
    else:
        properties.setdefault('type', 'linear')
        properties['linearsize'] = size
    return feature(parent, 'Array', name=name, input=input, displ=displ,
                   **properties)


def move(parent: Node, /, input, displ, *, name: str | None = None,
         **properties) -> Node:
    """
    Creates a Move of the `input` objects by `displ`, e.g. `(0, 0, 5)`.

    A component given as a list makes several copies, e.g.
    `displ=([10, 20], 0, 0)`. Pass `keep=True` to keep the originals.
    """
    _comsol.check_vector(parent, 'displ', displ)
    for axis, value in zip('xyz', displ):
        properties[f'displ{axis}'] = value
    return feature(parent, 'Move', name=name, input=input, **properties)


def rotate(parent: Node, /, input, rot, *, pos=None, axis=None,
           name: str | None = None, **properties) -> Node:
    """
    Creates a Rotate of the `input` objects by `rot` degrees.

    A list of angles makes several copies; pass `keep=True` to keep the
    originals. `pos` is a point on the axis.
    In 3D, `axis` is `'x'`, `'y'`, `'z'` (default) or a direction vector;
    2D rotations (also in a work plane) have no axis.
    """
    if pos is not None:
        _comsol.check_vector(parent, 'pos', pos)
    if axis is not None:
        if _comsol.parent_dim(parent) == 2:
            raise ValueError('A 2D rotation has no axis; leave out axis.')
        if not isinstance(axis, str):
            _comsol.check_vector(parent, 'axis', axis)
        properties.update(_comsol.axis_properties(axis))
    return feature(parent, 'Rotate', name=name, input=input, rot=rot,
                   pos=pos, **properties)


def mirror(parent: Node, /, input, axis, *, pos=None, name: str | None = None,
           **properties) -> Node:
    """
    Creates a Mirror of the `input` objects.

    `axis` is the **normal** of the mirror plane (3D) or line (2D), as in
    COMSOL: `axis=(1, 0, 0)` mirrors in the plane x = 0, so x changes
    sign. `pos` is a point on the plane. Pass `keep=True` to keep the
    originals.
    """
    _comsol.check_vector(parent, 'axis', axis)
    if pos is not None:
        _comsol.check_vector(parent, 'pos', pos)
    return feature(parent, 'Mirror', name=name, input=input, axis=axis,
                   pos=pos, **properties)


def revolve(geom: Node, /, input, angle=None, *, pos=None, axis=None,
            name: str | None = None, **properties) -> Node:
    """
    Creates a Revolve of `input`, usually a work plane, in a 3D geometry.

    `angle` is left out for a full turn, one value in degrees (from 0), or
    two values (start, end). The axis is given in the work plane's own
    coordinates by default: `pos` and `axis` with two values each, the
    work plane's y axis if left out (parallel to the global z axis for
    `quickplane='xz'` or `'yz'`). Three values, or `axis='x'|'y'|'z'`,
    give an axis in 3D coordinates instead (the model's x axis, not the
    work plane's); a 3D axis without `pos` passes through the origin.

    A full turn keeps the drawn cross-section as an interior face; with
    `angle` left out, `origfaces=False` drops it (with `angle=360` it
    stays). Every face the revolve sweeps, flat ones included, is split
    every 90° counted from the start angle: up to 90° gives one face, 100°
    two, a full turn four. As in `extrude()`, objects that touch in the
    plane become separate domains.
    """
    if _comsol.parent_dim(geom) != 3 or _comsol.is_workplane(geom.java):
        raise ValueError('Revolve needs a 3D geometry.')
    if angle is not None:
        if 'angle1' in properties or 'angle2' in properties:
            raise ValueError('Give either angle or angle1/angle2.')
        properties['angtype'] = 'specang'
        if isinstance(angle, (list, tuple, numpy.ndarray)):
            properties['angle1'], properties['angle2'] = angle
        else:
            properties['angle1'], properties['angle2'] = 0, angle
    if isinstance(axis, str):
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"axis must be 'x', 'y', 'z' or a vector, not "
                             f'{axis!r}.')
        axis = [int(a == axis) for a in 'xyz']
    sizes = {len(v) for v in (pos, axis) if v is not None}
    if len(sizes) > 1 or not sizes <= {2, 3}:
        raise ValueError('pos and axis need two values each (work plane) or '
                         'three each (3D).')
    if sizes == {3}:
        if axis is None:
            raise ValueError('A 3D pos needs an axis as well.')
        properties.update(axistype='3d', axis3=axis,
                          pos3=pos if pos is not None else (0, 0, 0))
    elif sizes == {2}:
        properties.update(axistype='2d', pos=pos, axis=axis)
    return feature(geom, 'Revolve', name=name, input=input, **properties)


def partition(parent: Node, /, input, tool, *, name: str | None = None,
              **properties) -> Node:
    """
    Creates a Partition of the `input` objects by `tool`.

    `tool` is geometry objects, or a work plane (node, tag or name) to cut
    along its plane. `keepinput`/`keeptool` keep the originals.
    """
    plane = _workplane_tag(parent, tool)
    if plane is not None:
        properties.update(partitionwith='workplane', workplane=plane)
    else:
        properties['tool'] = tool
    return feature(parent, 'Partition', name=name, input=input, **properties)


def _workplane_tag(parent: Node, ref) -> str | None:
    """Returns the tag if `ref` (node, tag or name) is a work plane."""
    if isinstance(ref, Node):
        if not _comsol.is_workplane(ref.java):
            return None
        if _comsol.geometry_of(ref) != _comsol.geometry_of(parent):
            raise ValueError(f'Work plane "{ref}" does not belong to '
                             f'"{parent}".')
        return ref.tag()
    if isinstance(ref, str):
        container = _comsol.feature_container(parent)
        try:
            tag = _comsol.resolve_object(container, ref)
        except LookupError:
            return None
        if (tag in [str(t) for t in container.tags()]
                and _comsol.is_workplane(container.get(tag))):
            return tag
    return None


def fillet(parent: Node, /, input, radius, *, name: str | None = None,
           **properties) -> Node:
    """
    Rounds edges (3D) or corners (2D and work planes) with `radius`.

    `input` is geometry objects, meaning all their edges or corners, or
    one selection of edges (3D) or points (2D) made with
    `sel.*(..., where='geometry')` or `sel.cumulative()`. In a work plane,
    pick single corners with a selection made in the plane, e.g.
    `fillet(plane, sel.box(plane, 'point', x=1, y=1), 0.3)`. This is COMSOL's
    Fillet3D `edge` in 3D and Fillet `point` in 2D. Array copies can be
    named like `'arr1(1,1,1)'`; such names are only checked when the
    geometry is built.
    """
    return _round(parent, 'Fillet', input, name, radius=radius,
                  **properties)


def chamfer(parent: Node, /, input, dist, *, name: str | None = None,
            **properties) -> Node:
    """
    Bevels edges (3D) or corners (2D and work planes) by `dist`.

    `input` works as in `fillet()`, including selections made in a work
    plane for single corners. This is COMSOL's Chamfer3D `edge` with the
    distance called `radius` in 3D, and Chamfer `point` with `dist` in 2D.
    """
    if _comsol.parent_dim(parent) == 3:
        return _round(parent, 'Chamfer', input, name, radius=dist,
                      **properties)
    return _round(parent, 'Chamfer', input, name, dist=dist, **properties)


def _round(parent: Node, kind: str, input, name, **properties) -> Node:
    """Creates a fillet or chamfer on edges (3D) or points (2D)."""
    dim = _comsol.parent_dim(parent)
    if dim == 3:
        return feature(parent, f'{kind}3D', name=name, edge=input,
                       **properties)
    if dim == 2:
        return feature(parent, kind, name=name, point=input, **properties)
    raise ValueError(f'A 1D geometry has no {kind.lower()}.')


def line_segment(parent: Node, /, start, end, *, name: str | None = None,
                 **properties) -> Node:
    """Creates a straight line segment from `start` to `end`."""
    _comsol.check_vector(parent, 'start', start)
    _comsol.check_vector(parent, 'end', end)
    return feature(parent, 'LineSegment', name=name, specify1='coord',
                   coord1=start, specify2='coord', coord2=end, **properties)


def interval(geom: Node, /, coord, *, name: str | None = None,
             **properties) -> Node:
    """
    Creates an Interval in a 1D geometry through the points `coord`.

    Consecutive points bound one domain each, e.g. `[0, 1, 3]` gives two.
    """
    if _comsol.parent_dim(geom) != 1:
        raise ValueError('An interval needs a 1D geometry.')
    if not isinstance(coord, (list, tuple, numpy.ndarray)) or len(coord) < 2:
        raise ValueError(f'coord needs at least two points, not {coord!r}.')
    return feature(geom, 'Interval', name=name, coord=coord, **properties)


###############
# Work planes #
###############

def workplane(geom: Node, /, *, name: str | None = None, **properties) -> Node:
    """
    Creates a Work Plane, for example `quickz=5`.

    By default the plane is the xy plane at z = 0 (`quickplane='xy'`,
    `quickz=0`). `quickplane` names the global axes that the plane's x and
    y lie on, e.g. `'yz'` or `'xz'`; the offset along the third axis is
    `quickx`, `quicky` or `quickz`. `extrude()` goes along x × y: +x for
    `'yz'`, +y for `'zx'`, but -y for `'xz'` (likewise -x for `'zy'` and
    -z for `'yx'`).

    With `unite=True` the plane's 2D objects are imprinted into the 3D
    geometry, e.g. to create an evaluation surface. Add 2D features with
    `square(wp, ...)`, `rectangle(wp, ...)`, `circle(wp, ...)` or
    `polygon(wp, ...)`.
    """
    return feature(geom, 'WorkPlane', name=name, **properties)


def square(parent: Node, /, size=None, pos=None, *, name: str | None = None,
           **properties) -> Node:
    """Creates a Square in a 2D geometry or a work plane."""
    return feature(parent, 'Square', name=name, size=size, pos=pos,
                   **properties)


def rectangle(parent: Node, /, size=None, pos=None, *, name: str | None = None,
              **properties) -> Node:
    """Creates a Rectangle `size=(width, height)` in 2D or a work plane."""
    return feature(parent, 'Rectangle', name=name, size=size, pos=pos,
                   **properties)


def circle(parent: Node, /, r=None, pos=None, *, name: str | None = None,
           **properties) -> Node:
    """Creates a Circle with radius `r` in 2D or a work plane."""
    return feature(parent, 'Circle', name=name, r=r, pos=pos, **properties)


def polygon(parent: Node, /, x, y, *, name: str | None = None,
            **properties) -> Node:
    """Creates a closed Polygon through the points `x`, `y`."""
    properties.setdefault('type', 'solid')
    return feature(parent, 'Polygon', name=name, source='vectors', x=x, y=y,
                   **properties)


def extrude(geom: Node, /, input, distance, *, name: str | None = None,
            **properties) -> Node:
    """
    Extrudes `input`, usually a work plane, by `distance`.

    The plane's 2D objects are extruded along its normal (see
    `workplane()`). `distance` is a length, or a list of distances from the
    plane, e.g. `[1, 3]` for layers from 0 to 1 and from 1 to 3.

    Objects that touch or overlap in the plane become separate domains;
    unite them there with `union(plane, [...], intbnd=False)` (uniting
    after the extrusion also gives one domain but keeps the top and bottom
    faces split).
    """
    if not isinstance(distance, (list, tuple)):
        distance = [distance]
    return feature(geom, 'Extrude', name=name, input=input,
                   distance=distance, **properties)


##########
# Import #
##########

def import_(geom: Node, file, /, *, type: str | None = None, name: str | None = None,
            **properties) -> Node:
    """
    Imports geometry from `file`.

    `type` defaults from the extension: `'native'` for `.mphbin`/`.mphtxt`,
    otherwise `'cad'` (STEP, IGES, Parasolid, ...; needs a CAD-capable
    license). The path is stored as an absolute path.
    """
    from pathlib import Path
    from .errors import LicenseError
    file = Path(file).resolve()
    if not file.exists():
        raise FileNotFoundError(f'File "{file}" does not exist.')
    if type is None:
        type = 'native' if file.suffix.lower() in ('.mphbin', '.mphtxt') \
            else 'cad'
    node = feature(geom, 'Import', name=name, type=type, filename=str(file),
                   **properties)
    try:
        # MPh's `Node.import_()` calls `discardData()`, which geometry
        # imports lack; `importData()` reads the file right away.
        _comsol.java_of(node).importData()
    except Exception as error:
        node.remove()
        if 'license' in str(error).lower():
            raise LicenseError(
                f'Importing "{file.name}" needs a license for CAD import '
                '(CAD Import Module, Design Module or a LiveLink).'
            ) from error
        raise
    return node
