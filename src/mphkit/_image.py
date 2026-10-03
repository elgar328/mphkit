"""
Pictures of geometries, selections and meshes. Public as `mk.image`.

COMSOL keeps an image export per geometry and per selection, saves it with
the model and records every change in the model history. So `image()`
draws a temporary selection instead, removed afterwards, and switches the
history off meanwhile: the model is left as it was. Mesh pictures, and the
result plots of `mk.plot`, go through a temporary plot group and image
export instead (`export()` below), removed the same way.
"""
from __future__ import annotations

import numbers
import os
from pathlib import Path

import numpy
from mph.node import Node

from . import _comsol, _mesh

# Suffix → COMSOL image type and the property holding its file name.
FORMATS = {'.png': ('png', 'pngfilename'), '.jpg': ('jpeg', 'jpegfilename'),
           '.jpeg': ('jpeg', 'jpegfilename')}

# A new image export inherits the user's preferences, so everything that
# shapes the picture is set. `saveprefs` goes first so that the export
# does not write the preferences, `size` next as it loads a preset.
COMMON = {'saveprefs': 'off', 'size': 'manualweb', 'unit': 'px',
          'lockratio': 'off', 'resolution': '96', 'antialias': 'on',
          'target': 'file', 'background': 'color',
          'customcolor': '1, 1, 1', 'colortheme': 'globaltheme',
          'fontsize': '9', 'highprecisioncolor': 'off',
          'qualityactive': 'off'}

# Axes and grid on, logo and title off, by space dimension.
DECORATION = {
    3: {'options3d': 'on', 'grid': 'on', 'axisorientation': 'on',
        'logo3d': 'off', 'title3d': 'off'},
    2: {'options2d': 'on', 'axes2d': 'on', 'logo2d': 'off',
        'title2d': 'off'},
    1: {'options1d': 'on', 'axes1d': 'on', 'showgrid': 'on',
        'logo1d': 'off', 'title1d': 'off'},
}


def image(geom: Node, filename, /, selection: Node | None = None, *,
          zoom: str = 'all', size=(800, 600), labels: bool = False,
          mesh: bool | Node | str = False) -> Path:
    """
    Saves a picture of a geometry, of a selection highlighted on it, or of
    its mesh, and returns the file path: a way to look at a model without
    the COMSOL Desktop, also for an AI assistant that reads images.

    ```python
    mk.image(geom, 'geometry.png')
    mk.image(geom, 'wall.png', wall, labels=True)
    mk.image(geom, 'mesh.png', mesh=True)
    ```

    The view is the geometry's current one (isometric for a new 3D model,
    from above in 2D), with axes. A selection is drawn in blue, the rest
    of the geometry as lines. `zoom='selection'` zooms to the selection
    instead of showing the whole geometry; `size` is `(width, height)` in
    pixels; `labels=True` adds the numbers of the visible entities at the
    selection's level (entities hidden from view may have no number;
    `sel.entities` lists them all). The selection must be an entity
    selection, the kind physics uses, not an `'object'`-level one. The
    file type follows the suffix, `.png`, `.jpg` or `.jpeg`; COMSOL writes
    the file on the machine it runs on (by default this one) and creates
    missing folders. The geometry must be built. The model is left as it
    was.

    `mesh=True` draws the mesh instead, its elements coloured by quality
    (skewness, 1 is best), in 2D and 3D; it needs a mesh, not a solution.
    In 3D, the volume elements are drawn (their faces on the outside),
    or the surface elements of a boundary selection. With a selection
    (boundaries or domains in 3D, domains in 2D), only that part's mesh
    is drawn and the picture shows just that part, so `zoom` has no
    effect. If the geometry's component has several meshes,
    pass one as `mesh=`, by name as in `model.meshes()`, tag or node. A
    mesh whose settings changed since it was built raises: run
    `model.mesh()`.
    """
    _comsol.check_not_workplane(geom, 'image')
    if isinstance(filename, Node):
        raise TypeError('The file name comes second: '
                        'mk.image(geom, filename, selection).')
    path = picture_path(filename)
    if zoom not in ('all', 'selection'):
        raise ValueError(f"zoom must be 'all' or 'selection', not {zoom!r}.")
    width, height = picture_size(size)
    if mesh is not False and not isinstance(mesh, (bool, Node, str)):
        raise TypeError(f'mesh must be True, False, or a mesh name, tag or '
                        f'node, not {mesh!r}.')
    if mesh is not False:
        if labels:
            raise ValueError('labels=True shows entity numbers on the '
                             'geometry; leave it out with mesh=True.')
        if zoom == 'selection' and selection is None:
            raise ValueError("zoom='selection' needs a selection.")
        return _mesh_picture(geom, path, selection, mesh, (width, height))
    imagetype, name_key = FORMATS[path.suffix]
    _comsol.check_built(geom)
    sdim = _comsol.sdim(geom)
    if selection is None:
        if zoom == 'selection':
            raise ValueError("zoom='selection' needs a selection.")
        if labels:
            raise ValueError('labels=True needs a selection: the numbers '
                             'shown are those of its entities.')
        level, source = sdim, None
    else:
        level, source, _ = drawn_selection(geom, selection)
    view = geometry_view(geom)
    if view is None and labels:
        raise RuntimeError(f'Geometry "{geom}" has no view to show labels '
                           'in.')
    settings = {**COMMON, 'width': str(width), 'height': str(height),
                **DECORATION[sdim],
                'zooming': 'zoomtoselection' if zoom == 'selection'
                else 'zoomextents',
                # the rest as lines only when a selection is shown
                'renderwireframe': 'off' if source is None else 'on',
                'view': 'auto' if view is None else str(view.tag()),
                'imagetype': imagetype, name_key: str(path)}
    model = geom.model.java
    container = _comsol.component_of(geom).selection()
    with _comsol.history_off(model):
        tag = str(model.selection().uniquetag('img'))
        shown = None if view is None else str(view.getString('showlabels'))
        try:
            temporary = _temporary(container, tag, geom, level, source)
            if view is not None:
                view.set('showlabels', 'on' if labels else 'off')
            picture = temporary.image()
            known = {str(p) for p in picture.properties()}
            for key, value in settings.items():
                if key in known:
                    picture.set(key, value)
            try:
                picture.export()
            except Exception as error:
                raise OSError(f'COMSOL could not write the picture "{path}": '
                              f'{_comsol.reason(error)}') from error
        finally:
            try:
                if tag in [str(t) for t in container.tags()]:
                    container.remove(tag)
            finally:
                if view is not None and shown is not None:
                    view.set('showlabels', shown)
    return path


############################
# Shared with result plots #
############################

def picture_path(filename) -> Path:
    """Returns the absolute path of a picture, checking its file type."""
    path = Path(os.fspath(filename)).expanduser().absolute()
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        *others, last = FORMATS
        raise ValueError(f'Save the picture as {", ".join(others)} or '
                         f'{last}, not "{path.name}".')
    return path.with_suffix(suffix)  # COMSOL writes x.PNG as x.png


def picture_size(size) -> tuple[int, int]:
    """Returns `(width, height)` in pixels, checked."""
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(isinstance(v, numbers.Integral)
                       and not isinstance(v, (bool, numpy.bool_))
                       and int(v) > 0 for v in size)):
        raise ValueError(f'size must be (width, height) in pixels, not '
                         f'{size!r}.')
    return int(size[0]), int(size[1])


def export(create, model, group, path: Path, size: tuple[int, int],
           sdim: int):
    """
    Writes the plot group `group` to `path` with a temporary image export
    made with `create` (see `_comsol.scratch`). The export draws the
    group: running it before (`group.run()`) would also open a window of
    the COMSOL server, one per picture on Windows. Title and colour legend
    are on. Every setting must exist, so that a renamed property raises
    instead of changing the picture silently.
    """
    imagetype, name_key = FORMATS[path.suffix]
    width, height = size
    # `saveprefs` first, then the source; `size` in COMMON loads a preset
    # that the width and height override.
    settings = {'saveprefs': 'off', 'sourceobject': str(group.tag()),
                **COMMON, 'width': str(width), 'height': str(height),
                **DECORATION[sdim], f'title{sdim}d': 'on',
                f'legend{sdim}d': 'on', 'zoomextents': 'on',
                'imagetype': imagetype, name_key: str(path)}
    picture = create(model.result().export(), 'Image')
    for key, value in settings.items():
        _comsol.set_property(picture, key, value)
    try:
        picture.run()
    except Exception as error:
        raise OSError(f'COMSOL could not write the picture "{path}": '
                      f'{_comsol.reason(error)}') from error


def drawing_failed(error: Exception) -> bool:
    """
    Tells whether an export failed in drawing the plot (e.g. an undefined
    variable, reported with the expression and the plot) rather than in
    writing the file.
    """
    text = _comsol.reason(error)
    return 'Plot:' in text or 'Expression:' in text


def drawn_selection(geom: Node, selection: Node) -> tuple[int, str, list]:
    """Returns the level, the tag and the entities of a selection to draw."""
    if (isinstance(selection, Node) and len(selection.path) == 3
            and selection.path[0] == 'geometries'):
        java = _comsol.java_of(selection)
        if _comsol.is_selection_feature(java):
            if str(java.getString('entitydim')) == '-1':
                raise TypeError(f'"{selection}" selects objects; a picture '
                                'shows a selection of entities, e.g. with '
                                "the same box at the 'domain' level.")
            raise TypeError(f'"{selection}" is the feature in the geometry '
                            'sequence; pass the selection it makes, '
                            f"model/'selections'/'{selection.name()}'.")
    java = _comsol.check_selection(geom, selection)
    level = _comsol.selection_dims(java)
    if len(level) != 1:
        raise ValueError(f'Selection "{selection}" has no single level.')
    entities = [int(e) for e in java.entities()]
    if not entities:
        raise ValueError(f'Selection "{selection}" is empty; nothing to '
                         'draw.')
    return level[0], str(java.tag()), entities


def geometry_view(geom: Node):
    """Returns the view that shows `geom`, or `None`."""
    views = _comsol.component_of(geom).view()
    found = []
    for tag in views.tags():
        view = views.get(tag)
        try:
            shows = str(view.geom().tag()) == geom.tag()
        except Exception:
            shows = False
        if shows:
            found.append(view)
    return found[0] if found else None


###########
# Helpers #
###########

def _temporary(container, tag: str, geom: Node, level: int,
               source: str | None):
    """
    Creates the selection to draw: a copy of `source`, or an empty one that
    shows the plain geometry.
    """
    if source is None:
        java = container.create(tag, 'Explicit')
        java.geom(geom.tag(), level)
    else:
        java = container.create(tag, 'Union')
        _comsol.set_property(java, 'entitydim', level)
        _comsol.set_property(java, 'input', [source])
    return java


def _mesh_picture(geom: Node, path: Path, selection, mesh,
                  size: tuple[int, int]) -> Path:
    """Draws the mesh of `geom`, or of a selection, coloured by quality."""
    sdim = _comsol.sdim(geom)
    if sdim < 2:
        raise ValueError(f'Mesh pictures are for 2D and 3D geometries; '
                         f'"{geom}" is 1D.')
    sequence = _mesh.find_mesh(geom, mesh)
    _comsol.check_built(geom)
    level, entities = sdim, None
    if selection is not None:
        level, _, entities = drawn_selection(geom, selection)
        if level not in ((3, 2) if sdim == 3 else (2,)):
            kind = _comsol.entity_level_name(level, sdim) or level
            raise ValueError('A mesh picture takes a boundary or domain '
                             'selection in 3D, a domain selection in 2D; '
                             f'"{selection}" selects {kind} entities.')
    model = geom.model.java
    view = geometry_view(geom)
    with _comsol.scratch(model) as create:
        data = create(model.result().dataset(), 'Mesh')
        _comsol.set_property(data, 'mesh', str(sequence.tag()))
        if entities is not None:
            data.selection().geom(geom.tag(), level)
            data.selection().set(entities)
        group = create(model.result(), f'PlotGroup{sdim}D')
        _comsol.set_property(group, 'data', str(data.tag()))
        _comsol.set_property(group, 'view',
                             'auto' if view is None else str(view.tag()))
        feature = group.create('mesh1', 'Mesh')
        # volume elements of domains in 3D, surface elements of boundaries
        _comsol.set_property(feature, 'meshdomain',
                             'volume' if sdim == 3 and level == 3
                             else 'surface')
        try:
            export(create, model, group, path, size, sdim)
        except OSError as failure:
            error = failure.__cause__
            if not isinstance(error, Exception) or not drawing_failed(error):
                raise
            raise RuntimeError(f'COMSOL could not draw the mesh: '
                               f'{_comsol.reason(error)}') from error
    return path
