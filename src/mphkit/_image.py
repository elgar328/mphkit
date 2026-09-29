"""
Pictures of geometries and selections. Public as `mk.image`.

COMSOL keeps an image export per geometry and per selection, saves it with
the model and records every change in the model history. So `image()`
draws a temporary selection instead, removed afterwards, and switches the
history off meanwhile: the model is left as it was.
"""
from __future__ import annotations

import os
from pathlib import Path

from mph.node import Node

from . import _comsol

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
          zoom: str = 'all', size=(800, 600), labels: bool = False) -> Path:
    """
    Saves a picture of a geometry, or of a selection highlighted on it, and
    returns the file path: a way to look at a model without the COMSOL
    Desktop, also for an AI assistant that reads images.

    ```python
    mk.image(geom, 'geometry.png')
    mk.image(geom, 'wall.png', wall, labels=True)
    ```

    The view is the geometry's current one (isometric for a new 3D model,
    from above in 2D), with axes. A selection is drawn in blue, the rest
    of the geometry as lines. `zoom='selection'` zooms to the selection
    instead of showing the whole geometry; `size` is `(width, height)` in
    pixels; `labels=True` adds the numbers of the visible entities at the
    selection's level (hidden ones may lack one; `sel.entities` lists them
    all). The selection is one of entities, as physics uses
    them, not of `'object'` level. The file type follows the suffix,
    `.png` or `.jpg`; COMSOL writes the file on the machine it runs on (by
    default this one) and creates missing folders. The geometry must be
    built. The model is left as it was.
    """
    _comsol.check_not_workplane(geom, 'image')
    if isinstance(filename, Node):
        raise TypeError('The file name comes second: '
                        'mk.image(geom, filename, selection).')
    path = Path(os.fspath(filename)).expanduser().absolute()
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f'Save the picture as .png or .jpg, not '
                         f'"{path.name}".')
    path = path.with_suffix(suffix)  # COMSOL writes x.PNG as x.png
    imagetype, name_key = FORMATS[suffix]
    if zoom not in ('all', 'selection'):
        raise ValueError(f"zoom must be 'all' or 'selection', not {zoom!r}.")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool)
                       and v > 0 for v in size)):
        raise ValueError(f'size must be (width, height) in pixels, not '
                         f'{size!r}.')
    width, height = size
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
        level, source = _source(geom, selection)
    view = _view(geom)
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
    history = model.hist()
    # disable() and enable() nest: a history the user switched off stays
    # off afterwards.
    history.disable()
    try:
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
                              f'{_reason(error)}') from error
        finally:
            try:
                if tag in [str(t) for t in container.tags()]:
                    container.remove(tag)
            finally:
                if view is not None and shown is not None:
                    view.set('showlabels', shown)
    finally:
        history.enable()
    return path


def _source(geom: Node, selection: Node) -> tuple[int, str]:
    """Returns the level and the tag of a selection to draw."""
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
    level = [int(d) for d in java.dimension()]
    if len(level) != 1:
        raise ValueError(f'Selection "{selection}" has no single level.')
    if not len(java.entities()):
        raise ValueError(f'Selection "{selection}" is empty; nothing to '
                         'highlight.')
    return level[0], str(java.tag())


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


def _view(geom: Node):
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


def _reason(error: Exception) -> str:
    """Returns the reason in a COMSOL error, without its boilerplate."""
    useful: list[str] = []
    for line in str(error).splitlines():
        line = line.strip().lstrip('- ')
        if (line and not line.startswith(('Exception', 'Messages'))
                and 'com.comsol.' not in line and line not in useful):
            useful.append(line)
    return ' '.join(useful) if useful else str(error)
