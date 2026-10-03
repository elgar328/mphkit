"""
Materials from COMSOL's material libraries. Public as `mk.materials` and
`mk.material`.

The libraries are the `.mph` files in the installation's `data` folder
(one per module, `comsol_basic_material_lib.mph` for the basic one). The
list of their materials is read from the files themselves, a zip archive
with the model as XML (`dmodel.xml`, checked with COMSOL 6.4 against what
COMSOL reads from them). Inserting a material uses COMSOL's own
`insert()` of the component's material list, which copies the whole
material: property groups, functions such as a density `rho(T)`, and
model inputs.
"""
from __future__ import annotations

import functools
import zipfile
from difflib import get_close_matches
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

from mph.node import Node

from . import _catalog, _comsol, _measure

BASIC = 'basic_material'


class Item(NamedTuple):
    """A material in a library file."""
    tag: str
    name: str
    groups: tuple[str, ...]
    properties: tuple[str, ...]


#############
# Libraries #
#############

def _folder() -> Path:
    """Returns the folder of the running COMSOL's material libraries."""
    import jpype  # type: ignore[import-untyped]
    if not jpype.isJVMStarted():
        raise RuntimeError('COMSOL is not running; call mph.start() first.')
    return Path(_catalog.comsol_root())/'data'


def _libraries() -> dict[str, Path]:
    """
    Returns the installed libraries by name, the basic one first: the file
    name without `comsol_` in front and `_lib` at the end.
    """
    folder = _folder()
    found = {}
    for path in sorted(folder.glob('*.mph')):
        name = path.stem.removeprefix('comsol_').removesuffix('_lib')
        found[name] = path
    if not found:
        raise RuntimeError(f'Found no material libraries (*.mph) in '
                           f'{folder}.')
    order = sorted(found, key=lambda name: (name != BASIC, name))
    return {name: found[name] for name in order}


def _chosen(library, search) -> dict[str, Path]:
    """
    Returns the libraries `library` names: one name, a list of names,
    `'all'`, or `None` for the basic one (all of them with a `search`).
    """
    installed = _libraries()
    if library is None:
        library = 'all' if _catalog.search_words(search) else BASIC
    if library == 'all':
        return installed
    names = [library] if isinstance(library, str) else library
    if not isinstance(names, (list, tuple)) or not names or \
            not all(isinstance(name, str) for name in names):
        raise TypeError(f"library must be a name, a list of names or 'all', "
                        f'not {library!r}.')
    unknown = [name for name in names if name not in installed]
    if unknown:
        raise ValueError(f'No material library {unknown[0]!r}; installed: '
                         f'{", ".join(installed)}.')
    return {name: path for name, path in installed.items() if name in names}


def _read(path: Path) -> tuple[Item, ...]:
    """Returns the materials in a library file, read once per version."""
    return _parse(str(path), path.stat().st_mtime)


@functools.cache
def _parse(path: str, mtime: float) -> tuple[Item, ...]:
    def fail(reason: str) -> RuntimeError:
        return RuntimeError(
            f'Cannot read the material library {path}: {reason}. COMSOL '
            'may have changed its format; client.load() opens the file to '
            'look inside.')
    try:
        with zipfile.ZipFile(path) as archive:
            top = ElementTree.fromstring(archive.read('dmodel.xml'))
    except (OSError, KeyError, zipfile.BadZipFile,
            ElementTree.ParseError) as error:
        raise fail(str(error)) from None
    lists = [element for element in top if element.tag == 'MaterialList'
             and element.get('tag') == 'material']
    if len(lists) != 1:
        raise fail('no list of materials')
    items = []
    for element in lists[0]:
        if element.tag != 'Material':
            continue
        tag, label = element.get('tag'), element.get('name')
        if not tag or not label:
            raise fail('a material without tag or name')
        groups = element.findall('MaterialModelList/MaterialModel')
        basic = [param.get('param') or ''
                 for group in groups if group.get('tag') == 'def'
                 for param in group.findall('param')
                 if _filled(param.get('value'))]
        items.append(Item(tag, _clean(label),
                          tuple(group.get('name') or group.get('tag') or ''
                                for group in groups),
                          tuple(name for name in basic
                                if name and ':' not in name)))
    if not items:
        raise fail('no materials')
    return tuple(items)


def _filled(value: str | None) -> bool:
    """
    Tells whether the first entry of a stored value is not empty. Values
    look like "9|1,'238[W/(m*K)]'|1,'0'|..." (a count, then the entries);
    a tensor with an empty diagonal starts with an empty entry.
    """
    parts = (value or '').split('|')
    return len(parts) > 1 and parts[1].split(',', 1)[-1].strip("'") != ''


def _clean(label: str) -> str:
    """Collapses the spaces in a label; some end in one or two."""
    return ' '.join(label.split())


def _key(name: str) -> str:
    return _clean(name).casefold()


###########
# Listing #
###########

def materials(*, search: str | None = None, library: str | list[str] |
              None = None) -> list[dict]:
    """
    Returns the materials in COMSOL's material libraries, the names that
    `mk.material` inserts, e.g.

    ```python
    mk.materials(search='structural steel')
    # [{'name': 'Structural steel', 'library': 'basic_material',
    #   'groups': ['Basic', "Young's modulus and Poisson's ratio", ...],
    #   'properties': ['relpermeability', 'heatcapacity',
    #                  'thermalconductivity', ..., 'lossfactor']}]
    steel = mk.material(geom, 'Structural steel')
    ```

    Each item has the `name` that `mk.material` takes, the `library` it
    is in, the labels of its property `groups` (e.g. "Young's modulus and
    Poisson's ratio" for solid mechanics, 'Refractive index' for optics)
    and the `properties` its 'Basic' group defines (e.g. 'density',
    'thermalconductivity', 'heatcapacity'), to check that a material
    has what a physics needs before inserting it. `mk.properties` shows
    the values once the material is in the model.

    `library` is a library name, a list of them or `'all'`. Without it
    the list holds the basic library (`'basic_material'`, 35 materials
    such as air, aluminum, copper, structural steel and water), and a
    `search` looks in all libraries. `search` keeps the materials whose
    name or group labels hold every word of it (any case), those that
    match by name first; the libraries hold 1307 materials in COMSOL 6.4
    and `search='steel'` finds 228 (the basic library's first), so use
    specific words. The libraries
    are named after their files in COMSOL's `data` folder, e.g. `'acdc'`,
    `'battery'`, `'bioheat'`, `'building'`, `'composite'`, `'corr'`,
    `'edis'`, `'eqdis'`, `'fce'`, `'fluids'`, `'magnetic'`, `'mems'`,
    `'piezo'`, `'piezor'`, `'rf'`, `'semi'` and `'thermoelectric'` in
    COMSOL 6.4; the list is read from the installed files, which may
    include libraries of modules that are not licensed (licenses are not
    checked). Sorted by library (the basic one first, then by library
    name, also when `library` lists them in another order), then by
    material name.

    These are the `.mph` libraries in COMSOL's `data` folder; the optical
    library (`comsol_optical_lib.xml`, refractive indices) and the
    separate Material Library product are not read.

    Reads the library files of the running COMSOL (call `mph.start()`
    first) and leaves nothing in any model.
    """
    words = _catalog.search_words(search)
    by_name: list[dict] = []
    by_group: list[dict] = []
    for name, path in _chosen(library, search).items():
        for item in sorted(_read(path),
                           key=lambda item: _catalog.name_key(item.name)):
            if not _catalog.words_found(words, item.name, *item.groups):
                continue
            found = by_name if _catalog.words_found(words, item.name) \
                else by_group
            found.append({'name': item.name, 'library': name,
                          'groups': list(item.groups),
                          'properties': list(item.properties)})
    return by_name + by_group


#############
# Inserting #
#############

def material(geom: Node, material: str, /, selection=None, *,
             library: str | list[str] | None = None,
             name: str | None = None) -> Node:
    """
    Inserts a material from COMSOL's material libraries into the
    component of the geometry (`mk.materials` lists them; not the optical
    library or the Material Library product), e.g.

    ```python
    steel = mk.material(geom, 'Structural steel')      # all domains
    water = mk.material(geom, 'Water, liquid', channel)
    ```

    `material` is a name as `mk.materials()` lists it (any case). It is
    looked up in the basic library unless `library` names another one
    (see `mk.materials`); a name found only elsewhere raises an error
    that names the library to pass, other unknown names suggest close
    ones. The whole material is copied, with its property groups and
    functions: water's density is `rho(T)`. Such properties follow the
    temperature a physics feature takes as model input
    (`minput_temperature`): the computed one in heat transfer, otherwise
    293.15 K by default. A solid in heat transfer takes its density at a
    fixed temperature (`ht.Trho`, 293.15 K by default), its heat capacity
    and conductivity at the computed temperature.

    `selection` is a domain selection node or domain numbers of the built
    geometry. Each domain takes the material lowest in the list (the one
    added last) among those that select it, so add the background
    material first, without a selection: it takes all domains, also after
    the geometry changes. Later materials need a selection, and an error
    is raised without one while another material holds domains. To give
    domains back to an earlier material, change or remove the later one.

    The new material is tagged `mat1`, `mat2`, ... (library tags such as
    `water,_liquid` cannot be used in expressions: `mat1.def.rho(300)`),
    and labelled with its name or `name`. A Java export of the model
    records the path of the library file in the `insert()` call; after
    `model.reset()` it lists the properties instead. Only domains are
    supported, not boundaries (shells). Returns the node under
    `model/'materials'`.

    A material with values of one's own, not from a library, is plain
    MPh, with `mk.set` for the values; it takes all domains:

    ```python
    steel = (model/'materials').create('Common', name='steel')
    mk.set(steel/'Basic', thermalconductivity='45', density='7850',
           heatcapacity='475')
    ```

    MPh creates it in the model's last component.
    """
    if not isinstance(geom, Node) or len(geom.path) != 2 \
            or geom.path[0] != 'geometries':
        raise TypeError(f'mk.material takes a geometry, not {geom!r}.')
    if not isinstance(material, str):
        raise TypeError(f'material must be a name, not {material!r}.')
    if name is not None and not isinstance(name, str):
        raise TypeError(f'name must be a string, not {name!r}.')
    path, item = _find(material, library)
    component = _comsol.component_of(geom)
    domains = _domains(geom, component, selection)
    everything = geom.model.java.material()
    label = _comsol.pick_label(name, item.name, _comsol.labels(everything))
    # Chosen before inserting: COMSOL may give the new material this tag
    # when its library tag is taken (library tags include mat1, ...).
    final = str(everything.uniquetag('mat'))
    tag = _insert(component, everything, path, item)
    try:
        java = component.material(tag)
        if tag != final:
            java.tag(final)
            tag = final
        java.label(label)
        node = _comsol.child_node(geom.model/'materials', label)
        _comsol.check_tag(node, tag)
        if selection is None:
            java.selection().all()
        elif isinstance(selection, Node):
            node.select(selection)
        else:
            java.selection().set(domains)
    except Exception:
        component.material().remove(tag)
        raise
    return node


def _insert(component, everything, path: Path, item: Item) -> str:
    """Inserts a library material and returns its tag in the model."""
    before = {str(tag) for tag in everything.tags()}
    # An empty list of passwords: [''] warns about a protected file.
    component.material().insert(str(path), [item.tag], [])
    new = [str(tag) for tag in everything.tags() if str(tag) not in before]
    if len(new) != 1:
        for tag in new:
            component.material().remove(tag)
        raise RuntimeError(f'Inserting "{item.name}" from {path} did not '
                           'give one new material.')
    return new[0]


def _find(material: str, library) -> tuple[Path, Item]:
    """Returns the library file and entry of a material name."""
    installed = _libraries()
    if library is None:
        chosen = {BASIC: installed[BASIC]} if BASIC in installed else {}
    else:
        chosen = _chosen(library, None)
    key = _key(material)
    hits = _hits(chosen, key)
    if len(hits) == 1:
        return hits[0][1:]
    names = list(dict.fromkeys(hit[0] for hit in hits))
    if hits:
        raise ValueError(f'"{material}" is in several libraries '
                         f'({", ".join(names)}); pass one of them as '
                         'library=.')
    others = _hits({n: p for n, p in installed.items() if n not in chosen},
                   key)
    if others:
        names = list(dict.fromkeys(hit[0] for hit in others))
        raise ValueError(f'No material "{material}" in {_scope(chosen)}; it '
                         f'is in {_scope(names)}: pass '
                         f'library={names[0]!r}.')
    message = (f'No material "{material}" in the libraries mk.materials '
               'reads (not the optical library or the Material Library '
               'product).')
    close = _close(installed, key)
    if close:
        message += f' Did you mean {", ".join(close)}?'
    raise ValueError(message + " mk.materials(search='...') lists them.")


def _scope(names) -> str:
    """Names libraries in a message, e.g. "the 'acdc' library"."""
    names = list(names)
    if not names:
        return 'no library'
    listed = ' and '.join(repr(name) for name in names)
    return f'the {listed} librar{"y" if len(names) == 1 else "ies"}'


def _hits(libraries: dict[str, Path], key: str) -> list[tuple]:
    return [(name, path, item) for name, path in libraries.items()
            for item in _read(path) if _key(item.name) == key]


def _close(libraries: dict[str, Path], key: str) -> list[str]:
    """Returns names close to `key`, with the library they are in."""
    where: dict[str, tuple[str, str]] = {}
    for name, path in libraries.items():
        for item in _read(path):
            where.setdefault(_key(item.name), (item.name, name))
    return [f'"{where[found][0]}" ({where[found][1]})'
            for found in get_close_matches(key, list(where), n=3, cutoff=0.6)]


def _domains(geom: Node, component, selection) -> list[int] | None:
    """
    Returns the domain numbers of `selection`, or `None` for all domains
    when no material holds any yet.
    """
    if selection is None:
        _comsol.check_built(geom)
        if not _comsol.entity_count(geom, _comsol.sdim(geom)):
            raise ValueError(f'Geometry "{geom}" has no domains.')
        held = held_domains(component, _comsol.sdim(geom))
        if held:
            taken = {n for _, numbers in held for n in numbers}
            free = [n for n in range(1, _comsol.entity_count(
                geom, _comsol.sdim(geom)) + 1) if n not in taken]
            listed = '; '.join(f'"{label}" {numbers}'
                               for label, numbers in held)
            advice = (f'Domains {free} have no material: pass them as the '
                      'selection' if free else 'Pass a selection')
            raise ValueError(
                f'Materials already hold domains: {listed}. {advice}, or '
                'remove those materials and add the background material '
                'first: a material added later takes every domain it '
                'selects, also from earlier ones.')
        return None
    found = _measure.numbers_of(geom, 'domain', selection)
    if not found:
        raise ValueError(f'Selection "{selection}" is empty.'
                         if isinstance(selection, Node)
                         else 'The selection is empty.')
    return found


def held_domains(component, dim: int) -> list[tuple[str, list[int]]]:
    """
    Returns the labels and domains of the members of the component's
    material list that hold domains (level `dim`): active materials,
    switches and links, also links to global materials, not layered
    materials or materials on boundaries. A disabled material still lists
    its domains.
    """
    members = component.material()
    held = []
    for tag in members.tags():
        member = members.get(tag)
        try:
            if not member.isActive():
                continue
            chosen = member.selection()
            if _comsol.selection_dims(chosen) != [dim]:
                continue
            entities = [int(e) for e in chosen.entities()]
        except Exception:
            continue
        if entities:
            held.append((str(member.label()), entities))
    return held
