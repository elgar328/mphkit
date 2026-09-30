"""
Checks the materials from COMSOL's libraries: the list, read from the
library files without COMSOL, and inserting them into a model.
"""
import os
import re
import zipfile
from pathlib import Path

import pytest

import mphkit as mk
from conftest import java_export
from mphkit import _materials


@pytest.fixture
def offline(monkeypatch):
    """Reads the libraries of the installed COMSOL without starting it."""
    discovery = pytest.importorskip('mph.discovery')
    try:
        root = discovery.backend()['root']
    except Exception as error:
        pytest.skip(f'COMSOL not found: {error}')
    monkeypatch.setattr(_materials, '_folder', lambda: root/'data')
    return root


def blocks(model, count=2, dim=3):
    """A geometry of blocks (squares in 2D) side by side, built."""
    geom = mk.geometry(model, dim)
    for n in range(count):
        if dim == 3:
            mk.block(geom, (1, 1, 1), (n, 0, 0))
        else:
            mk.square(geom, 1, (n, 0))
    model.build(geom)
    return geom


def domains(node):
    return [int(d) for d in node.java.selection().entities()]


def tags(model):
    return [str(t) for t in model.java.material().tags()]


###########
# Listing #
###########

def test_basic_library(offline):
    found = mk.materials()
    assert len(found) == 35
    assert {item['library'] for item in found} == {'basic_material'}
    aluminum = next(item for item in found if item['name'] == 'Aluminum')
    assert list(aluminum) == ['name', 'library', 'groups', 'properties']
    assert {'density', 'thermalconductivity', 'heatcapacity'} <= \
        set(aluminum['properties'])
    steel = next(item for item in found if item['name'] == 'Structural steel')
    assert "Young's modulus and Poisson's ratio" in steel['groups']
    assert steel['groups'][0] == 'Basic'
    names = [item['name'] for item in found]
    assert names == sorted(names, key=str.casefold)


def test_search(offline):
    # a search looks in all libraries, the basic one first
    found = mk.materials(search='steel')
    basic = [item['name'] for item in found
             if item['library'] == 'basic_material']
    assert basic == ['High-strength alloy steel', 'Steel AISI 4340',
                     'Structural steel']
    assert [item['name'] for item in found[:3]] == basic
    assert len(found) > 100
    graphite = mk.materials(search='graphite')
    assert ('Graphite', 'acdc') in [(i['name'], i['library'])
                                    for i in graphite]
    # matches by name first, then by group label only
    water = mk.materials(search='water')
    named = ['water' in item['name'].lower() for item in water]
    assert named == sorted(named, reverse=True) and not all(named)
    assert [i['library'] for i in water if 'water' not in i['name'].lower()
            ] == ['fce']*named.count(False)
    # several words, any case, also in group labels
    assert [item['name'] for item in
            mk.materials(search='STEEL structural')] == ['Structural steel']
    optical = mk.materials(search='refractive index', library='basic_material')
    assert 'Air' in [item['name'] for item in optical]
    assert mk.materials(search='steel', library='basic_material') == \
        found[:3]


def test_libraries(offline):
    assert len(mk.materials(library='acdc')) == 422
    both = mk.materials(library=['bioheat', 'acdc'])
    assert [item['library'] for item in both] == \
        ['acdc']*422 + ['bioheat']*13
    everything = mk.materials(library='all')
    assert len(everything) == 1307
    assert everything[0]['library'] == 'basic_material'
    with pytest.raises(ValueError, match="No material library 'ACDC'.*acdc"):
        mk.materials(library='ACDC')
    with pytest.raises(TypeError, match='library must be'):
        mk.materials(library=3)
    with pytest.raises(TypeError, match='search must be a string'):
        mk.materials(search=['steel'])
    # an empty search is no search
    assert mk.materials(search=' ') == mk.materials()


def test_library_labels(offline):
    # labels ending in spaces come without them
    composite = mk.materials(library='composite')
    names = [item['name'] for item in composite]
    assert 'Aluminum oxide ceramic fiber' in names
    assert all(name == ' '.join(name.split()) for name in names)
    # references and empty values are not properties
    for item in mk.materials(library='all'):
        assert not any(':' in name for name in item['properties'])
    aramid = next(item for item in composite
                  if item['name'] == 'K-119 aramid fiber')
    assert aramid['properties'] == ['density']


def test_library_format(offline, tmp_path, monkeypatch):
    # a file in another format raises instead of listing nothing
    path = tmp_path/'broken_lib.mph'
    for content, reason in ((None, 'File is not a zip file'),
                            ({'other.xml': '<a/>'}, 'dmodel.xml'),
                            ({'dmodel.xml': '<Model>'}, 'no element found'),
                            ({'dmodel.xml': '<Model/>'},
                             'no list of materials'),
                            ({'dmodel.xml': '<Model><MaterialList tag='
                              '"material"/></Model>'}, 'no materials'),
                            ({'dmodel.xml': '<Model><MaterialList tag='
                              '"material"><Material tag="m"/></MaterialList>'
                              '</Model>'}, 'without tag or name')):
        path.unlink(missing_ok=True)
        if content is None:
            path.write_text('not a zip')
        else:
            with zipfile.ZipFile(path, 'w') as archive:
                for name, text in content.items():
                    archive.writestr(name, text)
        with pytest.raises(RuntimeError, match=reason):
            _materials._read(path)
    # a changed file is read again
    for name, when in ((' A  b ', 1e9), ('C', 2e9)):
        path.unlink()
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('dmodel.xml', '<Model><MaterialList tag='
                             f'"material"><Material tag="m1" name="{name}"/>'
                             '</MaterialList></Model>')
        os.utime(path, (when, when))
        assert _materials._read(path) == \
            (_materials.Item('m1', ' '.join(name.split()), (), ()),)
    monkeypatch.setattr(_materials, '_folder', lambda: tmp_path/'nothing')
    with pytest.raises(RuntimeError, match='Found no material libraries'):
        mk.materials()


def test_needs_comsol(monkeypatch):
    jpype = pytest.importorskip('jpype')
    monkeypatch.setattr(jpype, 'isJVMStarted', lambda: False)
    with pytest.raises(RuntimeError, match=r'call mph.start\(\) first'):
        mk.materials()


def test_listing_leaves_models(client, model, tmp_path):
    blocks(model)
    before = client.models(), java_export(model, tmp_path/'model.java')
    assert mk.materials(search='water')
    assert (client.models(), java_export(model, tmp_path/'model.java')) == \
        before


#############
# Inserting #
#############

def test_insert(model, tmp_path):
    geom = blocks(model)
    aluminum = mk.material(geom, 'aluminum')
    assert str(aluminum) == 'materials/Aluminum'
    assert aluminum.tag() == 'mat1'
    assert domains(aluminum) == [1, 2]
    # a later material takes the domains it selects
    copper = mk.material(geom, 'Copper', [1])
    assert copper.tag() == 'mat2'
    assert domains(copper) == [1]
    assert domains(aluminum) == [2]
    # and one without a selection would take all of them
    with pytest.raises(ValueError, match=r'"Aluminum" \[2\]; "Copper" \[1\]. '
                                         'Pass a selection'):
        mk.material(geom, 'Iron')
    assert tags(model) == ['mat1', 'mat2']
    # the whole material is copied
    assert (aluminum/'Basic').property('density') == '2700[kg/m^3]'
    assert any('.insert("' in line for line in
               java_export(model, tmp_path/'model.java'))


def test_labels(model):
    geom = blocks(model)
    mk.material(geom, 'Aluminum')
    again = mk.material(geom, 'Aluminum', [2])
    assert str(again) == 'materials/Aluminum (2)'
    named = mk.material(geom, 'Copper', [1], name='core')
    assert str(named) == 'materials/core' and named.tag() == 'mat3'
    with pytest.raises(ValueError, match='"core" is already in use'):
        mk.material(geom, 'Iron', [1], name='core')
    # spaces at the end of a library label are dropped
    fiber = mk.material(geom, 'aluminum oxide ceramic fiber', [2],
                        library='composite')
    assert str(fiber) == 'materials/Aluminum oxide ceramic fiber'
    # slashes in labels
    magnet = mk.material(geom, 'BMN-35H/S', [1], library='acdc')
    assert str(magnet.java.label()) == 'BMN-35H/S' and magnet.exists()
    assert tags(model) == ['mat1', 'mat2', 'mat3', 'mat4', 'mat5']


def test_selections(model):
    geom = mk.geometry(model, 3)
    first = mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    mk.block(geom, (1, 1, 1), (2, 0, 0))
    model.build(geom)
    mk.material(geom, 'Air')
    box = mk.sel.box(geom, 'domain', x=(0.9, 2.1))
    copper = mk.material(geom, 'Copper', box)
    assert copper.selection() == box
    assert domains(copper) == [2]
    result = mk.sel.result(geom, first, 'domain')
    iron = mk.material(geom, 'Iron', result)
    assert domains(iron) == [1]
    assert mk.sel.entities(geom, result) == [1]
    with pytest.raises(ValueError, match='is empty'):
        mk.material(geom, 'Iron', mk.sel.box(geom, 'domain', x=(5, 6)))
    with pytest.raises(ValueError, match='The selection is empty'):
        mk.material(geom, 'Iron', [])
    with pytest.raises(ValueError, match='not a domain selection'):
        mk.material(geom, 'Iron', mk.sel.box(geom, 'boundary', x=0))
    with pytest.raises(ValueError, match=r'No domain \[4\]'):
        mk.material(geom, 'Iron', [4])
    other = blocks(model, 1)
    with pytest.raises(ValueError, match='does not belong'):
        mk.material(other, 'Iron', box)
    assert len(tags(model)) == 3


def test_selection_rules(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    with pytest.raises(RuntimeError, match='not built'):
        mk.material(geom, 'Iron')
    model.build(geom)
    with pytest.raises(TypeError, match='takes a geometry'):
        mk.material(mk.workplane(geom), 'Iron')
    with pytest.raises(TypeError, match='takes a geometry'):
        mk.material(model/'materials', 'Iron')
    model.build(geom)
    # a material on boundaries (a shell) holds no domains
    component = mk.component_of(geom).java
    component.material().create('shell', 'Common')
    shell = component.material('shell')
    shell.selection().geom(geom.tag(), 2)
    shell.selection().set([1])
    iron = mk.material(geom, 'Iron')
    assert domains(iron) == [1]
    # the error names the domains without a material
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    mk.block(geom, (1, 1, 1), (2, 0, 0))
    model.build(geom)
    iron.java.selection().set([2])
    with pytest.raises(ValueError, match=r'Domains \[1, 3\] have no '
                                         'material: pass them'):
        mk.material(geom, 'Copper')
    iron.java.selection().all()
    # a disabled material holds no domains, though COMSOL lists them
    iron.java.active(False)
    assert domains(iron) == [1, 2, 3]
    assert domains(mk.material(geom, 'Copper')) == [1, 2, 3]
    # nor can a geometry without domains
    lines = mk.geometry(model, 2)
    mk.line_segment(lines, (0, 0), (1, 1))
    model.build(lines)
    with pytest.raises(ValueError, match='has no domains'):
        mk.material(lines, 'Iron')


def test_unknown_names(model):
    geom = blocks(model, 1)
    with pytest.raises(ValueError, match=r'Did you mean "Aluminum" '
                                         r'\(basic_material\)'):
        mk.material(geom, 'Aluminium')
    with pytest.raises(ValueError, match="in the 'acdc' library: pass "
                                         "library='acdc'"):
        mk.material(geom, 'BMN-35')
    # not the basic library's Granite
    with pytest.raises(ValueError, match="'acdc' library") as error:
        mk.material(geom, 'Graphite')
    assert 'Granite' not in str(error.value)
    with pytest.raises(ValueError, match="'acdc' and 'bioheat' libraries"):
        mk.material(geom, 'Fat')
    with pytest.raises(ValueError, match=r'several libraries \(basic_material'
                                         r', acdc'):
        mk.material(geom, 'Copper', library='all')
    with pytest.raises(ValueError, match="No material \"Copper\" in the "
                                         "'bioheat' library; it is in"):
        mk.material(geom, 'Copper', library='bioheat')
    with pytest.raises(ValueError, match='No material library'):
        mk.material(geom, 'Copper', library='nope')
    with pytest.raises(ValueError, match="No material \"Water, liquid\" in "
                                         "the 'acdc' and 'bioheat' "
                                         "libraries; it is in the "
                                         "'basic_material'"):
        mk.material(geom, 'Water, liquid', library=['bioheat', 'acdc'])
    with pytest.raises(TypeError, match='material must be a name'):
        mk.material(geom, 3)
    assert tags(model) == []
    assert mk.material(geom, 'Fat', library='bioheat').exists()


def test_tag_clash(model):
    # library tags may be mat1, which COMSOL renames on insert
    corr = _materials._libraries()['corr']
    item = next((i for i in _materials._read(corr) if i.tag == 'mat1'), None)
    if item is None:
        pytest.skip('no material tagged mat1 in the corrosion library')
    geom = blocks(model)
    component = mk.component_of(geom).java
    component.material().create('mat1', 'Common')
    component.material('mat1').label('mine')
    model.java.material().create('mat2', 'Common', '')     # a global one
    model.java.material('mat2').label('global')
    node = mk.material(geom, item.name, [1], library='corr')
    assert node.tag() == 'mat3'
    assert str(component.material('mat1').label()) == 'mine'
    assert tags(model) == ['mat1', 'mat2', 'mat3']


def test_components(model):
    # tags and labels are unique across components
    first = blocks(model, 1)
    second = blocks(model, 1)
    a = mk.material(first, 'Aluminum')
    b = mk.material(second, 'Aluminum')
    assert (a.tag(), b.tag()) == ('mat1', 'mat2')
    assert str(b) == 'materials/Aluminum (2)'
    assert domains(a) == domains(b) == [1]


def test_cleanup(model, monkeypatch):
    geom = blocks(model)
    mk.material(geom, 'Aluminum')

    def fail(node, tag):
        raise RuntimeError('check failed')
    monkeypatch.setattr(mk._comsol, 'check_tag', fail)
    with pytest.raises(RuntimeError, match='check failed'):
        mk.material(geom, 'Copper', [1])
    assert tags(model) == ['mat1']


def test_axisymmetric(model):
    geom = mk.geometry(model, 2)
    geom.java.axisymmetric(True)
    mk.rectangle(geom, (1, 2))
    model.build(geom)
    water = mk.material(geom, 'Water, liquid')
    assert domains(water) == [1]


def test_solved(model):
    # the copied properties reach the physics, per domain
    geom = blocks(model, 3)
    mk.material(geom, 'Aluminum')
    mk.material(geom, 'Copper', [2])
    water = mk.material(geom, 'Water, liquid', [3])
    assert 'rho' in [str(f) for f in
                     water.java.propertyGroup('def').func().tags()]
    heat = (model/'physics').create('HeatTransfer', geom)
    boundary = heat.create('TemperatureBoundary', 2)
    boundary.select(mk.sel.box(geom, 'boundary', x=0))
    boundary.property('T0', '300[K]')
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    density = [mk.average(geom, 'domain', 'ht.rho', n, unit='kg/m^3')
               for n in (1, 2, 3)]
    assert density[:2] == pytest.approx([2700, 8960])
    # a solid in heat transfer takes the density at 293.15 K, the heat
    # capacity at the computed temperature, here 300 K
    tag = water.tag()
    assert density[2] == pytest.approx(998.2, abs=0.05)
    assert mk.average(geom, 'domain', f'{tag}.def.rho(293.15[K])', 3,
                      unit='kg/m^3') == pytest.approx(density[2])
    assert mk.average(geom, 'domain', 'ht.Cp', 3) == pytest.approx(
        mk.average(geom, 'domain', f'{tag}.def.Cp(300[K])', 3))
    assert mk.average(geom, 'domain', f'{tag}.def.rho(300[K])', 3,
                      unit='kg/m^3') == pytest.approx(996.5, abs=0.05)


##############
# Documented #
##############

def documented(source):
    """The material lines of the README or of help(mphkit)."""
    if source == 'README.md':
        readme = (Path(__file__).parents[1]/'README.md').read_text()
        return re.search(r"Materials from COMSOL's libraries.*?```python\n"
                         r'(.*?)```', readme, re.S).group(1).splitlines()
    block = re.search(r"Materials from COMSOL's libraries.*?\n\n(.*?)\n\n",
                      mk.__doc__, re.S).group(1)
    return [line[4:] for line in block.splitlines()]


@pytest.mark.parametrize('source', ['README.md', 'mphkit'])
def test_documented(model, source):
    lines = documented(source)
    assert len(lines) == 3, 'update the checks below with the docs'
    geom = blocks(model)
    namespace = {'mk': mk, 'geom': geom,
                 'channel': mk.sel.box(geom, 'domain', x=(0.9, 2.1))}
    found = eval(lines[0].split('#')[0], namespace)
    assert [item['name'] for item in found] == ['Structural steel']
    for line in lines[1:]:
        exec(line.split('#')[0], namespace)
    assert domains(namespace['steel']) == [1]
    assert domains(namespace['water']) == [2]
