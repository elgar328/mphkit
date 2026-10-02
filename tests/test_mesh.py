"""
Checks mk.mesh_quality against COMSOL's mesh statistics, on the example's
model and on meshes with boundary layers, identity pairs, quadrilaterals
and failed features.
"""
import json
import re
from pathlib import Path

import numpy
import pytest

import mphkit as mk
from conftest import java_export, read
from mphkit import _comsol
from test_example import plate_with_holes

VOLUMES = ('tet', 'pyr', 'prism', 'hex')
FACES = ('tri', 'quad')
MEASURES = ('skewness', 'maxangle', 'volcircum', 'vollength', 'condition',
            'growth')


def statistics(model, mesh, kinds, measure='skewness'):
    """
    COMSOL's statistics of element types: counts, lowest quality and the
    10-bin distribution. The measure is set with the history off and put
    back, as it would otherwise be recorded in the model.
    """
    java = model.java.mesh(mesh)
    stat = java.stat()
    with _comsol.history_off(model.java):
        old = str(stat.getQualityMeasure())
        stat.setQualityMeasure(measure)
        present = [str(t) for t in stat.getTypes() if str(t) in kinds]
        found = {'elements': {t: int(stat.getNumElem(t)) for t in present},
                 'min': min(float(stat.getMinQuality(t)) for t in present),
                 'histogram': numpy.sum([list(stat.getQualityDistr(t, 10))
                                         for t in present], axis=0).tolist()}
        stat.setQualityMeasure(old)
    return found


def matches(quality, expected):
    assert quality['elements'] == expected['elements']
    assert quality['histogram'] == expected['histogram']
    # the statistics round to four digits
    assert quality['min'] == pytest.approx(expected['min'], abs=1e-4)


@pytest.fixture
def plate(client):
    """The example's model with two holes, meshed (in mm)."""
    model, geom, _ = plate_with_holes.build_model(client, 2)
    model.mesh()
    yield model, geom
    client.remove(model)


@pytest.fixture
def layers(model):
    """
    A block with a notch and a cylinder, meshed with tetrahedra and
    boundary layers on the notch: tets, prisms, pyramids, hexes, triangles
    and quadrilaterals.
    """
    geom = mk.geometry(model, 3)
    outer = mk.block(geom, (2, 2, 1))
    notch = mk.block(geom, (1, 1, 1), (1, 1, 0))
    mk.difference(geom, outer, [notch])
    mk.cylinder(geom, 0.3, 1, (0.5, 1.5, 0))
    model.build(geom)
    mesh = model.java.mesh().create('mesh1', 'geom1')
    mesh.create('ftet1', 'FreeTet')
    layer = mesh.create('bl1', 'BndLayer')
    layer.selection().geom('geom1', 3)
    layer.selection().all()
    faces = layer.create('blp1', 'BndLayerProp')
    faces.selection().set(mk.sel.find(geom, 'boundary', x=(0.99, 1.01))
                          + mk.sel.find(geom, 'boundary', y=(0.99, 1.01)))
    faces.set('blnlayers', '8')
    faces.set('blhmin', '0.05')
    mesh.run()
    return geom


###########
# Example #
###########

def test_plate(plate, tmp_path):
    model, geom = plate
    before = java_export(model, tmp_path/'state.java')
    quality = mk.mesh_quality(geom)
    assert (quality['mesh'], quality['measure'], quality['level']) == \
        ('mesh', 'skewness', 'domain')
    matches(quality, statistics(model, 'mesh1', VOLUMES))
    assert quality['mean'] == pytest.approx(
        statistics_mean(model, 'mesh1', 'tet'), abs=1e-4)
    worst = quality['worst']
    assert len(worst) == 5 and worst[0]['quality'] == quality['min']
    assert [w['quality'] for w in worst] == sorted(w['quality']
                                                   for w in worst)
    box = mk.bounding_box(geom, 'domain')
    for item in worst:
        assert item['entity'] == 1
        for value, axis in zip(item['position'], 'xyz'):
            assert box[axis][0] <= value <= box[axis][1]
    # sizes in mm, as COMSOL's h
    assert 0 < quality['size']['min'] < quality['size']['max'] < 20
    count = quality['elements']['tet']
    assert quality['by_entity'] == {1: {'elements': count,
                                        'min': quality['min'],
                                        'mean': pytest.approx(
                                            quality['mean'])}}
    assert quality['unmeshed'] == []
    assert all(m['severity'] != 'error' for m in quality['messages'])
    json.dumps(quality)
    assert java_export(model, tmp_path/'state.java') == before
    assert model.datasets() == []


def statistics_mean(model, mesh, kind):
    java = model.java.mesh(mesh)
    stat = java.stat()
    with _comsol.history_off(model.java):
        old = str(stat.getQualityMeasure())
        stat.setQualityMeasure('skewness')
        mean = float(stat.getMeanQuality(kind))
        stat.setQualityMeasure(old)
    return mean


@pytest.mark.parametrize('measure', MEASURES)
def test_measures(plate, measure):
    model, geom = plate
    quality = mk.mesh_quality(geom, measure=measure)
    matches(quality, statistics(model, 'mesh1', VOLUMES, measure))


def test_boundaries(plate):
    model, geom = plate
    quality = mk.mesh_quality(geom, 'boundary')
    assert quality['level'] == 'boundary'
    matches(quality, statistics(model, 'mesh1', FACES))
    bottom = mk.sel.box(geom, 'boundary', z=0)
    assert list(mk.mesh_quality(geom, 'boundary', bottom)['by_entity']) == \
        mk.sel.entities(geom, bottom)
    with pytest.raises(ValueError, match='not a domain selection'):
        mk.mesh_quality(geom, 'domain', bottom)
    few = mk.mesh_quality(geom, 'boundary', [1, 2])
    assert list(few['by_entity']) == [1, 2]
    assert sum(item['elements'] for item in few['by_entity'].values()) == \
        sum(few['elements'].values())


def test_size(plate):
    # the same as the lowest and highest h COMSOL evaluates
    model, geom = plate
    quality = mk.mesh_quality(geom)
    java = model.java
    found = []
    with _comsol.scratch(java) as create:
        data = create(java.result().dataset(), 'Mesh')
        data.set('mesh', 'mesh1')
        for kind in ('MinVolume', 'MaxVolume'):
            numerical = create(java.result().numerical(), kind)
            numerical.set('data', str(data.tag()))
            numerical.selection().geom('geom1', 3)
            numerical.selection().all()
            numerical.set('expr', ['h'])
            table = create(java.result().table(), 'Table')
            numerical.set('table', str(table.tag()))
            numerical.setResult()
            found.append(float(table.getReal()[0][0]))
    assert (quality['size']['min'], quality['size']['max']) == \
        pytest.approx(tuple(found))


################
# Other meshes #
################

def test_boundary_layers(model, layers):
    geom = layers
    quality = mk.mesh_quality(geom)
    # pyramids are few (5): the mesher may make none in another version
    assert {'tet', 'prism', 'hex'} <= set(quality['elements'])
    matches(quality, statistics(model, 'mesh1', VOLUMES))
    assert sorted(quality['by_entity']) == [1, 2]
    assert sum(item['elements'] for item in quality['by_entity'].values()) \
        == sum(quality['elements'].values())
    matches(mk.mesh_quality(geom, 'boundary'),
            statistics(model, 'mesh1', FACES))
    one = mk.mesh_quality(geom, 'domain', 2)
    assert list(one['by_entity']) == [2]
    assert one['by_entity'][2] == quality['by_entity'][2]
    # COMSOL's information on the boundary layers (thickness reduced)
    infos = [m for m in quality['messages'] if m['severity'] == 'info']
    assert infos and infos[0]['node'].startswith('meshes/')
    assert all(isinstance(e, int) for m in infos for e in m['entities'] or [])


def test_identity_pair(model):
    # coinciding boundaries: centroids of one side hit the other first
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    final = geom.java.feature('fin')
    final.set('action', 'assembly')
    final.set('createpairs', True)
    model.build(geom)
    (model/'meshes').create(geom)
    model.mesh()
    from mphkit import _mesh
    missed = []
    original = _mesh._missed

    def spy(*args):
        found = original(*args)
        missed.append(int(found.sum()))
        return found

    _mesh._missed = spy
    try:
        quality = mk.mesh_quality(geom, 'boundary')
    finally:
        _mesh._missed = original
    # the first reading missed some elements, the second none
    assert missed[0] > 0 and missed[-1] == 0
    matches(quality, statistics(model, 'mesh1', FACES))
    shared = mk.sel.find(geom, 'boundary', x=1)
    assert len(shared) == 2
    for side in shared:
        assert list(mk.mesh_quality(geom, 'boundary', side)['by_entity']) \
            == [side]


def test_quadrilaterals(model):
    geom = mk.geometry(model, 2)
    mk.rectangle(geom, (2, 1))
    mk.circle(geom, 0.2, (0.5, 0.5))
    model.build(geom)
    (model/'meshes').create(geom).create('FreeQuad')
    model.mesh()
    quality = mk.mesh_quality(geom)
    assert 'quad' in quality['elements']
    matches(quality, statistics(model, 'mesh1', FACES))
    assert len(quality['worst'][0]['position']) == 2
    with pytest.raises(ValueError, match='edge and point elements have no quality'):
        mk.mesh_quality(geom, 'boundary')


def test_narrow_gap(model):
    # COMSOL's information on a short edge and a narrow face
    geom = mk.geometry(model, 3, length_unit='mm')
    mk.block(geom, (10, 10, 10))
    mk.block(geom, (10, 10.02, 10), (10, 0, 0))
    mk.cylinder(geom, 3, 10, (5, 5, 10))
    model.build(geom)
    (model/'meshes').create(geom)
    model.mesh()
    quality = mk.mesh_quality(geom)
    # a short edge and a narrow face (COMSOL 6.4: edges 27-28, boundary 16)
    assert quality['messages']
    assert all(m['severity'] == 'info' and m['entities']
               and m['level'] in ('edge', 'boundary')
               for m in quality['messages'])
    assert quality['histogram'][0] > 0


def test_failed_feature(model):
    # a sweep without a source face: the sphere stays without elements
    geom = mk.geometry(model, 3)
    mk.sphere(geom, 1)
    mk.block(geom, (1, 1, 1), (3, 0, 0))
    model.build(geom)
    mesh = model.java.mesh().create('mesh1', 'geom1')
    tets = mesh.create('ftet1', 'FreeTet')
    tets.selection().geom('geom1', 3)
    tets.selection().set([2])
    sweep = mesh.create('swe1', 'Sweep')
    sweep.selection().geom('geom1', 3)
    sweep.selection().set([1])
    with pytest.raises(Exception):
        model.mesh()
    quality = mk.mesh_quality(geom)
    assert quality['unmeshed'] == [1]
    [error] = [m for m in quality['messages'] if m['severity'] == 'error']
    assert error['node'] == 'meshes/Mesh 1/Swept 1'
    assert 'Source face must be specified.' in error['message']
    assert (error['level'], error['entities']) == ('domain', [1])
    assert mk.mesh_quality(geom, 'domain', 1)['elements'] == {}
    # fixed: the error is gone
    mesh.feature().remove('swe1')
    tets.selection().set([1, 2])
    model.mesh()
    quality = mk.mesh_quality(geom)
    assert quality['unmeshed'] == [] and quality['messages'] == []


def test_only_feature_failed(model):
    geom = mk.geometry(model, 3)
    mk.sphere(geom, 1)
    model.build(geom)
    mesh = model.java.mesh().create('mesh1', 'geom1')
    mesh.create('swe1', 'Sweep')
    with pytest.raises(Exception):
        model.mesh()
    with pytest.raises(RuntimeError,
                       match='Source face must be specified'):
        mk.mesh_quality(geom)


##########
# Errors #
##########

def test_errors(plate, model):
    plate_model, geom = plate
    with pytest.raises(TypeError, match=r"mk.mesh_quality\(geom, 'domain', 3\)"):
        mk.mesh_quality(geom, 3)
    with pytest.raises(ValueError, match='measure must be one of'):
        mk.mesh_quality(geom, measure='quality')
    with pytest.raises(ValueError, match='edge and point elements'):
        mk.mesh_quality(geom, 'edge')
    with pytest.raises(TypeError, match='mesh must be'):
        mk.mesh_quality(geom, mesh=1)
    assert mk.mesh_quality(geom, mesh=True)['mesh'] == 'mesh'
    with pytest.raises(TypeError, match=r"'domain', selection\)"):
        mk.mesh_quality(geom, mk.sel.box(geom, 'boundary', z=0))
    with pytest.raises(ValueError, match='No domain'):
        mk.mesh_quality(geom, 'domain', 5)
    # a mesh changed since it was built
    mk.set(plate_model/'meshes'/'mesh'/'Size', hauto=3)
    with pytest.raises(RuntimeError, match='changed since'):
        mk.mesh_quality(geom)
    # no mesh, 1D, work plane, surfaces only
    other = mk.geometry(model, 3)
    mk.block(other, (1, 1, 1))
    model.build(other)
    with pytest.raises(RuntimeError, match='has no mesh'):
        mk.mesh_quality(other)
    plane = mk.workplane(other, quickz=0)
    with pytest.raises(TypeError, match='work plane'):
        mk.mesh_quality(plane)


def test_surfaces_only(model):
    # a 3D geometry of faces: its elements are on boundaries
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    surface = geom.java.create('csur1', 'ConvertToSurface')
    surface.selection('input').set('blk1')
    model.build(geom)
    assert geom.java.getNDomains() == 0
    with pytest.raises(ValueError, match="pass 'boundary'"):
        mk.mesh_quality(geom)


def test_one_dimension(model):
    geom = mk.geometry(model, 1)
    mk.interval(geom, [0, 1])
    model.build(geom)
    with pytest.raises(ValueError, match='1D'):
        mk.mesh_quality(geom)


def test_named_mesh_errors(model):
    # with mesh=, only that mesh's errors are added
    geom = mk.geometry(model, 3)
    mk.sphere(geom, 1)
    model.build(geom)
    failed = model.java.mesh().create('mesh1', 'geom1')
    failed.create('swe1', 'Sweep')
    good = model.java.mesh().create('mesh2', 'geom1')
    good.create('ftet1', 'FreeTet')
    with pytest.raises(Exception):
        model.mesh()
    good.run()
    # the only mesh with elements is taken without a name
    assert mk.mesh_quality(geom)['mesh'] == 'Mesh 2'
    assert mk.mesh_quality(geom, mesh='mesh2')['mesh'] == 'Mesh 2'
    with pytest.raises(RuntimeError, match='has no elements; its last '
                       'build failed.*Source face'):
        mk.mesh_quality(geom, mesh='Mesh 1')
    mk.set(good.feature('size'), hauto=3)
    for name in ('mesh2', None):
        with pytest.raises(RuntimeError) as error:
            mk.mesh_quality(geom, mesh=name)
        assert 'changed since' in str(error.value)
        assert 'Source face' not in str(error.value)


def test_several_meshes(plate):
    model, geom = plate
    second = (model/'meshes').create(geom, name='fine')
    model.mesh()
    with pytest.raises(ValueError, match='several meshes'):
        mk.mesh_quality(geom)
    assert mk.mesh_quality(geom, mesh='fine')['mesh'] == 'fine'
    assert mk.mesh_quality(geom, mesh=second)['mesh'] == 'fine'


########
# Docs #
########

def test_documented(plate):
    model, geom = plate
    readme = read(Path(__file__).parents[1]/'README.md')
    code = re.search(r'Mesh quality in numbers.*?```python\n(.*?)```',
                     readme, re.S).group(1).splitlines()
    assert len(code) == 3, 'update the checks below with the README'
    namespace = {'mk': mk, 'geom': geom}
    whole, face, other = [eval(line.split('#')[0], namespace)
                          for line in code]
    # the mesh, and so the numbers, vary a little from run to run
    assert 0.03 < whole['min'] < 0.2
    assert face['level'] == 'boundary' and list(face['by_entity']) == [3]
    assert other['measure'] == 'volcircum'
