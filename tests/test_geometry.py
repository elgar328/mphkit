"""Tests for geometry creation and the generic feature() helper."""
import math
import re

import numpy
import pytest

import mphkit as mk
from conftest import count, import_or_skip


def test_geometry_components(model):
    g1 = mk.geometry(model, 3, length_unit='mm', name='A')
    g2 = mk.geometry(model, 2, name='B')
    assert str(g1) == 'geometries/A'
    assert str(g2) == 'geometries/B'
    assert str(g1.java.lengthUnit()) == 'mm'
    assert g2.java.getSDim() == 2
    assert len(list(model.java.component().tags())) == 2
    with pytest.raises(ValueError):
        mk.geometry(model, 3, name='A')
    assert len(list(model.java.component().tags())) == 2


def test_axisymmetric_geometry(model):
    axi = mk.geometry(model, 2, axisymmetric=True)
    assert axi.java.isAxisymmetric() and axi.java.getSDim() == 2
    plain = mk.geometry(model, numpy.int64(2))
    assert not plain.java.isAxisymmetric()
    components = len(list(model.java.component().tags()))
    for dim in (1, 3):
        with pytest.raises(ValueError, match=f'axisymmetric=True is for 2D '
                                             f'geometries .*, not {dim}D'):
            mk.geometry(model, dim, axisymmetric=True)
    for dim in (0, 4):
        with pytest.raises(ValueError, match=f'dim is 1, 2 or 3, not {dim}'):
            mk.geometry(model, dim)
    for dim in ('2D', '2Daxi', 2.0, True):
        with pytest.raises(TypeError, match='dim is the number of dimensions'
                                            '.*pass axisymmetric=True'):
            mk.geometry(model, dim)
    with pytest.raises(TypeError, match='axisymmetric must be True or False'):
        mk.geometry(model, 2, axisymmetric='yes')
    # nothing made by the calls that failed
    assert len(list(model.java.component().tags())) == components


def test_feature_properties(model, geom):
    model.parameter('x', '3')
    blk = mk.feature(geom, 'Block', name='plate', size=(200, 200, 2.5),
                     pos=(0, 'x', 1), base='center', axis=None)
    assert str(blk) == 'geometries/Geometry 1/plate'
    assert list(blk.property('size')) == [200, 200, 2.5]
    assert list(blk.java.getStringArray('pos')) == ['0', 'x', '1']
    assert blk.property('base') == 'center'
    cyl = mk.feature(geom, 'Cylinder', r=numpy.float64(2.5),
                     pos=numpy.array([1, 2, 3]))
    assert cyl.property('r') == 2.5
    assert list(cyl.java.getStringArray('pos')) == ['1', '2', '3']


def test_feature_typo_removes_node(geom):
    with pytest.raises(ValueError, match="'size'"):
        mk.feature(geom, 'Block', name='bad', sise=(1, 1, 1))
    assert not (geom/'bad').exists()
    assert 'blk1' not in list(geom.java.feature().tags())


def test_feature_name_clash(geom):
    mk.feature(geom, 'Block', name='a')
    with pytest.raises(ValueError):
        mk.feature(geom, 'Cylinder', name='a')
    assert 'cyl1' not in list(geom.java.feature().tags())


def test_feature_auto_label_clash(geom):
    # A user label equal to the next automatic label must not be retagged.
    cyl = mk.feature(geom, 'Cylinder', name='Block 1')
    blk = mk.feature(geom, 'Block')
    assert cyl.tag() == 'cyl1'
    assert blk.tag() == 'blk1'
    assert blk.name() != 'Block 1'
    assert cyl.name() == 'Block 1'


def test_feature_type_keyword(geom):
    imp = mk.feature(geom, 'Import', type='cad')
    assert imp.property('type') == 'cad'


def test_difference_inputs(geom):
    mk.feature(geom, 'Block', name='plate', size=(10, 10, 10))
    hole = mk.feature(geom, 'Cylinder', r=2, h=10, pos=(5, 5, 0))
    cut = mk.feature(geom, 'Block', size=(2, 2, 2))
    dif = mk.feature(geom, 'Difference', input='plate', input2=[hole, 'blk2'])
    assert list(dif.java.selection('input').objects()) == ['blk1']
    assert list(dif.java.selection('input2').objects()) == ['cyl1', 'blk2']
    geom.model.build(geom)
    assert count(geom, 'domains') == 1
    with pytest.raises(LookupError):
        mk.feature(geom, 'Union', input=['nothing'])
    assert cut.exists()


def test_input_from_other_geometry(model, geom):
    other = mk.geometry(model, 3)
    blk = mk.feature(other, 'Block')
    mk.feature(geom, 'Block')
    with pytest.raises(ValueError):
        mk.feature(geom, 'Union', input=[blk])


def test_workplane_faces(model):
    # unused 2D objects of a work plane stay as faces (help(mk.workplane))
    for z, faces in ((20, 7), (10, 7)):
        geom = mk.geometry(model, 3, length_unit='mm')
        mk.block(geom, (200, 100, 20))
        plane = mk.workplane(geom, quickz=z)
        mk.square(plane, 20, (90, 40))
        model.build(geom)
        assert mk.summary(geom)['boundaries'] == faces
        if z == 20:      # the top face is split: the patch is its own face
            patch = mk.sel.box(geom, 'boundary', x=(90, 110), y=(40, 60),
                               z=20)
            assert len(mk.sel.entities(geom, patch)) == 1
            assert mk.measure(geom, 'boundary', patch) == pytest.approx(400)


def test_workplane_node(model, geom):
    mk.feature(geom, 'Block', name='a', size=(10, 10, 10))
    mk.feature(geom, 'Block', name='b', size=(10, 10, 10), pos=(20, 0, 0))
    wp = mk.feature(geom, 'WorkPlane', name='wp', quickz=5, unite=True)
    assert isinstance(wp, mk._comsol.WorkPlaneNode)
    sq = mk.feature(wp, 'Square', name='sq', size=10)
    assert isinstance(sq, mk._comsol.WorkPlaneNode)
    assert str(sq) == 'geometries/Geometry 1/wp/sq'
    assert sq.exists()
    assert sq.tag() == 'sq1'
    assert sq.property('size') == 10
    assert 'size' in sq.properties()
    assert sq.parent() == wp
    assert [str(c) for c in wp.children()] == [str(sq)]
    other = wp.create('Circle', name='c')
    assert other.exists() and other.tag() == 'c1'
    other.rename('circle')
    assert (wp/'circle').exists()
    (wp/'circle').remove()
    assert not (wp/'circle').exists()
    model.build(geom)
    assert count(geom, 'boundaries') == 17


def test_named_helpers(model, geom):
    model.parameter('H', '10[mm]')
    plate = mk.block(geom, (200, 200, 'H'), (0, 0, 'H/2'), base='center',
                     name='plate')
    hole = mk.cylinder(geom, 3, 'H', (0, 0, 0), axis=(0, 0, 1))
    solid = mk.difference(geom, plate, [hole], keepsubtract=False)
    mk.sphere(geom, 5, (0, 0, 50))
    mk.point(geom, (0, 0, 100))
    model.build(geom)
    assert solid.exists()
    assert count(geom, 'domains') == 2


def test_workplane_extrude_polygon(model, geom):
    wp = mk.workplane(geom, quickz=0)
    mk.rectangle(wp, (10, 20))
    mk.circle(wp, 2, (30, 0))
    mk.polygon(wp, x=(40, 50, 45), y=(0, 0, 8))
    ext = mk.extrude(geom, wp, 5)
    model.build(geom)
    assert ext.exists()
    assert count(geom, 'domains') == 3


def test_union_rigid_transform(model, geom):
    a = mk.block(geom, (10, 10, 10))
    b = mk.block(geom, (10, 10, 10), (5, 0, 0))
    uni = mk.union(geom, [a, b], intbnd=False)
    mk.rigid_transform(geom, uni, displ=(100, 0, 0))
    model.build(geom)
    assert count(geom, 'domains') == 1
    box = geom.java.getBoundingBox()
    assert box[0] == pytest.approx(100)


def test_import_cad(model, geom):
    # tests/data/part.step: a 10 x 10 x 5 mm block with an r=2 mm hole
    from pathlib import Path
    step = Path(__file__).parent/'data'/'part.step'
    cad = import_or_skip(lambda: mk.import_(geom, step))
    model.build(geom)
    assert cad.property('type') == 'cad'
    assert count(geom, 'domains') == 1
    assert count(geom, 'boundaries') == 10
    assert mk.bounding_box(geom, 'domain')['z'] == pytest.approx((0, 5))


def test_import_missing_file(geom):
    with pytest.raises(FileNotFoundError):
        mk.import_(geom, 'missing.step')


def test_boolean_string_property(model, geom):
    # All three are string-typed switches ('on'/'off'), but `sellayer`
    # rejects a Java boolean while `selresult` and `unite` accept one:
    # True/False must work for all of them.
    blk = mk.block(geom, (40, 40, 40), name='free', layername=['pml'],
                   layer=[5], layertop=True, sellayer=True)
    assert blk.java.getString('sellayer') == 'on'
    blk.property('sellayer', 'off')
    mk.feature(geom, 'Block', sellayer=False, size=(1, 1, 1))
    wp = mk.workplane(geom, quickz=1, unite=True)
    assert wp.java.getString('unite') == 'on'
    pt = mk.point(geom, (0, 0, 5), selresult=True)
    assert pt.java.getString('selresult') == 'on'


def test_property_error_lists_allowed_values(geom):
    with pytest.raises(ValueError, match='center'):
        mk.block(geom, (1, 1, 1), base='middle')


def test_component_of(model):
    first = mk.geometry(model, 3, name='first')
    second = mk.geometry(model, 2, name='second')
    a, b = mk.component_of(first), mk.component_of(second)
    assert a.tag() != b.tag()
    assert str(a).startswith('components/')
    assert first.tag() in [str(t) for t in a.java.geom().tags()]
    assert second.tag() in [str(t) for t in b.java.geom().tags()]


def test_component_of_coordinate_system(model, geom):
    mk.block(geom, (40, 40, 40), name='free', layername=['pml'], layer=[5],
             layertop=True, layerbottom=False, sellayer=True)
    model.build(geom)
    container = mk.component_of(geom).java.coordSystem()
    tag = str(container.uniquetag('pml'))
    container.create(tag, geom.tag(), 'PML')
    container.get(tag).label('PML region')
    node = model/'coordinates'/'PML region'
    assert node.exists()
    node.select(mk.sel.layer(geom, geom/'free', 'pml'))
    assert len(list(node.java.selection().entities())) > 0


def test_coordinate_system(model):
    first = mk.geometry(model, 3, name='first')
    mk.block(first, (1, 1, 1))
    model.build(first)
    second = mk.geometry(model, 3, name='second')
    mk.block(second, (40, 40, 40), name='free', layername=['pml'],
             layer=[5], layertop=True, layerbottom=False, sellayer=True)
    model.build(second)
    pml = mk.sel.layer(second, second/'free', 'pml')
    node = mk.coordinate_system(second, 'PML', selection=pml, name='PML',
                                stretchingType='rational')
    assert str(node) == 'coordinates/PML'

    def tags(geom):
        return [str(t) for t in mk.component_of(geom).java.coordSystem().tags()]

    assert node.tag() in tags(second)
    assert node.tag() not in tags(first)
    assert node.property('stretchingType') == 'rational'
    assert node.selection() == pml


def test_coordinate_system_errors(model, geom):
    mk.block(geom, (40, 40, 40), name='free', layername=['pml'], layer=[5],
             layertop=True, layerbottom=False, sellayer=True)
    model.build(geom)
    pml = mk.sel.layer(geom, geom/'free', 'pml')
    mk.coordinate_system(geom, 'PML', name='PML')
    other = mk.geometry(model, 3)    # its component adds a boundary system

    def systems():
        return [str(t) for t in model.java.coordSystem().tags()]

    before = systems()
    with pytest.raises(ValueError, match='already in use'):
        mk.coordinate_system(geom, 'PML', name='PML')
    with pytest.raises(ValueError, match='already in use'):   # model-wide
        mk.coordinate_system(other, 'PML', name='PML')
    with pytest.raises(ValueError, match='has no selection'):
        mk.coordinate_system(geom, 'Rotated', selection=pml)
    with pytest.raises(ValueError, match='not at the level'):
        mk.coordinate_system(geom, 'PML',
                             selection=mk.sel.box(geom, 'boundary', z=0))
    with pytest.raises(ValueError, match='stretchingType'):
        mk.coordinate_system(geom, 'PML', stretchingtype='rational')
    with pytest.raises(TypeError, match='not entity numbers such as 1: '):
        mk.coordinate_system(geom, 'PML', selection=1)
    with pytest.raises(Exception):
        mk.coordinate_system(geom, 'NoSuchType')
    assert systems() == before
    assert mk.coordinate_system(geom, 'Rotated').exists()


def bbox(geom, entity='domain', selection=None):
    """Bounding box rounded to 6 digits, for exact comparisons."""
    box = mk.bounding_box(geom, entity, selection)
    return {axis: tuple(round(v, 6) + 0.0 for v in pair)
            for axis, pair in box.items()}


def test_array(model):
    grid = mk.geometry(model, 3)
    b = mk.block(grid, (1, 1, 1))
    mk.array(grid, b, size=(3, 2, 1), displ=(2, 2, 0))
    model.build(grid)
    assert count(grid, 'domains') == 6
    assert 'arr1(3,2,1)' in [str(o) for o in grid.java.objectNames()]
    line = mk.geometry(model, 3)
    arr = mk.array(line, mk.block(line, (1, 1, 1)), size=4, displ=(2, 0, 0))
    model.build(line)
    assert count(line, 'domains') == 4
    assert arr.property('type') == 'linear'
    model.parameter('n', '3')
    expr = mk.geometry(model, 3)
    mk.array(expr, mk.block(expr, (1, 1, 1)), size='n', displ=(2, 0, 0))
    model.build(expr)
    assert count(expr, 'domains') == 3
    given = mk.array(expr, 'blk1', size=2, displ=(0, 2, 0),
                     type='three-dimensional')   # a given type is kept
    assert given.property('type') == 'three-dimensional'


def test_array_errors(geom):
    b = mk.block(geom, (1, 1, 1))
    with pytest.raises(TypeError, match='size must be'):
        mk.array(geom, b, size=True, displ=(1, 0, 0))
    with pytest.raises(ValueError, match='size needs 3'):
        mk.array(geom, b, size=(2, 2), displ=(1, 1, 0))
    with pytest.raises(TypeError, match='positional'):
        mk.array(geom, b, (2, 2, 1), (3, 3, 0))


def test_array_in_workplane(model, geom):
    plane = mk.workplane(geom, quickz=0)
    circle = mk.circle(plane, 0.4, (0.5, 0.5))
    mk.array(plane, circle, size=(3, 2), displ=(1, 1))
    mk.extrude(geom, plane, 1)
    model.build(geom)
    assert count(geom, 'domains') == 6


def test_move(model):
    up = mk.geometry(model, 3)
    mk.move(up, mk.block(up, (1, 1, 1)), (0, 0, 5))
    model.build(up)
    assert bbox(up)['z'] == (5, 6)
    copies = mk.geometry(model, 3)
    mk.move(copies, mk.block(copies, (1, 1, 1)), ([2, 4], 0, 0))
    model.build(copies)
    assert count(copies, 'domains') == 2
    assert bbox(copies)['x'] == (2, 5)
    kept = mk.geometry(model, 3)
    mk.move(kept, mk.block(kept, (1, 1, 1)), (3, 0, 0), keep=True)
    model.build(kept)
    assert count(kept, 'domains') == 2
    flat = mk.geometry(model, 3)
    plane = mk.workplane(flat, quickz=0)
    mk.move(plane, mk.square(plane, 1), (3, 0))
    mk.extrude(flat, plane, 1)
    model.build(flat)
    assert bbox(flat)['x'] == (3, 4)
    with pytest.raises(ValueError, match='displ needs 3'):
        mk.move(flat, 'ext1', 5)


def test_rotate(model):
    def turned(**kwargs):
        geom = mk.geometry(model, 3)
        mk.rotate(geom, mk.block(geom, (1, 2, 3)), 90, **kwargs)
        model.build(geom)
        return bbox(geom)
    assert turned()['x'] == (-2, 0)
    about_x = {'x': (0, 1), 'y': (-3, 0), 'z': (0, 2)}
    assert turned(axis=(1, 0, 0)) == about_x
    assert turned(axis='x') == about_x
    copies = mk.geometry(model, 3)
    mk.rotate(copies, mk.block(copies, (1, 1, 1), (2, 0, 0)), [0, 90, 180])
    model.build(copies)
    assert count(copies, 'domains') == 3
    flat = mk.geometry(model, 3)
    plane = mk.workplane(flat, quickz=0)
    square = mk.square(plane, 1, (1, 0))
    mk.rotate(plane, square, 90)
    mk.extrude(flat, plane, 1)
    model.build(flat)
    assert bbox(flat)['x'] == (-1, 0)
    with pytest.raises(ValueError, match='no axis'):
        mk.rotate(plane, square, 90, axis='x')


def test_mirror(model):
    both = mk.geometry(model, 3)
    mk.mirror(both, mk.block(both, (1, 1, 1), (1, 0, 0)), (1, 0, 0),
              keep=True)
    model.build(both)
    assert count(both, 'domains') == 2
    assert bbox(both)['x'] == (-2, 2)
    # 2D: the axis is the normal of the mirror line, so x flips
    flat = mk.geometry(model, 2)
    mk.mirror(flat, mk.square(flat, 0.2, (1, 0.5)), (1, 0))
    model.build(flat)
    box = mk.bounding_box(flat, 'domain')
    assert box['x'] == pytest.approx((-1.2, -1))
    assert box['y'] == pytest.approx((0.5, 0.7))


def test_intersection(model, geom):
    a = mk.block(geom, (2, 2, 2))
    b = mk.block(geom, (2, 2, 2), (1, 1, 1))
    mk.intersection(geom, [a, b])
    model.build(geom)
    assert count(geom, 'domains') == 1
    assert bbox(geom) == {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)}


def test_delete(model, geom):
    a = mk.block(geom, (1, 1, 1))
    c = mk.block(geom, (1, 1, 1), (3, 0, 0))
    gone = mk.delete(geom, c)                    # failed before: input starts
    model.build(geom)                            # at boundary level
    assert count(geom, 'domains') == 1
    assert bbox(geom)['x'] == (0, 1)
    mk.set(gone, input=[a])
    model.build(geom)
    assert bbox(geom)['x'] == (3, 4)


def test_entity_inputs_stay_entity_level(geom):
    block = mk.block(geom, (1, 1, 1))
    turn = mk.rotate(geom, block, 30)
    edge = turn.java.selection('edge')
    before = [int(d) for d in edge.dimension()]
    with pytest.raises(TypeError):
        mk.set(turn, edge=[block])
    assert [int(d) for d in edge.dimension()] == before


def test_objects_to_entity_input(model, geom):
    # Used to switch the input to objects, which COMSOL then refused.
    block = mk.block(geom, (1, 1, 1))
    rounded = mk.feature(geom, 'Fillet3D', edge=[block], radius=0.1)
    assert [int(d) for d in rounded.java.selection('edge').dimension()] == [1]
    model.build(geom)
    assert count(geom, 'boundaries') == 26


def test_revolve(model):
    def ring(*args, plane='xz', **kwargs):
        geom = mk.geometry(model, 3)
        plane = mk.workplane(geom, quickplane=plane)
        mk.rectangle(plane, (1, 1), (1, 0))
        node = mk.revolve(geom, plane, *args, **kwargs)
        model.build(geom)
        return geom, node
    full, _ = ring()
    # a full turn keeps the cross-section as an interior face
    assert count(full, 'boundaries') == 17
    assert count(ring(origfaces=False)[0], 'boundaries') == 16
    assert count(ring(360, origfaces=False)[0], 'boundaries') == 17
    # swept faces are split every 90 degrees from the start angle
    assert count(ring(100)[0], 'boundaries') == 10
    assert count(ring((30, 120))[0], 'boundaries') == 6
    assert count(ring(270)[0], 'boundaries') == 14
    assert bbox(ring(plane='yz')[0]) == bbox(full)    # about the global z
    model.parameter('a', '90')
    assert count(ring('a')[0], 'boundaries') == 6
    assert mk.measure(full, 'domain') == pytest.approx(3*math.pi, rel=0.01)
    assert bbox(full)['x'] == (-2, 2)
    half, _ = ring(180)
    assert mk.measure(half, 'domain') == pytest.approx(1.5*math.pi, rel=0.01)
    quarter, _ = ring((0, 90))
    assert mk.measure(quarter, 'domain') == \
        pytest.approx(0.75*math.pi, rel=0.01)
    given, _ = ring(angtype='specang', angle2=90)   # own properties kept
    assert mk.measure(given, 'domain') == pytest.approx(0.75*math.pi,
                                                        rel=0.01)
    about_x = {'x': (1, 2), 'y': (-1, 1), 'z': (-1, 1)}
    assert bbox(ring(axis=(1, 0, 0))[0]) == about_x
    assert bbox(ring(axis='x')[0]) == about_x
    # the work plane's own x axis through (0, 0): the same axis here
    assert bbox(ring(pos=(0, 0), axis=(1, 0))[0]) == about_x
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom, quickplane='xz')
    with pytest.raises(ValueError, match='either angle'):
        mk.revolve(geom, plane, 90, angle2=90)
    with pytest.raises(ValueError, match='two values each'):
        mk.revolve(geom, plane, pos=(0, 0), axis='x')
    with pytest.raises(ValueError, match='needs an axis'):
        mk.revolve(geom, plane, pos=(0, 0, 0))
    with pytest.raises(ValueError, match='3D geometry'):
        mk.revolve(plane, 'r1')
    with pytest.raises(ValueError, match='3D geometry'):
        mk.revolve(mk.geometry(model, 2), 'r1')


def test_partition(model):
    thin = mk.geometry(model, 3)
    block = mk.block(thin, (2, 1, 1))
    mk.partition(thin, block, [mk.block(thin, (0.5, 3, 3), (1, -1, -1))])
    model.build(thin)
    assert count(thin, 'domains') == 3
    for tool in ('node', 'name'):
        cut = mk.geometry(model, 3)
        block = mk.block(cut, (2, 1, 1))
        plane = mk.workplane(cut, quickplane='yz', quickx=1, name='cut')
        mk.partition(cut, block, plane if tool == 'node' else 'cut')
        model.build(cut)
        assert count(cut, 'domains') == 2


def fillet_volume(a, r):
    """Volume of a cube of side `a` with all edges rounded by `r`."""
    return (a**3 - 3*(4 - math.pi)*r**2*(a - 2*r) - 8*r**3
            + 4*math.pi*r**3/3)


def test_fillet(model):
    whole = mk.geometry(model, 3)
    mk.fillet(whole, mk.block(whole, (1, 1, 1)), 0.1)
    model.build(whole)
    assert count(whole, 'boundaries') == 26
    assert mk.measure(whole, 'domain') == pytest.approx(fillet_volume(1, 0.1),
                                                        rel=0.005)
    top = mk.geometry(model, 3)
    mk.block(top, (1, 1, 1))
    edges = mk.sel.box(top, 'edge', z=1, where='geometry')
    rounded = mk.fillet(top, 'blk1', 0.1)
    model.build(top)
    assert count(top, 'boundaries') == 26
    mk.set(rounded, edge=edges)                  # the top only, nothing left
    model.build(top)
    assert count(top, 'boundaries') == 10
    rim = mk.geometry(model, 3)                  # edges around a face
    mk.block(rim, (1, 1, 1))
    face = mk.sel.box(rim, 'boundary', z=1, where='geometry')
    mk.fillet(rim, mk.sel.adjacent(rim, face, 'edge', input_entity='boundary',
                                   where='geometry'), 0.1)
    model.build(rim)
    assert count(rim, 'boundaries') == 10
    two = mk.geometry(model, 3)
    a = mk.block(two, (1, 1, 1))
    b = mk.block(two, (1, 1, 1), (3, 0, 0))
    mk.fillet(two, [a, b], 0.1)
    model.build(two)
    assert count(two, 'boundaries') == 52
    copies = mk.geometry(model, 3)
    mk.array(copies, mk.block(copies, (1, 1, 1)), size=(2, 1, 1),
             displ=(3, 0, 0))
    mk.fillet(copies, 'arr1(1,1,1)', 0.1)
    model.build(copies)
    assert count(copies, 'boundaries') == 32


def test_fillet_follows_object(model, geom):
    model.parameter('L', '1')
    mk.fillet(geom, mk.block(geom, ('L', 1, 1)), 0.1)
    model.build(geom)
    model.parameter('L', '2')
    model.build(geom)
    assert count(geom, 'boundaries') == 26
    assert mk.measure(geom, 'domain') == pytest.approx(1.9669, rel=0.005)


def test_fillet_2d_and_workplane(model):
    flat = mk.geometry(model, 2)
    mk.fillet(flat, mk.square(flat, 1), 0.2)
    model.build(flat)
    assert mk.measure(flat, 'domain') == \
        pytest.approx(1 - (4 - math.pi)*0.04, rel=0.01)
    solid = mk.geometry(model, 3)
    plane = mk.workplane(solid, quickz=0)
    mk.fillet(plane, mk.square(plane, 1), 0.2)
    mk.extrude(solid, plane, 1)
    model.build(solid)
    assert mk.measure(solid, 'domain') == \
        pytest.approx(1 - (4 - math.pi)*0.04, rel=0.01)
    with pytest.raises(ValueError, match='not to the work plane'):
        mk.fillet(plane, mk.sel.box(solid, 'point', where='geometry'), 0.1)
    with pytest.raises(ValueError, match='1D'):
        mk.fillet(mk.geometry(model, 1), 'i1', 0.1)


def test_fillet_errors_leave_input(model, geom):
    block = mk.block(geom, (1, 1, 1))
    faces = mk.sel.box(geom, 'boundary', z=1, where='geometry')
    with pytest.raises(ValueError, match='level 1'):
        mk.fillet(geom, faces, 0.1)
    assert not [t for t in geom.java.feature().tags()
                if str(t).startswith('fil')]
    rounded = mk.fillet(geom, block, 0.1)
    edge = rounded.java.selection('edge')
    before = [str(o) for o in edge.objects()]
    rims = mk.sel.cumulative(geom, 'rims', 'boundary', create=True)
    with pytest.raises(ValueError, match='level 1'):
        mk.set(rounded, edge=rims)
    objects = mk.sel.box(geom, 'object', where='geometry')
    with pytest.raises(ValueError, match='level 1'):
        mk.set(rounded, edge=objects)
    assert [str(o) for o in edge.objects()] == before
    model.build(geom)
    assert count(geom, 'boundaries') == 26        # still all edges


def test_fillet_cumulative_edges(model, geom):
    rims = mk.sel.cumulative(geom, 'rims', 'edge', create=True)
    mk.block(geom, (1, 1, 1), contributeto=rims)
    mk.fillet(geom, rims, 0.1)
    model.build(geom)
    assert count(geom, 'boundaries') == 26


def test_chamfer(model):
    solid = mk.geometry(model, 3)
    mk.chamfer(solid, mk.block(solid, (1, 1, 1)), 0.3)
    model.build(solid)
    assert count(solid, 'boundaries') == 26
    assert 1 - 6*0.09 < mk.measure(solid, 'domain') < 1 - 6*0.09*0.4
    flat = mk.geometry(model, 2)
    mk.square(flat, 1)
    corner = mk.sel.box(flat, 'point', x=1, y=1, where='geometry')
    mk.chamfer(flat, corner, 0.2)
    model.build(flat)
    assert mk.measure(flat, 'domain') == pytest.approx(1 - 0.02)


def test_line_segment(model, geom):
    mk.line_segment(geom, (0, 0, 0), (1, 2, 3))
    model.build(geom)
    assert mk.measure(geom, 'edge') == pytest.approx(math.sqrt(14))
    flat = mk.geometry(model, 2)
    mk.line_segment(flat, (0, 0), (3, 4))
    model.build(flat)
    assert mk.measure(flat, 'boundary') == pytest.approx(5)
    with pytest.raises(ValueError, match='end needs 3'):
        mk.line_segment(geom, (0, 0, 0), (1, 1))
    lifted = mk.geometry(model, 3)
    plane = mk.workplane(lifted, quickz=1)
    mk.line_segment(plane, (0, 0), (2, 0))
    model.build(lifted)
    assert mk.measure(lifted, 'edge') == pytest.approx(2)
    assert bbox(lifted, 'edge')['z'] == (1, 1)


def test_interval(model, geom):
    line = mk.geometry(model, 1)
    mk.interval(line, [0, 1, 3])
    model.build(line)
    assert count(line, 'domains') == 2
    assert mk.measure(line, 'domain') == pytest.approx(3)
    with pytest.raises(ValueError, match='1D'):
        mk.interval(geom, [0, 1])


def test_cumulative_as_input(model):
    union = mk.geometry(model, 3)
    parts = mk.sel.cumulative(union, 'parts', 'domain', create=True)
    mk.block(union, (1, 1, 1), contributeto=parts)
    mk.block(union, (1, 1, 1), (0.5, 0, 0), contributeto=parts)
    mk.block(union, (1, 1, 1), (5, 0, 0))
    mk.union(union, parts)
    model.build(union)
    assert sorted(str(o) for o in union.java.objectNames()) == \
        ['blk3', 'uni1']
    cut = mk.geometry(model, 3)
    holes = mk.sel.cumulative(cut, 'holes', 'domain', create=True)
    plate = mk.block(cut, (10, 10, 1))
    for x in (3, 7):
        mk.cylinder(cut, 1, 1, (x, x, 0), contributeto=holes)
    mk.difference(cut, plate, holes)
    model.build(cut)
    assert count(cut, 'domains') == 1
    assert mk.measure(cut, 'domain') == pytest.approx(100 - 2*math.pi,
                                                      rel=0.01)
    gone = mk.geometry(model, 3)
    junk = mk.sel.cumulative(gone, 'junk', 'domain', create=True)
    mk.block(gone, (1, 1, 1))
    mk.block(gone, (1, 1, 1), (3, 0, 0), contributeto=junk)
    mk.delete(gone, junk)
    model.build(gone)
    assert [str(o) for o in gone.java.objectNames()] == ['blk1']


def test_delete_domain_of_object(model, geom):
    layered = mk.block(geom, (4, 1, 1), layername=['a'], layer=[1],
                       layerleft=True, layerbottom=False)
    other = mk.block(geom, (1, 1, 1), (10, 0, 0))
    first = mk.sel.box(geom, 'domain', x=(-0.1, 1.1), y=(-0.1, 1.1),
                       z=(-0.1, 1.1), where='geometry')
    gone = mk.delete(geom, first)
    model.build(geom)
    assert count(geom, 'domains') == 2           # one of the layered two
    mk.set(gone, input=[other])
    model.build(geom)
    assert count(geom, 'domains') == 2
    mk.set(gone, input=first)                    # same result after objects
    model.build(geom)
    assert count(geom, 'domains') == 2
    assert bbox(geom)['x'] == (1, 11)
    mk.set(gone, input=geom/'Box Selection 1')   # the feature node itself
    model.build(geom)
    assert count(geom, 'domains') == 2
    assert layered.exists()


def test_delete_explicit_selection(model, geom):
    # An explicit selection has no entitydim; its level comes from the
    # derived selection, so only the chosen domain goes.
    mk.block(geom, (4, 1, 1), layername=['a'], layer=[1], layerleft=True,
             layerbottom=False)
    chosen = mk.feature(geom, 'ExplicitSelection', name='chosen')
    chosen.java.selection('selection').init(3)
    chosen.java.selection('selection').set('blk1', [1])
    mk.delete(geom, chosen)
    model.build(geom)
    assert count(geom, 'domains') == 1
    assert [str(o) for o in geom.java.objectNames()] == ['del1']


@pytest.mark.parametrize('make, meant', [
    (lambda g: mk.cylinder(g, radius=1, h=2), "'r'"),
    (lambda g: mk.block(g, (1, 1, 1), position=(0, 0, 0)), "'pos'"),
    (lambda g: mk.block(g, (1, 1, 1), center=(0, 0, 0)),
     "'pos' with base='center'"),
    (lambda g: mk.circle(mk.workplane(g), 1, center=(0, 0)), "'pos'?"),
    (lambda g: mk.block(g, (1, 1, 1), dims=(1, 1, 1)), "'size'"),
    (lambda g: (mk.block(g, (1, 1, 1)), mk.feature(g, 'Array', count=3)),
     "'size'"),
    (lambda g: (mk.block(g, (1, 1, 1)), mk.feature(g, 'Rotate', angle=45)),
     "'rot'"),
    (lambda g: mk.feature(g, 'Torus', radius=1), "'rmaj', 'rmin'"),
    (lambda g: mk.cylinder(g, 1, 2, center=(0, 0, 0)),
     "'pos' (the center of the base)"),
    (lambda g: mk.block(g, (1, 1, 1), POS=(0, 0, 0)), "'pos'?"),
    (lambda g: mk.workplane(g, normal=(0, 0, 1)), "'normalvector'"),
])
def test_property_suggestion(geom, make, meant):
    with pytest.raises(ValueError) as error:
        make(geom)
    text = str(error.value)
    assert f'Did you mean {meant}' in text
    assert re.search(r"Extra keyword arguments are COMSOL property names; "
                     r"mk\.properties\((geom|plane), '\w+', search=\.\.\.\) "
                     r"lists them\.$", text)
    assert "'axis'" not in text


def test_property_listing(model, geom):
    # without a close name, the call that lists the properties
    names = 'Extra keyword arguments are COMSOL property names'
    with pytest.raises(ValueError, match=f"{names}; "
                       r"mk\.properties\(geom, 'Block', search=\.\.\.\) "
                       r"lists them\.$"):
        mk.block(geom, (1, 1, 1), bogus=1)
    plane = mk.workplane(geom)
    with pytest.raises(ValueError, match=r"^\"Circle\" has no property "
                       f'"rad". {names}; '
                       r"mk\.properties\(plane, 'Circle', search=\.\.\.\)"):
        mk.circle(plane, 1, rad=1)
    block = mk.block(geom, (1, 1, 1))
    model.build(geom)
    node = r'mk\.properties\(node, search=\.\.\.\) lists them\.$'
    with pytest.raises(ValueError, match=f'{names}; {node}'):
        mk.set(block, bogus=1)
    assert 'size' in mk.properties(block, search='size')   # as the hint says
    physics = (model/'physics').create('HeatTransfer', geom)
    flux = physics.create('HeatFluxBoundary', 2)
    with pytest.raises(ValueError, match=r'^"HeatFluxBoundary" has no '
                       f'property "h_coef". {names}; {node}'):
        mk.set(flux, h_coef=50)
    assert 'h' in mk.properties(flux, search='coefficient')
    # no hint where mk.properties does not list the node's own properties
    for target in (flux.java, mk.sel.box(geom, 'boundary', x=0)):
        with pytest.raises(ValueError, match=f'{names}\\.$'):
            mk.set(target, bogus=1)
    with pytest.raises(ValueError, match=f'{names}\\.$'):
        mk.sel.box(geom, 'boundary', x=0, where='geometry', bogus=1)
    mk.sel.box(geom, 'boundary', x=0, where='geometry', name='seqbox')
    with pytest.raises(ValueError, match=f'{names}\\.$'):
        mk.set(geom/'seqbox', bogus=1)
    mesh = (model/'meshes').create(geom)
    study = (model/'studies').create()
    study.create('Stationary')
    system = mk.coordinate_system(geom, 'Rotated')
    for target in (geom, mesh, study, physics, system):
        # COMSOL's own error or none at all: either way no hint
        try:
            mk.set(target, bogus=1)
        except Exception as error:
            assert 'mk.properties' not in str(error), target


@pytest.mark.parametrize('dim, types', [
    (3, ['Block', 'Cylinder', 'Sphere', 'Point', 'Union', 'Difference',
         'RigidTransform', 'Intersection', 'Delete', 'Array', 'Move',
         'Rotate', 'Mirror', 'Revolve', 'Partition', 'Fillet3D', 'Chamfer3D',
         'LineSegment', 'WorkPlane', 'Extrude', 'Import']),
    ('plane', ['Square', 'Rectangle', 'Circle', 'Polygon', 'Fillet',
               'Chamfer', 'LineSegment', 'Union', 'Point', 'Difference',
               'Intersection', 'Delete', 'Array', 'Move', 'Rotate', 'Mirror',
               'Partition']),
    (2, ['Square', 'Rectangle', 'Circle', 'Polygon', 'Fillet', 'Chamfer',
         'LineSegment', 'Union', 'Difference', 'Intersection', 'Delete',
         'Array', 'Move', 'Rotate', 'Mirror', 'Partition', 'Point']),
    (1, ['Interval']),
])
def test_property_listing_types(model, dim, types):
    # every type the helpers create can be listed, as the hint says
    geom = mk.geometry(model, 3 if dim == 'plane' else dim)
    parent = mk.workplane(geom) if dim == 'plane' else geom
    for type in types:
        assert mk.properties(parent, type), type


def test_property_suggestion_by_type(geom):
    mk.block(geom, (1, 1, 1), name='b')
    with pytest.raises(ValueError, match="'displx', 'disply', 'displz'"):
        mk.feature(geom, 'Move', input=['b'], offset=(1, 0, 0))
    # the axis of a Revolve is its rotation axis, not a normal
    plane = mk.workplane(geom)
    mk.circle(plane, 1, (3, 0))
    with pytest.raises(ValueError) as error:
        mk.revolve(geom, plane, normal=(0, 1))
    assert "'axis'" not in str(error.value)
    # an alias without a candidate still gets the close names
    with pytest.raises(ValueError, match="'angles'|'angle1'"):
        mk.feature(geom, 'Revolve', input=[plane.tag()], angle=90)


def test_workplane_defaults(geom):
    plane = mk.workplane(geom)
    assert plane.java.getString('quickplane') == 'xy'
    assert plane.java.getString('quickz') == '0'


@pytest.mark.parametrize('quickplane, box', [
    ('xy', {'x': (0, 1), 'y': (0, 2), 'z': (0, 3)}),
    ('yz', {'x': (0, 3), 'y': (0, 1), 'z': (0, 2)}),
    ('zx', {'x': (0, 2), 'y': (0, 3), 'z': (0, 1)}),
    ('xz', {'x': (0, 1), 'y': (-3, 0), 'z': (0, 2)}),
    ('zy', {'x': (-3, 0), 'y': (0, 2), 'z': (0, 1)}),
    ('yx', {'x': (0, 2), 'y': (0, 1), 'z': (-3, 0)}),
])
def test_workplane_axes(model, geom, quickplane, box):
    plane = mk.workplane(geom, quickplane=quickplane)
    mk.rectangle(plane, (1, 2))
    mk.extrude(geom, plane, 3)
    model.build(geom)
    found = mk.bounding_box(geom, 'domain')
    for axis in 'xyz':
        assert found[axis] == pytest.approx(box[axis], abs=1e-6)


def test_extrude_distance_list(model, geom):
    plane = mk.workplane(geom)
    mk.square(plane, 2)
    mk.extrude(geom, plane, [1, 3])
    model.build(geom)
    assert count(geom, 'domains') == 2
    assert len(mk.sel.find(geom, 'boundary', z=1)) == 1
    assert len(mk.sel.find(geom, 'boundary', z=3)) == 1


LAYERED = {'layername': ['pml'], 'layer': [5], 'layertop': True,
           'layerbottom': False}


def built(model, geom):
    """Builds `geom` and returns its numbers of domains and boundaries."""
    model.build(geom)
    return count(geom, 'domains'), count(geom, 'boundaries')


def counts(model, make):
    """Runs `make(geom)` in a new 3D geometry and returns `built()`."""
    geom = mk.geometry(model, 3)
    make(geom)
    return built(model, geom)


def boss(geom):
    """A cylinder standing on a plate."""
    return [mk.block(geom, (10, 10, 2)), mk.cylinder(geom, 1, 3, (5, 5, 2))]


def overlapping(geom):
    return [mk.block(geom, (1, 1, 1)), mk.block(geom, (1, 1, 1), (0.5, 0, 0))]


def apart(geom):
    return [mk.block(geom, (1, 1, 1)), mk.block(geom, (1, 1, 1), (3, 0, 0))]


def layered(geom):
    return [mk.block(geom, (40, 40, 40), **LAYERED),
            mk.block(geom, (10, 10, 10), (40, 0, 0))]


def cut(geom):
    """A block cut in two by a partition, and a block touching it."""
    blk = mk.block(geom, (2, 1, 1))
    plane = mk.workplane(geom, quickplane='yz', quickx=1)
    return [mk.partition(geom, blk, plane),
            mk.block(geom, (1, 1, 1), (0, 1, 0))]


@pytest.mark.parametrize('objects, apart_, united, merged', [
    (boss, (2, 12), (2, 12), (1, 11)),
    (overlapping, (3, 16), (3, 16), (1, 14)),
    (layered, (3, 17), (3, 17), (1, 15)),
    (cut, (3, 16), (3, 16), (1, 14)),
])
def test_union_keeps_interior_boundaries(model, objects, apart_, united,
                                         merged):
    assert counts(model, objects) == apart_
    assert counts(model, lambda g: mk.union(g, objects(g))) == united
    assert counts(model, lambda g: mk.union(g, objects(g),
                                            intbnd=False)) == merged


def test_union_keeps_objects_apart(model):
    assert counts(model, lambda g: mk.union(g, apart(g),
                                            intbnd=False)) == (2, 12)


@pytest.mark.parametrize('objects, kept, merged', [
    (lambda g: [mk.block(g, (1, 1, 1)), mk.block(g, (1, 1, 1), (1, 0, 0))],
     (2, 11), (1, 10)),
    (overlapping, (3, 16), (1, 14)),
])
def test_difference_intbnd(model, objects, kept, merged):
    def cut_away(geom, **properties):
        tool = mk.block(geom, (0.1, 0.1, 0.1), (5, 5, 5))
        mk.difference(geom, objects(geom), [tool], **properties)

    assert counts(model, cut_away) == kept
    assert counts(model, lambda g: cut_away(g, intbnd=False)) == merged


def test_intersection_intbnd_layers(model):
    def common(geom, **properties):
        around = mk.block(geom, (60, 60, 60), (-10, -10, -10))
        block = mk.block(geom, (40, 40, 40), **LAYERED)
        mk.intersection(geom, [block, around], **properties)

    assert counts(model, common) == (2, 11)
    assert counts(model, lambda g: common(g, intbnd=False)) == (1, 10)


def test_workplane_objects_stay_separate(model):
    def extruded(rectangles, unite=None):
        geom = mk.geometry(model, 3)
        plane = mk.workplane(geom)
        shapes = [mk.rectangle(plane, size, pos) for size, pos in rectangles]
        if unite == 'plane':
            mk.union(plane, shapes, intbnd=False)
        solid = mk.extrude(geom, plane, 1)
        if unite == 'after':
            mk.union(geom, [solid], intbnd=False)
        return built(model, geom)

    touching = [((2, 1), None), ((1, 2), (2, 0))]
    assert extruded(touching) == (2, 12)
    assert extruded(touching, 'plane') == (1, 9)
    assert extruded(touching, 'after') == (1, 11)   # top and bottom split
    assert extruded([((2, 1), None), ((1, 2), (1.5, 0))])[0] == 3
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom, quickplane='xz')
    mk.rectangle(plane, (1, 1), (1, 0))
    mk.rectangle(plane, (1, 1), (2, 0))
    mk.revolve(geom, plane)
    assert built(model, geom)[0] == 2


def test_workplane_difference(model):
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom)
    outer = mk.square(plane, 2)
    notch = mk.square(plane, 1, (1, 1))
    mk.difference(plane, outer, [notch])
    mk.extrude(geom, plane, 1)
    assert built(model, geom)[0] == 1
    assert mk.measure(geom, 'domain') == pytest.approx(3)


def test_curved_faces_split(model):
    def boundaries(make):
        geom = mk.geometry(model, 3)
        make(geom)
        return geom, built(model, geom)[1]

    geom, n = boundaries(lambda g: mk.cylinder(g, 1, 2))
    assert n == 6
    side = mk.sel.cylinder(geom, 'boundary', (0, 0, 0), 1.01, rin=0.99,
                           bottom=-0.02, top=2.02)
    assert len(mk.sel.entities(geom, side)) == 4
    assert boundaries(lambda g: mk.sphere(g, 1))[1] == 8
    for rtop, expected in ((0, 5), (0.5, 6)):
        cone = boundaries(lambda g: mk.feature(
            g, 'Cone', r=1, h=2, specifytop='radius', rtop=rtop))
        assert cone[1] == expected

    def pillar(g):
        plane = mk.workplane(g)
        mk.circle(plane, 1)
        mk.extrude(g, plane, 2)

    assert boundaries(pillar)[1] == 6

    def plate(g):
        mk.difference(g, mk.block(g, (10, 10, 1)),
                      [mk.cylinder(g, 1, 1, (5, 5, 0))])

    geom, n = boundaries(plate)
    assert n == 10
    wall = mk.sel.cylinder(geom, 'boundary', (5, 5, 0), 1.01, rin=0.99,
                           bottom=-0.01, top=1.01)
    assert len(mk.sel.entities(geom, wall)) == 4


def test_geometry_inputs_reject_numbers(model):
    # entity numbers fail before anything is created, with a way out
    geom = mk.geometry(model, 3)
    plate = mk.block(geom, (1, 1, 1), name='plate')
    mk.block(geom, (1, 1, 1), (2, 0, 0), name='other')
    model.build(geom)

    def features():
        return [str(t) for t in geom.java.feature().tags()]
    before = features()
    reason = ('entity numbers change with the geometry, so select by '
              'location, e.g. ')
    with pytest.raises(TypeError) as error:
        mk.fillet(geom, [3, 5], 0.1)
    assert str(error.value) == (
        'Input "edge" of Fillet3D (the input of mk.fillet) takes geometry '
        'objects or one selection, not numbers such as [3, 5]: ' + reason +
        "mk.sel.box(geom, 'edge', ..., where='geometry').")
    with pytest.raises(TypeError, match=r'^Input "edge" of Chamfer3D \(the '
                       r'input of mk\.chamfer\) .* such as 3: '):
        mk.chamfer(geom, 3, 0.1)
    with pytest.raises(TypeError, match=r'such as \[3, 5\]: '):
        mk.fillet(geom, numpy.array([3, 5]), 0.1)
    with pytest.raises(TypeError) as error:
        mk.delete(geom, 4)
    assert str(error.value) == (
        'Input "input" of Delete takes geometry objects (their nodes or '
        'names) or one selection, not numbers such as 4: ' + reason +
        "mk.sel.box(geom, 'domain', ..., where='geometry').")
    with pytest.raises(TypeError) as error:
        mk.union(geom, [1, 2])
    assert str(error.value) == (
        'Input "input" of Union takes geometry objects (their nodes or '
        'names) or one selection, not numbers such as [1, 2].')
    with pytest.raises(TypeError, match=r'^Input "input2" of Difference .* '
                                        r'such as 2\.$'):
        mk.difference(geom, plate, [2])
    with pytest.raises(TypeError, match=r'such as 1\.$'):
        mk.feature(geom, 'Move', input=[1], displx=1)
    assert features() == before
    # objects in a numpy array, and the selection the message suggests
    mk.move(geom, numpy.array([plate]), (0, 0, 1))
    top = mk.sel.box(geom, 'edge', z=2, where='geometry')
    fillet = mk.fillet(geom, top, 0.1)
    model.build(geom)
    named = str(fillet.java.selection('edge').named())
    with pytest.raises(TypeError, match=r'such as 3: '):
        mk.set(fillet, edge=[3])
    assert str(fillet.java.selection('edge').named()) == named


def test_geometry_inputs_reject_numbers_2d(model):
    geom = mk.geometry(model, 2)
    mk.square(geom, 1)
    model.build(geom)
    with pytest.raises(TypeError, match=r"e\.g\. mk\.sel\.box\(geom, 'point', "
                                        r"\.\.\., where='geometry'\)\.$"):
        mk.fillet(geom, [3], 0.1)
    corner = mk.sel.box(geom, 'point', x=1, y=1, where='geometry')
    mk.fillet(geom, corner, 0.2)                 # as the message suggests
    model.build(geom)
    assert mk.measure(geom, 'domain') == pytest.approx(
        1 - (1 - math.pi/4)*0.2**2, rel=1e-4)    # the arc is approximated
    plane = mk.workplane(mk.geometry(model, 3))
    mk.square(plane, 1)
    with pytest.raises(TypeError, match=r"e\.g\. mk\.sel\.box\(plane, "
                                        r"'point', \.\.\.\)\.$"):
        mk.fillet(plane, [1], 0.1)
    with pytest.raises(TypeError, match=r"e\.g\. mk\.sel\.box\(plane, "
                                        r"'object', \.\.\.\)\.$"):
        mk.delete(plane, 1)
