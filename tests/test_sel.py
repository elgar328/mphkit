"""Tests for geometry-based selections."""
import math

import pytest
from mph import Node

import mphkit as mk
from conftest import count
from mphkit._comsol import WorkPlaneNode


def entities(node, dim):
    return sorted(int(e) for e in node.java.entities(dim))


@pytest.fixture
def cube(model, geom):
    """A built 10 mm cube: faces x=0 → 1, z=0 → 3, x=10 → 6."""
    mk.block(geom, (10, 10, 10), name='cube')
    model.build(geom)
    return geom


def test_box_planes(cube):
    assert entities(mk.sel.box(cube, 'boundary', x=0), 2) == [1]
    assert entities(mk.sel.box(cube, 'boundary', x=10), 2) == [6]
    assert entities(mk.sel.box(cube, 'boundary', z=(0, 0)), 2) == [3]
    wide = mk.sel.box(cube, 'boundary', z=0, condition='intersects')
    assert entities(wide, 2) == [1, 2, 3, 5, 6]
    assert entities(mk.sel.box(cube, 'domain', x=(-1, 11)), 3) == [1]


def test_box_names(cube):
    node = mk.sel.box(cube, 'boundary', x=0, name='left')
    assert str(node) == 'selections/left'
    with pytest.raises(ValueError):
        mk.sel.box(cube, 'boundary', x=10, name='left')
    mk.sel.result(cube, cube/'cube', 'domain')           # derived: 'cube'
    with pytest.raises(ValueError):
        mk.sel.box(cube, 'boundary', x=10, name='cube')


def test_box_2d(model):
    geom = mk.geometry(model, 2)
    mk.square(geom, 1)
    model.build(geom)
    assert entities(mk.sel.box(geom, 'boundary', x=0), 1) == [1]
    with pytest.raises(ValueError):
        mk.sel.box(geom, 'boundary', z=0)


def test_ball_all_and_set_operations(cube):
    corner = mk.sel.ball(cube, 'point', (0, 0, 0), 0.1)
    assert entities(corner, 0) == [1]
    every = mk.sel.all(cube, 'boundary')
    assert entities(every, 2) == [1, 2, 3, 4, 5, 6]
    left = mk.sel.box(cube, 'boundary', x=0)
    right = mk.sel.box(cube, 'boundary', x=10)
    sides = mk.sel.union(cube, 'boundary', [left, right])
    assert entities(sides, 2) == [1, 6]
    rest = mk.sel.difference(cube, 'boundary', every, sides)
    assert entities(rest, 2) == [2, 3, 4, 5]
    assert entities(mk.sel.complement(cube, 'boundary', sides), 2) == \
        [2, 3, 4, 5]
    assert entities(mk.sel.intersection(cube, 'boundary', [every, left]),
                    2) == [1]


def test_adjacent(model, geom):
    mk.block(geom, (10, 10, 10), name='a')
    mk.block(geom, (10, 10, 10), (10, 0, 0), name='b')
    model.build(geom)
    a = mk.sel.box(geom, 'domain', x=(-1, 11))
    assert len(entities(mk.sel.adjacent(geom, a), 2)) == 6
    both = mk.sel.all(geom, 'domain')
    assert len(entities(mk.sel.adjacent(geom, both), 2)) == 10
    inner = mk.sel.adjacent(geom, both, exterior=False, interior=True)
    assert len(entities(inner, 2)) == 1


def test_result_selection(model, geom):
    blk = mk.block(geom, (10, 10, 10), name='blk')
    pt = mk.point(geom, (5, 5, 20), name='src')
    faces = mk.sel.result(geom, blk, 'boundary')
    assert str(faces) == 'selections/blk (boundary)'
    assert mk.sel.result(geom, blk, 'boundary') == faces
    dom = mk.sel.result(geom, blk, 'domain')
    source = mk.sel.result(geom, pt, 'point')
    model.build(geom)
    assert entities(faces, 2) == [1, 2, 3, 4, 5, 6]
    assert entities(dom, 3) == [1]
    assert len(entities(source, 0)) == 1
    assert blk.java.getString('selresultshow') == 'all'
    physics = (model/'physics').create('HeatTransfer', geom)
    boundary = physics.create('TemperatureBoundary', 2)
    boundary.select(faces)
    assert boundary.selection() == faces


def test_geometry_selection_drives_delete(model, geom):
    mk.block(geom, (10, 10, 10))
    mk.block(geom, (10, 10, 10), (20, 0, 0))
    left = mk.sel.box(geom, 'domain', x=(-1, 11), where='geometry',
                      name='left')
    assert str(left) == 'selections/left'
    assert left.tag() == f'{geom.tag()}_boxsel1'
    mk.feature(geom, 'Delete', input=left)
    model.build(geom)
    assert count(geom, 'domains') == 1


def test_geometry_selection_sees_only_earlier_objects(model, geom):
    mk.block(geom, (10, 10, 10))
    top = mk.sel.box(geom, 'boundary', z=10, where='geometry')
    mk.block(geom, (10, 10, 10), (20, 0, 0))
    model.build(geom)
    assert len(entities(top, 2)) == 1


def test_geometry_set_operation_rejects_component_selection(model, geom):
    mk.block(geom, (10, 10, 10))
    component = mk.sel.box(geom, 'boundary', x=0)
    with pytest.raises(TypeError):
        mk.sel.union(geom, 'boundary', [component], where='geometry')
    with pytest.raises(TypeError):
        mk.feature(geom, 'Delete', input=component)
    holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
    with pytest.raises(TypeError, match='cumulative'):
        mk.sel.union(geom, 'domain', [holes], where='geometry')


@pytest.fixture
def layered(model, geom):
    """A 40 mm block with a 5 mm layer on top only (layers default to the
    bottom as well, which is why `layerbottom` is switched off here)."""
    mk.block(geom, (40, 40, 40), name='free', layername=['pml'], layer=[5],
             layertop=True, layerbottom=False, sellayer=True)
    model.build(geom)
    return geom


def test_layer_selection(layered):
    free = layered/'free'
    pml = mk.sel.layer(layered, free, 'pml')
    core = mk.sel.layer(layered, free, 'core')
    # cross-check against the same regions selected by location
    assert entities(pml, 3) == entities(
        mk.sel.box(layered, 'domain', z=(35, 40)), 3)
    assert entities(core, 3) == entities(
        mk.sel.box(layered, 'domain', z=(0, 35)), 3)
    assert entities(pml, 3) != entities(core, 3)
    assert str(pml) == 'selections/free (pml)'
    assert mk.sel.layer(layered, free, 1) == pml
    assert mk.sel.layer(layered, free, 'pml') == pml


def test_layer_after_boolean(model, geom):
    outer = mk.block(geom, (40, 40, 40), name='free', layername=['pml'],
                     layer=[5], layertop=True, layerbottom=False,
                     sellayer=True)
    hole = mk.cylinder(geom, 5, 10, (20, 20, 0), name='hole')
    mk.difference(geom, outer, [hole], name='cut')
    model.build(geom)
    pml = mk.sel.layer(geom, outer, 'pml')
    assert entities(pml, 3) == entities(
        mk.sel.box(geom, 'domain', z=(35, 40)), 3)


def test_layer_errors(layered, geom):
    free = layered/'free'
    with pytest.raises(ValueError, match='no layer named'):
        mk.sel.layer(layered, free, 'nope')
    with pytest.raises(TypeError):
        mk.sel.layer(layered, free, 1.5)
    with pytest.raises(ValueError, match='layer selection'):
        mk.sel.layer(layered, free, 7)
    point = mk.point(layered, (0, 0, 50))
    with pytest.raises(ValueError, match='does not support layers'):
        mk.sel.layer(layered, point, 'pml')


#######################################
# Queries: entity numbers and lookups #
#######################################

def test_entities_levels(model, cube):
    assert mk.sel.entities(cube, mk.sel.box(cube, 'boundary', x=0)) == [1]
    domain = mk.sel.all(cube, 'domain')
    # an adjacent selection reports its input level in `entitydim`
    assert mk.sel.entities(cube, mk.sel.adjacent(cube, domain)) == \
        [1, 2, 3, 4, 5, 6]
    assert mk.sel.entities(cube, mk.sel.adjacent(cube, domain, 'point')) == \
        list(range(1, 9))
    explicit = (model/'selections').create('Explicit', name='three edges')
    explicit.java.geom(cube.tag(), 1)
    explicit.java.set([1, 2, 3])
    assert mk.sel.entities(cube, explicit) == [1, 2, 3]
    result = mk.sel.result(cube, cube/'cube', 'domain')
    assert mk.sel.entities(cube, result) == [1]
    # a selection in the geometry sequence changes it: build again
    bottom = mk.sel.box(cube, 'boundary', z=0, where='geometry',
                        name='bottom')
    with pytest.raises(RuntimeError, match='model.build'):
        mk.sel.entities(cube, bottom)
    model.build(cube)
    assert mk.sel.entities(cube, bottom) == [3]


def test_entities_2d(model):
    geom = mk.geometry(model, 2)
    mk.square(geom, 1)
    model.build(geom)
    assert mk.sel.entities(geom, mk.sel.box(geom, 'boundary', x=0)) == [1]
    assert mk.sel.entities(geom, mk.sel.all(geom, 'point')) == [1, 2, 3, 4]


def test_entities_errors(model, cube):
    other = mk.geometry(model, 3)
    mk.block(other, (1, 1, 1))
    model.build(other)
    with pytest.raises(ValueError, match='does not belong'):
        mk.sel.entities(cube, mk.sel.box(other, 'domain'))
    with pytest.raises(LookupError):
        mk.sel.entities(cube, model/'selections'/'missing')
    with pytest.raises(TypeError):
        mk.sel.entities(cube, cube/'cube')
    # the selections COMSOL derives from one feature share its label
    block = cube/'cube'
    block.property('selresult', True)
    block.property('selresultshow', 'all')
    model.build(cube)
    with pytest.raises(ValueError, match='ambiguous'):
        mk.sel.entities(cube, model/'selections'/'cube')


def test_entities_requires_build(model, geom):
    model.parameter('L', '10')
    mk.block(geom, ('L', 'L', 'L'), name='blk')
    faces = mk.sel.box(geom, 'boundary', x=0)
    with pytest.raises(RuntimeError, match='model.build'):
        mk.sel.entities(geom, faces)
    model.build(geom)
    assert mk.sel.entities(geom, faces) == [1]
    model.parameter('L', '12')
    with pytest.raises(RuntimeError, match='blk1'):
        mk.sel.entities(geom, faces)
    model.build(geom)
    mk.sel.result(geom, geom/'blk', 'domain')     # does not need a rebuild
    assert mk.sel.entities(geom, faces) == [1]


def test_find(model, geom):
    mk.block(geom, (10, 10, 10))
    mk.block(geom, (10, 10, 10), (20, 0, 0))

    def tags():
        return [str(t) for t in model.java.selection().tags()]

    before = tags()
    with pytest.raises(RuntimeError):
        mk.sel.find(geom, 'domain', x=(15, 35))
    assert tags() == before
    model.build(geom)
    assert mk.sel.find(geom, 'domain', x=(15, 35)) == [2]
    assert mk.sel.find(geom, 'boundary', x=0) == [1]
    assert tags() == before


def odd_block(model, unit='m', pos=(1.1, 0.7, 0.3)):
    """A built block at coordinates single precision cannot represent."""
    geom = mk.geometry(model, 3, length_unit=unit)
    mk.block(geom, (13, 2, 3), pos)
    model.build(geom)
    return geom


@pytest.mark.parametrize('unit', ['m', 'mm'])
def test_box_non_round_coordinates(model, unit):
    geom = odd_block(model, unit)
    assert len(mk.sel.find(geom, 'boundary', x=1.1)) == 1
    assert len(mk.sel.find(geom, 'boundary', z=0.3)) == 1
    assert len(mk.sel.find(geom, 'boundary', x=14.1)) == 1
    assert len(mk.sel.find(geom, 'edge', x=1.1, z=0.3)) == 1
    assert mk.sel.find(geom, 'domain', x=(1.1, 14.1), y=(0.7, 2.7),
                       z=(0.3, 3.3)) == [1]
    assert len(mk.sel.find(geom, 'point', x=1.1, y=0.7, z=0.3)) == 1


def test_box_non_round_negative(model):
    geom = odd_block(model, pos=(-14.1, -2.7, -3.3))
    assert len(mk.sel.find(geom, 'boundary', x=-1.1)) == 1
    assert len(mk.sel.find(geom, 'boundary', z=-0.3)) == 1


def test_box_non_round_2d(model):
    flat = mk.geometry(model, 2)
    mk.rectangle(flat, (13, 2), (1.1, 0.7))
    model.build(flat)
    assert len(mk.sel.find(flat, 'boundary', x=1.1)) == 1
    assert len(mk.sel.find(flat, 'boundary', y=0.7)) == 1


def test_box_non_round_in_geometry_sequence(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (13, 2, 3), (1.1, 0.7, 0.3))
    face = mk.sel.box(geom, 'boundary', x=1.1, where='geometry')
    model.build(geom)
    assert len(entities(face, 2)) == 1
    other = mk.geometry(model, 3)
    mk.block(other, (13, 2, 3), (1.1, 0.7, 0.3))
    mk.block(other, (1, 1, 1), (20, 0, 0))
    odd = mk.sel.box(other, 'object', x=(1.1, 14.1), y=(0.7, 2.7),
                     z=(0.3, 3.3), where='geometry')
    mk.delete(other, odd)
    model.build(other)
    assert count(other, 'domains') == 1


def test_box_expression_bounds(model):
    model.parameter('L', '0.3')
    geom = odd_block(model)
    assert len(mk.sel.find(geom, 'boundary', z='L')) == 1
    top = mk.sel.box(geom, 'boundary', z='L')
    assert top.java.getString('zmin') == '(L)-1e-6*abs(L)'
    assert top.java.getString('zmax') == '(L)+1e-6*abs(L)'
    # COMSOL's own bounds are set as given, without a margin
    exact = mk.sel.box(geom, 'boundary', zmin=0.3, zmax=0.3)
    assert exact.java.getString('zmin') == exact.java.getString('zmax') \
        == '0.3'


def test_box_open_and_infinite_bounds(model):
    geom = odd_block(model)
    face = mk.sel.find(geom, 'boundary', x=1.1)
    assert len(face) == 1
    assert mk.sel.find(geom, 'boundary', x=(None, 1.1)) == face
    assert mk.sel.find(geom, 'boundary', x=math.inf) == []
    assert len(mk.sel.find(geom, 'boundary', x=(-math.inf, math.inf))) == 6
    assert len(mk.sel.find(geom, 'boundary', x=(0, 10**400))) == 6


def test_box_zero_after_rotation(model, geom):
    # Guard: zero gets no margin, as COMSOL rounds coordinates near it to 0.
    blk = mk.block(geom, (2, 2, 2))
    mk.rotate(geom, blk, 90)
    model.build(geom)
    assert len(mk.sel.find(geom, 'boundary', x=0)) == 1


def test_bounding_box_round_trip(model, geom):
    # Regression guard for the box/bounding_box contract; it passed before
    # the margin too.
    mk.block(geom, (13, 2, 3), (1.1, 0.7, 0.3))
    mk.sphere(geom, 0.7, (20.3, 1.1, 1.3))
    model.build(geom)
    for entity in ('domain', 'boundary', 'edge', 'point'):
        for n in mk.sel.find(geom, entity):
            box = mk.bounding_box(geom, entity, n)
            assert n in mk.sel.find(geom, entity, **box)


def test_cylinder(model, geom):
    mk.block(geom, (10, 10, 4), name='low')
    shaft = mk.cylinder(geom, 2, 10, (5, 5, 0), name='shaft')
    model.build(geom)
    shaft_domains = mk.sel.entities(geom, mk.sel.result(geom, shaft,
                                                        'domain'))
    inside = mk.sel.cylinder(geom, 'domain', (5, 5, 0), 2.5)
    assert mk.sel.entities(geom, inside) == shaft_domains
    # top and bottom are measured from pos along the axis
    upper = mk.sel.cylinder(geom, 'domain', (5, 5, 4), 2.5, top=6, bottom=0)
    assert mk.sel.entities(geom, upper) == \
        mk.sel.find(geom, 'domain', x=(2, 8), y=(2, 8), z=(4, 10))
    # a shell picks the side faces only
    sides = set(mk.sel.entities(geom, mk.sel.result(geom, shaft,
                                                    'boundary')))
    sides -= set(mk.sel.find(geom, 'boundary', z=0))
    sides -= set(mk.sel.find(geom, 'boundary', z=4))
    sides -= set(mk.sel.find(geom, 'boundary', z=10))
    shell = mk.sel.cylinder(geom, 'boundary', (5, 5, 0), 2.1, rin=1.9)
    assert mk.sel.entities(geom, shell) == sorted(sides)
    staged = mk.sel.cylinder(geom, 'domain', (5, 5, 0), 2.5,
                             where='geometry')
    model.build(geom)
    assert mk.sel.entities(geom, staged) == shaft_domains
    flat = mk.geometry(model, 2)
    with pytest.raises(ValueError, match='sel.disk'):
        mk.sel.cylinder(flat, 'domain', (0, 0), 1)


def test_cylinder_axis(model, geom):
    rod = mk.cylinder(geom, 1, 10, (0, 5, 5), axis=(1, 0, 0))
    mk.block(geom, (10, 10, 10))
    model.build(geom)
    expected = mk.sel.entities(geom, mk.sel.result(geom, rod, 'domain'))
    assert len(expected) == 1
    for axis in ((1, 0, 0), 'x'):
        along = mk.sel.cylinder(geom, 'domain', (0, 5, 5), 1.5, axis=axis)
        assert mk.sel.entities(geom, along) == expected


def test_disk(model, geom):
    flat = mk.geometry(model, 2)
    mk.square(flat, 10)
    spot = mk.circle(flat, 2, (5, 5))
    model.build(flat)
    inside = mk.sel.disk(flat, 'domain', (5, 5), 2.5)
    assert mk.sel.entities(flat, inside) == \
        mk.sel.entities(flat, mk.sel.result(flat, spot, 'domain'))
    with pytest.raises(ValueError, match='sel.cylinder'):
        mk.sel.disk(geom, 'domain', (0, 0), 1)


def test_cumulative(model, geom):
    holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
    assert str(holes) == 'selections/holes (domain)'
    a = mk.block(geom, (1, 1, 1), contributeto=holes)
    mk.block(geom, (1, 1, 1), (3, 0, 0), contributeto='holes')
    c = mk.block(geom, (1, 1, 1), (6, 0, 0))
    model.build(geom)
    assert mk.sel.entities(geom, holes) == [1, 2]
    walls = mk.sel.cumulative(geom, 'holes', 'boundary')
    assert len(mk.sel.entities(geom, walls)) == 12
    assert mk.sel.cumulative(geom, 'holes', 'domain') == holes
    assert mk.sel.cumulative(geom, holes, 'domain') == holes
    # contributions survive later Boolean operations
    mk.union(geom, [a, c])
    model.build(geom)
    assert mk.sel.entities(geom, holes) == \
        mk.sel.find(geom, 'domain', x=(-1, 4.5))
    # physics can use it
    physics = (model/'physics').create('HeatTransfer', geom)
    boundary = physics.create('TemperatureBoundary', 2)
    boundary.select(walls)
    assert boundary.selection() == walls


def test_cumulative_errors(model, geom):
    holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
    block = mk.block(geom, (1, 1, 1))
    with pytest.raises(ValueError, match='already exists'):
        mk.sel.cumulative(geom, 'holes', 'boundary', create=True)
    with pytest.raises(ValueError, match="Known: \\['holes'\\]"):
        mk.sel.cumulative(geom, 'hoels', 'boundary')
    with pytest.raises(ValueError, match='no cumulative selection'):
        mk.block(geom, (1, 1, 1), contributeto='hoels')
    with pytest.raises(TypeError):
        mk.sel.cumulative(geom, holes, 'domain', create=True)
    with pytest.raises(ValueError, match='"none"'):
        mk.sel.cumulative(geom, 'none', 'domain', create=True)
    mk.sel.box(geom, 'domain', name='taken')
    with pytest.raises(ValueError, match='already named'):
        mk.sel.cumulative(geom, 'taken', 'domain', create=True)
    with pytest.raises(ValueError, match='already in use'):
        mk.sel.cumulative(geom, 'rods', 'domain', create=True, name='taken')
    rods = mk.sel.cumulative(geom, 'rods', 'domain', create=True)  # retry
    assert str(rods) == 'selections/rods (domain)'
    with pytest.raises(ValueError, match='not a cumulative selection'):
        mk.set(block, contributeto=mk.sel.result(geom, block, 'domain'))
    with pytest.raises(ValueError, match='not a cumulative selection'):
        mk.block(geom, (1, 1, 1), contributeto=geom)
    other = mk.geometry(model, 3)
    theirs = mk.sel.cumulative(other, 'theirs', 'domain', create=True)
    with pytest.raises(ValueError, match='not a cumulative selection'):
        mk.block(geom, (1, 1, 1), contributeto=theirs)


def test_cumulative_set_and_workplane(model, geom):
    holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
    block = mk.block(geom, (1, 1, 1))
    mk.set(block, contributeto=holes)
    model.build(geom)
    assert mk.sel.entities(geom, holes) == [1]
    plane = mk.workplane(geom, quickz=5)
    with pytest.raises(ValueError, match='work plane'):
        mk.square(plane, 1, contributeto=holes)
    mk.square(plane, 1, contributeto='none')


def test_cumulative_2d(model):
    flat = mk.geometry(model, 2)
    dots = mk.sel.cumulative(flat, 'dots', 'edge', create=True)
    assert str(dots) == 'selections/dots (boundary)'
    mk.circle(flat, 1, contributeto=dots)
    mk.square(flat, 1, (5, 0))
    model.build(flat)
    assert len(mk.sel.entities(flat, dots)) == 4


def test_object_level(model, geom):
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (3, 0, 0))
    mk.block(geom, (1, 1, 1), (6, 0, 0))
    far = mk.sel.box(geom, 'object', x=(2.5, 4.5), where='geometry',
                     name='far')
    assert str(far) == f'{geom}/far'             # the feature, not selections/
    mk.delete(geom, far)
    model.build(geom)
    assert sorted(str(o) for o in geom.java.objectNames()) == \
        ['blk1', 'blk3']


def test_object_level_set_operation_and_ball(model, geom):
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (3, 0, 0))
    mk.block(geom, (1, 1, 1), (6, 0, 0))
    a = mk.sel.box(geom, 'object', x=(-0.5, 1.5), where='geometry')
    b = mk.sel.ball(geom, 'object', (6.5, 0.5, 0.5), 1, where='geometry')
    both = mk.sel.union(geom, 'object', [a, b], where='geometry')
    mk.delete(geom, both)
    model.build(geom)
    assert [str(o) for o in geom.java.objectNames()] == ['blk2']


def test_object_level_errors(model, geom):
    mk.block(geom, (1, 1, 1))
    with pytest.raises(ValueError, match="where='geometry'"):
        mk.sel.box(geom, 'object')
    with pytest.raises(ValueError):
        mk.sel.adjacent(geom, mk.sel.all(geom, 'domain'), 'object')
    objects = mk.sel.all(geom, 'object', where='geometry')
    model.build(geom)
    with pytest.raises(TypeError):
        mk.sel.entities(geom, objects)


def plane_square(model):
    """A work plane with a unit square, in a new 3D geometry."""
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom, quickz=0, name='plane')
    mk.square(plane, 1, name='square')
    return geom, plane


def rounded(corners):
    """Volume of a unit square prism with `corners` corners rounded by 0.3."""
    return 1 - corners*(4 - math.pi)*0.09/4


def extruded_volume(model, geom, plane):
    mk.extrude(geom, plane, 1)
    model.build(geom)
    return mk.measure(geom, 'domain')


def test_workplane_fillet_one_corner(model):
    geom, plane = plane_square(model)
    corner = mk.sel.box(plane, 'point', x=1, y=1)
    assert isinstance(corner, WorkPlaneNode)
    assert corner.parent() == plane
    mk.fillet(plane, corner, 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(1),
                                                                rel=0.005)


def test_workplane_fillet_adjacent(model):
    geom, plane = plane_square(model)
    top = mk.sel.box(plane, 'edge', y=1)
    ends = mk.sel.adjacent(plane, top, 'point', input_entity='edge')
    mk.fillet(plane, ends, 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(2),
                                                                rel=0.005)


def test_workplane_set_operations(model):
    geom, plane = plane_square(model)
    every = mk.sel.all(plane, 'point')
    origin = mk.sel.box(plane, 'point', x=0, y=0, where='geometry')
    mk.fillet(plane, mk.sel.difference(plane, 'point', every, origin), 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(3),
                                                                rel=0.005)
    geom, plane = plane_square(model)
    left = mk.sel.box(plane, 'point', x=0)
    low = mk.sel.box(plane, 'point', y=0)
    mk.fillet(plane, mk.sel.intersection(plane, 'point', [left, low]), 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(1),
                                                                rel=0.005)
    geom, plane = plane_square(model)
    right = mk.sel.box(plane, 'point', x=1)
    mk.fillet(plane, mk.sel.complement(plane, 'point', right), 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(2),
                                                                rel=0.005)
    geom, plane = plane_square(model)
    left = mk.sel.box(plane, 'point', x=0)
    low = mk.sel.box(plane, 'point', y=0)
    mk.fillet(plane, mk.sel.union(plane, 'point', [left, low]), 0.3)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(3),
                                                                rel=0.005)


def test_workplane_ball_disk_and_chamfer(model):
    for make in (lambda p: mk.sel.ball(p, 'point', (1, 1), 0.1),
                 lambda p: mk.sel.disk(p, 'point', (1, 1), 0.1)):
        geom, plane = plane_square(model)
        mk.fillet(plane, make(plane), 0.3)
        assert extruded_volume(model, geom, plane) == \
            pytest.approx(rounded(1), rel=0.005)
    geom, plane = plane_square(model)
    mk.chamfer(plane, mk.sel.box(plane, 'point', x=1, y=1), 0.2)
    assert extruded_volume(model, geom, plane) == pytest.approx(1 - 0.02)


def test_workplane_object_level_delete(model):
    geom, plane = plane_square(model)
    mk.circle(plane, 0.2, (3, 0))
    far = mk.sel.box(plane, 'object', x=(2, 4))
    mk.delete(plane, far)
    mk.extrude(geom, plane, 1)
    model.build(geom)
    assert count(geom, 'domains') == 1


def test_workplane_selection_switch(model):
    geom, plane = plane_square(model)
    one = mk.sel.box(plane, 'point', x=1, y=1)
    two = mk.sel.box(plane, 'point', y=1, name='top points')
    fillet = mk.fillet(plane, one, 0.3)
    mk.set(fillet, point=two)
    assert extruded_volume(model, geom, plane) == pytest.approx(rounded(2),
                                                                rel=0.005)
    plain = Node(model, str(one))                # a path, not a returned node
    mk.set(fillet, point=plain)
    model.build(geom)
    assert mk.measure(geom, 'domain') == pytest.approx(rounded(1), rel=0.005)


def test_workplane_selection_errors(model):
    geom, plane = plane_square(model)
    corner = mk.sel.box(plane, 'point', x=1, y=1)
    with pytest.raises(ValueError, match='no component selections'):
        mk.sel.box(plane, 'point', where='component')
    mk.block(geom, (1, 1, 1), (5, 0, 0))
    with pytest.raises(ValueError, match='belongs to the work plane'):
        mk.fillet(geom, corner, 0.1)
    with pytest.raises(ValueError, match='belongs to the work plane'):
        mk.sel.union(geom, 'point', [corner], where='geometry')
    solid = mk.sel.box(geom, 'point', where='geometry')
    with pytest.raises(ValueError, match='not to the work plane'):
        mk.fillet(plane, solid, 0.1)
    objects = mk.sel.box(geom, 'object', where='geometry')
    with pytest.raises(TypeError, match="use where='geometry'"):
        mk.sel.union(geom, 'point', [objects])      # a feature, not selections/
    other, other_plane = plane_square(model)
    theirs = mk.sel.box(other_plane, 'point', x=1, y=1)
    with pytest.raises(ValueError, match='not to the work plane'):
        mk.fillet(plane, theirs, 0.1)
    features = [str(t) for t in plane.java.geom().feature().tags()]
    for call in (lambda: mk.sel.cylinder(plane, 'point', (0, 0, 0), 1),
                 lambda: mk.sel.cumulative(plane, 'g', 'point', create=True),
                 lambda: mk.sel.find(plane, 'point', x=1),
                 lambda: mk.sel.entities(plane, corner),
                 lambda: mk.measure(plane, 'domain'),
                 lambda: mk.measure(plane, 'point'),
                 lambda: mk.bounding_box(plane, 'domain')):
        with pytest.raises(TypeError, match='work plane'):
            call()
    assert [str(t) for t in plane.java.geom().feature().tags()] == features
