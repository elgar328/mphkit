"""Tests for the COMSOL/MPh internals."""
import pytest

import mphkit as mk
from mphkit import _comsol


def test_pick_label():
    assert _comsol.pick_label('a', 'Block 1', {'b'}) == 'a'
    with pytest.raises(ValueError):
        _comsol.pick_label('b', 'Block 1', {'b'})
    assert _comsol.pick_label(None, 'Block 1', set()) == 'Block 1'
    taken = {'Block 1', 'Block 1 (2)'}
    assert _comsol.pick_label(None, 'Block 1', taken) == 'Block 1 (3)'


def test_convert_pure():
    import numpy
    assert _comsol.convert((200, 200, 2.5)) == ['200', '200', '2.5']
    assert _comsol.convert([0, 'x', 1]) == ['0', 'x', '1']
    assert _comsol.convert(numpy.array([1.0, 2.5])) == ['1', '2.5']
    assert _comsol.convert(numpy.float64(2.5)) == 2.5
    assert _comsol.convert([[0, 1], [2, 'a']]) == [['0', '1'], ['2', 'a']]
    assert _comsol.convert([True, False]) == [True, False]
    assert _comsol.convert('r/2') == 'r/2'
    assert _comsol.convert(True) is True


def test_entity_dim(model):
    g3 = mk.geometry(model, 3)
    g2 = mk.geometry(model, 2)
    assert [_comsol.entity_dim(g3, e) for e in _comsol.ENTITIES] == [3, 2, 1, 0]
    assert [_comsol.entity_dim(g2, e) for e in _comsol.ENTITIES] == [2, 1, 1, 0]
    assert _comsol.entity_name(g2, 'edge') == 'boundary'
    plane = mk.workplane(g3, quickz=0)
    assert [_comsol.entity_dim(plane, e) for e in _comsol.ENTITIES] == \
        [2, 1, 1, 0]
    with pytest.raises(ValueError):
        _comsol.entity_dim(g3, 'face')


def test_component_of(model):
    g1 = mk.geometry(model, 3)
    g2 = mk.geometry(model, 2)
    assert str(_comsol.component_of(g1).tag()) != str(_comsol.component_of(g2).tag())
    assert g2.tag() in [str(t) for t in _comsol.component_of(g2).geom().tags()]


def test_geometry_of(geom):
    blk = mk.feature(geom, 'Block')
    assert _comsol.geometry_of(blk) == geom
    assert _comsol.geometry_of(geom) == geom
    with pytest.raises(TypeError):
        _comsol.geometry_of(geom.model/'selections')


def test_check_built_and_entity_count(model, geom):
    model.parameter('L', '10')
    mk.block(geom, ('L', 'L', 'L'))
    with pytest.raises(RuntimeError, match='not built'):
        _comsol.check_built(geom)
    model.build(geom)
    _comsol.check_built(geom)
    assert [_comsol.entity_count(geom, d) for d in (3, 2, 1, 0)] == \
        [1, 6, 12, 8]
    model.parameter('L', '12')
    with pytest.raises(RuntimeError):
        _comsol.check_built(geom)
    square = mk.geometry(model, 2)
    mk.square(square, 1)
    model.build(square)
    assert [_comsol.entity_count(square, d) for d in (2, 1, 0)] == [1, 4, 4]


def test_create_java_args(model, geom):
    container = _comsol.component_of(geom).coordSystem()
    tag, label = _comsol.create_java(container, 'coordinates', 'Rotated',
                                     'turned', set(),
                                     tags=model.java.coordSystem(),
                                     args=(geom.tag(),))
    assert str(container.get(tag).getType()) == 'Rotated'
    assert label == 'turned'


@pytest.mark.parametrize('entity, meant', [
    ('face', 'boundary'), ('Surfaces', 'boundary'), ('Boundary', 'boundary'),
    ('volume', 'domain'), ('vertices', 'point'), ('lines', 'edge'),
])
def test_entity_suggestion(geom, entity, meant):
    with pytest.raises(ValueError, match=f"Did you mean '{meant}'\\?"):
        mk.sel.box(geom, entity, z=0)


def test_entity_suggestion_2d(model, geom):
    # in 2D, and in a work plane, a face is a domain
    plane = mk.workplane(geom, quickz=0)
    flat = mk.geometry(model, 2, name='flat')
    for parent in (plane, flat):
        with pytest.raises(ValueError, match="Did you mean 'domain'\\?"):
            mk.sel.box(parent, 'face', x=0)


def test_entity_unknown_explains_names(geom):
    with pytest.raises(ValueError) as error:
        mk.sel.box(geom, 'xyz', z=0)
    assert 'Did you mean' not in str(error.value)
    assert "COMSOL's names: 'domain'" in str(error.value)
    # 'object' is suggested only by selections, with where='geometry'
    with pytest.raises(ValueError, match="Did you mean 'object'\\?") as error:
        mk.sel.box(geom, 'objects', x=0, where='geometry')
    assert 'needs' not in str(error.value)
    with pytest.raises(ValueError, match="needs where='geometry'"):
        mk.sel.box(geom, 'objects', x=0)
    with pytest.raises(ValueError) as error:
        mk.measure(geom, 'objects')
    assert 'Did you mean' not in str(error.value)


def test_entity_level_number(model, geom):
    # a level given as a number, as physics features take it
    with pytest.raises(ValueError, match="Did you mean 'boundary'\\?"):
        mk.sel.box(geom, 2, z=0)
    with pytest.raises(ValueError, match="Did you mean 'edge'\\?"):
        mk.sel.box(geom, 1, z=0)
    flat = mk.geometry(model, 2, name='flat')
    with pytest.raises(ValueError, match="Did you mean 'boundary'\\?"):
        mk.sel.box(flat, 1, x=0)
    for level in (5, True):
        with pytest.raises(ValueError) as error:
            mk.sel.box(geom, level, z=0)
        assert 'Did you mean' not in str(error.value)
