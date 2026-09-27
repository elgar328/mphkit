"""Tests for `mk.set` on MPh nodes and on Java objects MPh does not reach."""
import mph
import numpy
import pytest

import mphkit as mk
from conftest import count
from mphkit._comsol import WorkPlaneNode


@pytest.fixture
def component(geom):
    """The Java component of a 3D geometry with one built block."""
    mk.block(geom, (10, 10, 10))
    geom.model.build(geom)
    return mk.component_of(geom).java


def test_mesh_features(component):
    mesh = component.mesh().create('mesh1')
    size = mesh.feature('size')
    # raw Java set() rejects a plain int: "Ambiguous overloads"
    assert mk.set(size, custom=True, hgrad=2, hcurve=0.5) is size
    assert [str(size.getString(p)) for p in ('custom', 'hgrad', 'hcurve')] \
        == ['on', '2', '0.5']
    layers = mesh.create('bl1', 'BndLayer').create('blp', 'BndLayerProp')
    mk.set(layers, blnlayers=5, inittype='blhmin')
    assert str(layers.getString('blnlayers')) == '5'


def test_study_material_and_physics(model, geom, component):
    step = model.java.study().create('std1').create('freq', 'Frequency')
    mk.set(step, plist=[100, 200, 'f0'])
    assert [str(v) for v in step.getStringArray('plist')] == \
        ['100', '200', 'f0']
    mk.set(step, plist=numpy.array([1.0, 2.5]))
    assert [str(v) for v in step.getStringArray('plist')] == ['1', '2.5']
    material = component.material().create('mat1', 'Common')
    function = material.propertyGroup('def').func().create('eta', 'Piecewise')
    mk.set(function, arg='T', pieces=[[200.0, 1600.0, '1+T']])
    assert [str(v) for v in function.getStringMatrix('pieces')[0]] == \
        ['200', '1600', '1+T']
    group = material.propertyGroup('def')
    mk.set(group, thermalconductivity=['0.026', '0', '0', '0', '0.026', '0',
                                       '0', '0', '0.026'], density=1.2)
    assert str(group.getString('density')) == '1.2'
    heat = (model/'physics').create('HeatTransfer', geom)
    shape = heat.java.prop('ShapeProperty')
    mk.set(shape, order_temperature=2)
    assert str(shape.getString('order_temperature')) == '2'
    with pytest.raises(ValueError, match="Did you mean 'order_temperature'"):
        mk.set(shape, order_temperatur=2)


def test_probe_and_variables(component):
    probe = component.probe().create('bnd1', 'Boundary')
    mk.set(probe, expr='T', unit='K', descractive=True)
    assert str(probe.getString('descractive')) == 'on'
    with pytest.raises(ValueError, match="Did you mean 'expr'"):
        mk.set(probe, exprr='T')
    variables = component.variable().create('var1')
    mk.set(variables, a='2*pi', n=2)
    assert [str(variables.get(v)) for v in ('a', 'n')] == ['2*pi', '2']


def test_node_mixed_list(model):
    model.java.study().create('std1').create('freq', 'Frequency')
    step = (model/'studies').children()[0].children()[0]
    with pytest.raises(TypeError):
        step.property('plist', [100, 'f0'])      # plain MPh fails here
    assert mk.set(step, plist=[100, 'f0']) is step
    assert [str(v) for v in step.java.getStringArray('plist')] == \
        ['100', 'f0']


def test_geometry_feature_inputs(model, geom):
    model.parameter('L', '10')
    a = mk.block(geom, ('L', 10, 10), name='a')
    b = mk.block(geom, (4, 4, 4), name='b')
    c = mk.block(geom, (4, 4, 4), (6, 6, 6), name='c')
    cut = mk.difference(geom, a, [b], keepsubtract=False)
    model.build(geom)
    assert count(geom, 'domains') == 2           # a minus b, and c
    mk.set(cut, input2=[c])
    mk.set(a, size=['L', 10, 12])
    model.build(geom)
    assert count(geom, 'domains') == 2           # b inside a minus c
    assert mk.measure(geom, 'domain') == pytest.approx(10*10*12 - 4**3)


def test_workplane_feature_inputs(model, geom):
    mk.block(geom, (10, 10, 10))
    plane = mk.workplane(geom, quickz=5, unite=True, name='wp')
    square = mk.square(plane, 4, name='sq')
    first = mk.circle(plane, 1, (1, 1), name='c1')
    second = mk.circle(plane, 1, (3, 3), name='c2')
    cut = mk.feature(plane, 'Difference', input=[square], input2=[first],
                     name='dif')
    mk.set(cut, input2=[second])
    assert [str(o) for o in cut.java.selection('input2').objects()] == \
        ['c2']
    plain = mph.Node(model, str(cut))
    assert plain.java is None                    # MPh cannot resolve it
    resolved = mk.set(plain, input2=[first])
    assert isinstance(resolved, WorkPlaneNode)
    assert [str(o) for o in resolved.java.selection('input2').objects()] == \
        ['c1']


def test_java_geometry_feature_input(geom):
    cylinder = mk.cylinder(geom, 1, 2)
    java = cylinder.java
    with pytest.raises(TypeError, match='input selection'):
        mk.set(java, r=5, input=['blk1'])
    assert str(java.getString('r')) == '1'       # nothing was set


def test_order_skip_and_errors(model, component):
    size = component.mesh().create('mesh1').feature('size')
    with pytest.raises(ValueError, match="Did you mean 'hmax'"):
        mk.set(size, hgrad=3, hmx=1)
    assert str(size.getString('hgrad')) == '3'   # earlier keys stay set
    mk.set(size, hgrad=None)
    assert str(size.getString('hgrad')) == '3'
    with pytest.raises(ValueError, match='accepts'):
        mk.set(size, hauto='huge')
    with pytest.raises(TypeError):
        mk.set(5, a=1)
    with pytest.raises(LookupError):
        mk.set(model/'meshes'/'missing', hmax=1)
    assert 'set' not in mk.__all__
