"""
Tests of the results helpers: integral, average, maximum, minimum, value.

Most tests read one solved plate, 0.1 x 0.05 x 0.01 m held at 100 degC at
x = 0 and 20 degC at x = 0.1, so T = 100 - 800 x degC exactly (steel,
k = 45 W/(m*K): 18 W flow through it). Tests that change a model build
their own.
"""
import math

import numpy
import pytest

import mphkit as mk
from conftest import java_export

HELPERS = ('integral', 'average', 'maximum', 'minimum', 'value')


def leftovers(model):
    """Tags of what the helpers could leave in a model."""
    results = model.java.result()
    return [sorted(str(t) for t in container.tags())
            for container in (results.numerical(), results.table(),
                              results.dataset())]


@pytest.fixture(autouse=True)
def leaves_nothing(monkeypatch):
    """Checks after every call, also one that raises, that nothing is left."""
    for name in HELPERS:
        def checked(geom, *args, _helper=getattr(mk, name), **kwargs):
            before = leftovers(geom.model)
            try:
                return _helper(geom, *args, **kwargs)
            finally:
                assert leftovers(geom.model) == before
        monkeypatch.setattr(mk, name, checked)


def plate(client, name='plate', solve=True, study=True):
    """The linear plate; returns the model, geometry and face selections."""
    model = client.create(name)
    geom = mk.geometry(model, 3)
    mk.block(geom, (0.1, 0.05, 0.01))
    model.build(geom)
    faces = {face: mk.sel.box(geom, 'boundary', name=face, **where)
             for face, where in (('hot', {'x': 0}), ('cold', {'x': 0.1}),
                                 ('top', {'z': 0.01}))}
    steel = (model/'materials').create('Common', name='steel')
    for key, value in (('thermalconductivity', '45'), ('density', '7850'),
                       ('heatcapacity', '475')):
        (steel/'Basic').property(key, [value])
    model.parameter('Th', '100[degC]')
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    for face, temperature in (('hot', 'Th'), ('cold', '20[degC]')):
        boundary = heat.create('TemperatureBoundary', 2, name=face)
        boundary.select(faces[face])
        boundary.property('T0', temperature)
    (model/'meshes').create(geom, name='mesh')
    if study:
        (model/'studies').create(name='static').create('Stationary')
    if solve and study:
        model.solve()
    return model, geom, faces


@pytest.fixture(scope='module')
def solved(client):
    model, geom, faces = plate(client, 'solved plate')
    yield geom, faces
    client.remove(model)


@pytest.fixture
def fresh(client):
    """Builds own plates (by calling it) and removes them afterwards."""
    made = []

    def build(**options):
        model, geom, faces = plate(client, f'plate {len(made) + 1}',
                                   **options)
        made.append(model)
        return model, geom, faces
    yield build
    for model in made:
        client.remove(model)


#################
# Over entities #
#################

def test_values(solved):
    geom, faces = solved
    assert mk.average(geom, 'domain', 'T', unit='degC') == \
        pytest.approx(60)
    assert mk.average(geom, 'boundary', 'T', faces['top'],
                      unit='degC') == pytest.approx(60)
    assert mk.average(geom, 'boundary', 'T', faces['hot'],
                      unit='degC') == pytest.approx(100)
    assert mk.maximum(geom, 'domain', 'T', unit='degC') == \
        pytest.approx(100)
    assert mk.minimum(geom, 'domain', 'T', unit='degC') == \
        pytest.approx(20)
    assert mk.integral(geom, 'boundary', '1', faces['top']) == \
        pytest.approx(0.005)
    heat = mk.integral(geom, 'boundary', 'ht.ntflux', faces['hot'],
                       unit='W')
    assert isinstance(heat, float)
    assert heat == pytest.approx(-18)


def test_selection_forms(solved):
    geom, faces = solved
    hot = mk.sel.entities(geom, faces['hot'])
    for selection in (hot, hot[0], faces['hot']):
        assert mk.average(geom, 'boundary', 'T', selection) == \
            pytest.approx(373.15)
    assert mk.integral(geom, 'domain', '1') == pytest.approx(5e-5)
    assert mk.integral(geom, 'edge', '1') == pytest.approx(4*0.16)


def test_position(solved):
    geom, faces = solved
    value, where = mk.maximum(geom, 'domain', 'T', position=True)
    assert value == pytest.approx(373.15)
    assert where.shape == (3,)
    assert where[0] == pytest.approx(0)
    value, where = mk.minimum(geom, 'domain', 'T', unit='degC',
                              position=True)
    assert value == pytest.approx(20)
    assert where[0] == pytest.approx(0.1)


@pytest.mark.parametrize('expr, unit, expected', [
    ('T', 'degC', 60), ('T', 'K', 333.15), ('T', 'mK', 333150),
    ('T', 'degF', 140), ('Tx', 'K/mm', -0.8), ('1', 'mm^3', 5e4),
    ('ht.ntflux', 'W', -18), ('ht.ntflux', 'mW', -18000),
    ('ht.ntflux', 'W/m^2', -36000),
    ('45[W/(m*K)]', 'W/(m*K)', 45)])
def test_units(solved, expr, unit, expected):
    geom, faces = solved
    if unit == 'W/m^2':
        found = mk.average(geom, 'boundary', expr, faces['hot'], unit=unit)
    elif expr == 'ht.ntflux':
        found = mk.integral(geom, 'boundary', expr, faces['hot'], unit=unit)
    elif expr == '1':
        found = mk.integral(geom, 'domain', expr, unit=unit)
    else:
        found = mk.average(geom, 'domain', expr, unit=unit)
    assert found == pytest.approx(expected)


@pytest.mark.parametrize('call, unit', [
    (lambda geom, unit: mk.average(geom, 'domain', 'T', unit=unit), 'kg'),
    (lambda geom, unit: mk.integral(geom, 'domain', 'T', unit=unit), 'W'),
    (lambda geom, unit: mk.maximum(geom, 'domain', 'T', unit=unit,
                                   position=True), 'kg'),
    (lambda geom, unit: mk.value(geom, 'T', (0.05, 0.025, 0.005),
                                 unit=unit), 'kg')])
def test_wrong_unit(solved, call, unit):
    geom, faces = solved
    with pytest.raises(ValueError, match=rf'COMSOL evaluated "T" in .*K, '
                                         rf"not '{unit}'"):
        call(geom, unit)


def test_errors(solved):
    geom, faces = solved
    with pytest.raises(RuntimeError, match='Undefined variable'):
        mk.average(geom, 'domain', 'nothing')
    nowhere = mk.sel.box(geom, 'domain', z=5, name='nowhere')
    with pytest.raises(ValueError, match='Selection "selections/nowhere" '
                                         'is empty'):
        mk.average(geom, 'domain', 'T', nowhere)
    with pytest.raises(ValueError, match='The selection is empty'):
        mk.average(geom, 'domain', 'T', [])
    with pytest.raises(ValueError, match='use mk.value'):
        mk.average(geom, 'point', 'T')
    with pytest.raises(ValueError, match=r"'boundary' is an entity kind; "
                       r'the order is mk.integral\(geom, entity, expr'):
        mk.integral(geom, 'T', 'boundary')
    with pytest.raises(TypeError, match='the order is'):
        mk.average(geom, 'boundary', faces['hot'], 'T')
    with pytest.raises(TypeError, match='one expression at a time'):
        mk.average(geom, 'domain', ['T', 'T^2'])
    with pytest.raises(ValueError, match='Entity must be one of'):
        mk.average(geom, 'face', 'T')
    with pytest.raises(ValueError, match='has no property "intordr".*'
                                         "'intorder'"):
        mk.integral(geom, 'domain', 'T', intordr=4)
    assert mk.integral(geom, 'domain', 'T', intorder=4) == \
        pytest.approx(333.15*5e-5)


@pytest.mark.parametrize('key, hint', [
    ('expr', 'expression'), ('data', 'dataset='),
    ('solnum', 'step='), ('innerinput', 'step='), ('t', 'step='),
    ('outersolnum', 'outer loop'), ('dataseries', "step='all'"),
    ('includepos', 'position='), ('table', 'temporary')])
def test_reserved(solved, key, hint):
    geom, faces = solved
    with pytest.raises(ValueError, match=rf'sets "{key}" itself; .*{hint}'):
        mk.average(geom, 'domain', 'T', **{key: '1'})


def test_position_only_extremes(solved):
    geom, faces = solved
    with pytest.raises(ValueError, match='mk.maximum and mk.minimum do'):
        mk.average(geom, 'domain', 'T', position=True)


@pytest.mark.parametrize('step', [0, -1, True, 1.0, 'end', []])
def test_bad_steps(solved, step):
    geom, faces = solved
    with pytest.raises((ValueError, TypeError), match='step'):
        mk.average(geom, 'domain', 'T', step=step)


def test_steps_of_a_stationary_study(solved):
    geom, faces = solved
    for step in (None, 1, 'first', 'last', numpy.int64(1)):
        assert mk.average(geom, 'domain', 'T', unit='degC', step=step) == \
            pytest.approx(60)
    for step in ('all', [1], numpy.array([1])):
        found = mk.average(geom, 'domain', 'T', unit='degC', step=step)
        assert isinstance(found, numpy.ndarray)
        assert found == pytest.approx([60])
    with pytest.raises(ValueError, match='has 1 step, not 2'):
        mk.average(geom, 'domain', 'T', step=2)


##########
# Points #
##########

def test_value(solved):
    geom, faces = solved
    found = mk.value(geom, 'T', (0.05, 0.025, 0.005), unit='degC')
    assert isinstance(found, float)
    assert found == pytest.approx(60)
    found = mk.value(geom, 'T', [(0, 0, 0), (0.025, 0.01, 0.01)],
                     unit='degC')
    assert found.shape == (2,)
    assert found == pytest.approx([100, 80])
    found = mk.value(geom, 'T', numpy.array([[0.1, 0.05, 0]]))
    assert found == pytest.approx([293.15])


def test_outside(solved):
    geom, faces = solved
    points = [(0.05, 0.025, 0.005), (0.2, 0, 0), (0.05, 0.025, 0.005),
              (0, 0, -1)]
    with pytest.raises(ValueError, match=r'Points 2 and 4 of 4 are outside '
                                         r"the geometry.*outside='nan'"):
        mk.value(geom, 'T', points)
    found = mk.value(geom, 'T', points, unit='degC', outside='nan')
    assert found[[0, 2]] == pytest.approx([60, 60])
    assert numpy.isnan(found[[1, 3]]).all()
    with pytest.raises(ValueError, match='No point has a value to check'):
        mk.value(geom, 'T', (1, 1, 1), unit='degC', outside='nan')
    assert math.isnan(mk.value(geom, 'T', (1, 1, 1), outside='nan'))
    # a nan expression inside is not taken for outside
    assert math.isnan(mk.value(geom, '0/0', (0.05, 0.025, 0.005)))


def test_value_errors(solved):
    geom, faces = solved
    with pytest.raises(ValueError, match=r'shape \(3, 2\), which looks '
                                         'transposed if you meant 2 points'):
        mk.value(geom, 'T', [[0, 0.1], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match=r'rows of them \(shape \(n, 3\)\)'):
        mk.value(geom, 'T', (0, 0))
    with pytest.raises(TypeError, match='points must be numbers'):
        mk.value(geom, 'T', 'x')
    with pytest.raises(RuntimeError, match='value\\(\\) evaluates domain '
                                           'variables; if "ht.ntflux" '
                                           'exists on boundaries only'):
        mk.value(geom, 'ht.ntflux', (0, 0.025, 0.005))
    with pytest.raises(ValueError, match="outside must be 'error' or 'nan'"):
        mk.value(geom, 'T', (0, 0, 0), outside='zero')
    with pytest.raises(ValueError, match=r'the order is mk.value'):
        mk.value(geom, 'domain', (0, 0, 0))
    with pytest.raises(TypeError, match=r'the order is mk.value'):
        mk.value(geom, (0, 0, 0), 'T')


##########################
# Datasets and solutions #
##########################

def test_unsolved(fresh):
    model, geom, faces = fresh(solve=False)
    with pytest.raises(RuntimeError, match=r'No solved dataset; run '
                                           r'model.solve\(\) first'):
        mk.average(geom, 'domain', 'T')
    model.solve()
    assert mk.average(geom, 'domain', 'T', unit='degC') == pytest.approx(60)
    model.clear()
    with pytest.raises(RuntimeError, match='No solved dataset'):
        mk.value(geom, 'T', (0, 0, 0))


def test_several_studies(fresh):
    model, geom, faces = fresh()
    (model/'studies').create(name='second').create('Stationary')
    # an unsolved study adds no dataset
    assert mk.average(geom, 'domain', 'T', unit='degC') == pytest.approx(60)
    model.parameter('Th', '200[degC]')
    model.solve('second')
    with pytest.raises(ValueError, match=r'several solved datasets: '
                       r'"static//Solution 1" \(dset1\), "second//Solution '
                       r'2" \(dset2\); pass dataset='):
        mk.average(geom, 'domain', 'T')
    for dataset, expected in (('static//Solution 1', 60), ('dset2', 110),
                              (model/'datasets'/'second//Solution 2', 110)):
        assert mk.average(geom, 'domain', 'T', unit='degC',
                          dataset=dataset) == pytest.approx(expected)
        assert mk.value(geom, 'T', (0.05, 0, 0), unit='degC',
                        dataset=dataset) == pytest.approx(expected)
    with pytest.raises(LookupError, match='No dataset "Solution 9"'):
        mk.average(geom, 'domain', 'T', dataset='Solution 9')


def test_several_components(fresh):
    model, geom, faces = fresh()
    other = mk.geometry(model, 3)
    mk.block(other, (1, 1, 1))
    model.build(other)
    with pytest.raises(RuntimeError, match='No solved dataset for geometry'):
        mk.average(other, 'domain', 'T')
    with pytest.raises(ValueError, match=r'does not belong to geometry '
                                         r'".*" \(geom2\), but to geom1'):
        mk.average(other, 'domain', '1', dataset='dset1')
    assert mk.average(geom, 'domain', 'T', unit='degC') == pytest.approx(60)


def test_stale_solution(fresh):
    model, geom, faces = fresh()
    block = geom/'Block 1'
    block.property('size', ['0.2', '0.05', '0.01'])
    with pytest.raises(RuntimeError, match='changed or its mesh was cleared'):
        mk.average(geom, 'domain', 'T')
    model.build(geom)
    with pytest.raises(RuntimeError, match=r'run model.build\(geom\), '
                                           r'model.mesh\(\) and '
                                           r'model.solve\(\)'):
        mk.value(geom, 'T', (0, 0, 0))
    model.mesh()
    model.solve()
    assert mk.integral(geom, 'domain', '1') == pytest.approx(1e-4)
    model.build(geom)       # unchanged: the solution still fits
    assert mk.integral(geom, 'domain', '1') == pytest.approx(1e-4)
    model.java.component('comp1').mesh('mesh1').clearMesh()
    with pytest.raises(RuntimeError, match='mesh was cleared'):
        mk.average(geom, 'domain', 'T')


def test_saved_and_loaded(fresh, client, tmp_path):
    model, geom, faces = fresh()
    path = tmp_path/'plate.mph'
    model.save(path)
    loaded = client.load(path)
    try:
        geom = (loaded/'geometries').children()[0]
        assert mk.average(geom, 'domain', 'T', unit='degC') == \
            pytest.approx(60)
    finally:
        client.remove(loaded)


def test_time_dependent(fresh):
    model, geom, faces = fresh(study=False)
    study = (model/'studies').create(name='transient')
    study.create('Transient').property('tlist', 'range(0,1,4)')
    model.solve()
    with pytest.raises(ValueError, match=r'"transient//Solution 1" has 5 '
                       r"steps; pass step='last'.*model.inner\("):
        mk.average(geom, 'domain', 'T')
    every = mk.average(geom, 'domain', 'T', step='all')
    assert every.shape == (5,)
    assert numpy.all(numpy.diff(every) > 0)
    assert mk.average(geom, 'domain', 'T', step='last') == \
        pytest.approx(every[-1])
    assert mk.average(geom, 'domain', 'T', step=[2, 4]) == \
        pytest.approx(every[[1, 3]])
    # in the order asked, a repeated step included
    assert mk.average(geom, 'domain', 'T', step=[4, 2, 4]) == \
        pytest.approx(every[[3, 1, 3]])
    found = mk.value(geom, 'T', (0.05, 0, 0), step=[5, 1])
    assert found == pytest.approx(mk.value(geom, 'T', (0.05, 0, 0),
                                           step=[1, 5])[::-1])
    with pytest.raises(ValueError, match='has 5 steps, not 7'):
        mk.average(geom, 'domain', 'T', step=7)
    values, where = mk.maximum(geom, 'domain', 'T', step=[2, 3],
                               position=True)
    assert values == pytest.approx([373.15, 373.15])
    assert where.shape == (2, 3)
    found = mk.value(geom, 'T', [(0, 0, 0), (0.05, 0, 0)], step='all')
    assert found.shape == (2, 5)
    assert found[0, 1:] == pytest.approx([373.15]*4)
    assert mk.value(geom, 'T', (0.05, 0, 0), step=1) == \
        pytest.approx(293.15)


def test_stationary_then_transient(fresh):
    model, geom, faces = fresh(study=False)
    study = (model/'studies').create(name='two steps')
    study.create('Stationary')
    study.create('Transient').property('tlist', 'range(0,1,2)')
    model.solve()
    with pytest.raises(ValueError, match='several solved datasets'):
        mk.average(geom, 'domain', 'T')
    assert mk.average(geom, 'domain', 'T', unit='degC', step='all',
                      dataset='dset1') == pytest.approx([60]*3)


def test_parametric_sweep(fresh):
    model, geom, faces = fresh(study=False)
    study = (model/'studies').create(name='sweep')
    study.create('Stationary')
    sweep = study.create('Parametric')
    sweep.property('pname', ['Th'])
    sweep.property('plistarr', ['100 200'])
    sweep.property('punit', ['degC'])
    model.solve()
    assert mk.average(geom, 'domain', 'T', unit='degC', step='all') == \
        pytest.approx([60, 110])
    assert mk.value(geom, 'T', (0, 0, 0), unit='degC', step=2) == \
        pytest.approx(200)


def test_sweep_with_outer_loop(fresh):
    model, geom, faces = fresh(study=False)
    study = (model/'studies').create(name='sweep')
    study.create('Transient').property('tlist', 'range(0,1,2)')
    sweep = study.create('Parametric')
    sweep.property('pname', ['Th'])
    sweep.property('plistarr', ['100 200'])
    sweep.property('punit', ['degC'])
    model.solve()
    for dataset in (None, 'dset1', 'dset2'):
        with pytest.raises(NotImplementedError, match='outer loop are not '
                                                      'supported yet'):
            mk.average(geom, 'domain', 'T', step='last', dataset=dataset)


def test_physics_in_one_domain(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    heat.java.selection().set([1])
    material = (model/'materials').create('Common')
    for key in ('thermalconductivity', 'density', 'heatcapacity'):
        (material/'Basic').property(key, ['1'])
    boundary = heat.create('TemperatureBoundary', 2)
    boundary.select(mk.sel.box(geom, 'boundary', x=0))
    boundary.property('T0', '300[K]')
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    with pytest.raises(RuntimeError, match='Some of these entities have no '
                                           'solution'):
        mk.maximum(geom, 'domain', 'T')
    assert mk.maximum(geom, 'domain', 'T', 1) == pytest.approx(300)
    assert mk.value(geom, 'T', (0.5, 0.5, 0.5), unit='K') == \
        pytest.approx(300)
    with pytest.raises(ValueError, match='Point 1 of 1 is outside the '
                                         'geometry or where nothing was '
                                         'solved'):
        mk.value(geom, 'T', (1.5, 0.5, 0.5))
    # a selection on the solution dataset does not restrict the results:
    # the solved domain 1 still has them
    dataset = model.java.result().dataset('dset1')
    dataset.selection().geom(geom.tag(), 3)
    dataset.selection().set([2])
    assert mk.integral(geom, 'domain', '1', 1) == pytest.approx(1)
    assert mk.value(geom, 'T', (0.5, 0.5, 0.5)) == pytest.approx(300)


########################
# Other geometry kinds #
########################

def solve_heat(model, geom, hot, cold):
    """Adds heat transfer between a hot and a cold boundary and solves."""
    material = (model/'materials').create('Common')
    for key in ('thermalconductivity', 'density', 'heatcapacity'):
        (material/'Basic').property(key, ['1'])
    heat = (model/'physics').create('HeatTransfer', geom)
    for selection, temperature in ((hot, '100[degC]'), (cold, '20[degC]')):
        boundary = heat.create('TemperatureBoundary', 1)
        boundary.select(selection)
        boundary.property('T0', temperature)
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()


def test_2d(model):
    geom = mk.geometry(model, 2)
    mk.rectangle(geom, (0.1, 0.05))
    model.build(geom)
    solve_heat(model, geom, mk.sel.box(geom, 'boundary', x=0),
               mk.sel.box(geom, 'boundary', x=0.1))
    assert mk.integral(geom, 'domain', '1') == pytest.approx(0.005)
    assert mk.integral(geom, 'boundary', '1') == pytest.approx(0.3)
    assert mk.average(geom, 'edge', 'T', unit='degC') == pytest.approx(60)
    value, where = mk.maximum(geom, 'domain', 'T', unit='degC',
                              position=True)
    assert where.shape == (2,)
    assert mk.value(geom, 'T', (0.025, 0.01), unit='degC') == \
        pytest.approx(80)


def test_axisymmetric(model):
    geom = mk.geometry(model, 2)
    geom.java.axisymmetric(True)
    mk.rectangle(geom, (2, 1))
    model.build(geom)
    solve_heat(model, geom, mk.sel.box(geom, 'boundary', y=0),
               mk.sel.box(geom, 'boundary', y=1))
    assert mk.integral(geom, 'domain', '1') == pytest.approx(4*math.pi)
    assert mk.average(geom, 'domain', 'r') == pytest.approx(4/3)
    assert mk.average(geom, 'domain', 'r', intvolume=False) == \
        pytest.approx(1)
    side = mk.sel.box(geom, 'boundary', x=2)
    assert mk.integral(geom, 'boundary', '1', side) == \
        pytest.approx(4*math.pi)
    assert mk.value(geom, 'T', (1, 0.5), unit='degC') == pytest.approx(60)


def test_millimeters(model):
    geom = mk.geometry(model, 3, length_unit='mm')
    mk.block(geom, (100, 50, 10))
    model.build(geom)
    material = (model/'materials').create('Common')
    for key in ('thermalconductivity', 'density', 'heatcapacity'):
        (material/'Basic').property(key, ['1'])
    heat = (model/'physics').create('HeatTransfer', geom)
    for x, temperature in ((0, '100[degC]'), (100, '20[degC]')):
        boundary = heat.create('TemperatureBoundary', 2)
        boundary.select(mk.sel.box(geom, 'boundary', x=x))
        boundary.property('T0', temperature)
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    assert mk.integral(geom, 'domain', '1') == pytest.approx(5e-5)
    assert mk.integral(geom, 'domain', '1', unit='mm^3') == \
        pytest.approx(5e4)
    value, where = mk.minimum(geom, 'domain', 'T', unit='degC',
                              position=True)
    assert where[0] == pytest.approx(100)
    assert mk.value(geom, 'T', (25, 25, 5), unit='degC') == \
        pytest.approx(80)


def test_complex(model):
    geom = mk.geometry(model, 2)
    mk.rectangle(geom, (1, 0.2))
    model.build(geom)
    acoustics = (model/'physics').create('PressureAcoustics', geom)
    source = acoustics.create('Pressure', 1)
    source.select(mk.sel.box(geom, 'boundary', x=0))
    source.property('p0', '1[Pa]')
    acoustics.create('PlaneWaveRadiation', 1).select(
        mk.sel.box(geom, 'boundary', x=1))
    material = (model/'materials').create('Common')
    (material/'Basic').property('density', ['1.2'])
    (material/'Basic').property('soundspeed', ['343'])
    (model/'meshes').create(geom)
    (model/'studies').create().create('Frequency').property('plist', '500')
    model.solve()
    mean = mk.average(geom, 'domain', 'p', unit='Pa')
    assert isinstance(mean, complex)
    assert mk.integral(geom, 'domain', 'p') == pytest.approx(mean*0.2)
    point = mk.value(geom, 'p', (0, 0.1), unit='Pa')
    assert isinstance(point, complex)
    assert point == pytest.approx(1, abs=0.01)
    assert isinstance(mk.maximum(geom, 'domain', 'p'), float)
    assert mk.maximum(geom, 'domain', 'abs(p)') == pytest.approx(1, abs=0.01)


def test_1d_and_workplane(model):
    line = mk.geometry(model, 1)
    mk.interval(line, [0, 1])
    model.build(line)
    with pytest.raises(ValueError, match='needs a 2D or 3D geometry'):
        mk.integral(line, 'domain', '1')
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom)
    with pytest.raises(TypeError, match='work plane'):
        mk.value(plane, 'T', (0, 0))


###########
# History #
###########

def test_history(fresh, tmp_path):
    model, geom, faces = fresh()
    before = java_export(model, tmp_path/'model.java')
    mk.integral(geom, 'boundary', 'ht.ntflux', faces['hot'], unit='W')
    mk.maximum(geom, 'domain', 'T', unit='degC', position=True)
    mk.value(geom, 'T', [(0, 0, 0)], unit='degC')
    with pytest.raises(ValueError):
        mk.average(geom, 'domain', 'T', unit='kg')
    with pytest.raises(ValueError):
        mk.value(geom, 'T', (1, 1, 1))
    assert java_export(model, tmp_path/'model.java') == before
