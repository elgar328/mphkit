"""
Tests of parametric sweeps that COMSOL stores as an outer loop: which
datasets the helpers take or refuse, reading them with `outer=`, and
`mk.outer_values` and `mk.step_values`.

The plate of test_results, 0.1 x 0.05 x 0.01 m, held at Th at x = 0 and
20 degC at x = 0.1, swept over Th around a short time-dependent study.
"""
import math

import numpy
import pytest

import mphkit as mk
from test_results import leftovers, leaves_nothing, plate  # noqa: F401

KELVIN = 273.15


def sweep(study, values='100 200 300', name='Th', unit='degC'):
    """Adds a parametric sweep to a study node."""
    node = study.create('Parametric')
    node.property('pname', [name])
    node.property('plistarr', [values])
    node.property('punit', [unit])
    return node


def transient(model, name='sweep', values='100 200 300'):
    """Adds and solves a time-dependent study swept over Th."""
    study = (model/'studies').create(name=name)
    study.create('Transient').property('tlist', 'range(0,1,2)')
    sweep(study, values)
    model.solve(name)
    return study


def outer_values(geom, **options):
    """mk.outer_values, checked to leave nothing in the model."""
    before = leftovers(geom.model)
    try:
        return mk.outer_values(geom, **options)
    finally:
        assert leftovers(geom.model) == before


def step_values(geom, **options):
    """mk.step_values, checked to leave nothing in the model."""
    before = leftovers(geom.model)
    try:
        return mk.step_values(geom, **options)
    finally:
        assert leftovers(geom.model) == before


@pytest.fixture(scope='module')
def swept(client):
    """The plate swept over Th = 100, 200, 300 degC, time-dependent."""
    model, geom, faces = plate(client, 'swept plate', study=False)
    study = transient(model)
    yield model, geom, study, faces
    client.remove(model)


@pytest.fixture
def fresh(client):
    """Builds own plates (by calling it) and removes them afterwards."""
    made = []

    def build(**options):
        model, geom, faces = plate(client, f'sweep plate {len(made) + 1}',
                                   study=False, **options)
        made.append(model)
        return model, geom
    yield build
    for model in made:
        client.remove(model)


def dataset_named(model, part):
    """Returns the tag of the dataset whose name contains `part`."""
    found = [str(tag) for tag in model.java.result().dataset().tags()
             if part in str(model.java.result().dataset(tag).label())]
    assert len(found) == 1, found
    return found[0]


################
# outer_values #
################

def test_outer_values(swept):
    model, geom, study, faces = swept
    expected = [{'Th': 100 + KELVIN}, {'Th': 200 + KELVIN},
                {'Th': 300 + KELVIN}]
    for dataset in (None, 'dset2', study, 'sweep'):
        assert outer_values(geom, dataset=dataset) == \
            pytest.approx(expected)


def test_plain_and_inner_sweep(fresh):
    model, geom = fresh()
    study = (model/'studies').create(name='static')
    study.create('Stationary')
    model.solve()
    assert outer_values(geom) == []
    assert step_values(geom) == {}
    with pytest.raises(ValueError, match=r'"static//Solution 1" \(dset1\) '
                                         r'has no outer sweep; leave out '
                                         r'outer=\.'):
        mk.average(geom, 'domain', 'T', outer=1)
    sweep(study, '100 200')
    model.solve()
    with pytest.raises(ValueError, match=r'holds its sweep over Th as steps, '
                                         r'not as outer values: pass step='):
        outer_values(geom)
    with pytest.raises(ValueError, match=r'sweep over Th as steps, not as '
                       r'outer values: pass step=2 instead of outer= '
                       r'\(mk.step_values\(geom\) gives the values\)'):
        mk.average(geom, 'domain', 'T', outer=2)
    assert step_values(geom)['Th'] == pytest.approx([100 + KELVIN,
                                                    200 + KELVIN])
    with pytest.raises(ValueError, match='as steps, not as outer values; '
                                         'leave out outer= to get them'):
        step_values(geom, outer=1)
    with pytest.raises(ValueError, match=r'as steps, not as outer values: '
                                         r'pass step=2 instead of outer='):
        mk.average(geom, 'domain', 'T', outer={'Th': '200[degC]'})


############
# Datasets #
############

def test_copy(swept):
    model, geom, study, faces = swept
    with pytest.raises(ValueError, match=r'Dataset "sweep//Solution 1" '
                       r"\(dset1\) holds only the last value of a parametric "
                       r"sweep; pass dataset='dset2', .* with outer='last' "
                       r'\(Th=573.15 \(300 degC\)\)'):
        outer_values(geom, dataset='dset1')
    with pytest.raises(ValueError, match='holds only the last value'):
        mk.average(geom, 'domain', 'T', dataset='dset1')


def test_child(swept):
    model, geom, study, faces = swept
    java = model.java
    sweep = [str(tag) for tag in java.sol().tags()
             if len(java.sol(tag).getSolutioninfo().getOuterSolnum())][0]
    child = mk._datasets.stored(java.sol(sweep))[1]
    dataset = (model/'datasets').create('Solution', name='one value')
    try:
        dataset.property('solution', child)
        with pytest.raises(ValueError, match=r'"one value" \(dset\d\) holds '
                           r'one value of a parametric sweep \(Th=473.15 '
                           r"\(200 degC\)\); pass dataset='dset2', .* with "
                           r'outer=2\.'):
            outer_values(geom, dataset='one value')
    finally:
        dataset.remove()


def test_store_of_two_steps(fresh):
    model, geom = fresh()
    study = (model/'studies').create(name='two steps')
    study.create('Stationary')
    study.create('Transient').property('tlist', 'range(0,1,2)')
    sweep(study, '100 200')
    model.solve()
    store = dataset_named(model, 'Solution Store')
    with pytest.raises(ValueError, match=r'holds the first study step of the '
                       r"sweep's last value only.* pass dataset='dset\d', "):
        outer_values(geom, dataset=store)
    assert outer_values(geom) == pytest.approx([{'Th': 100 + KELVIN},
                                                {'Th': 200 + KELVIN}])


def test_sweep_removed(fresh):
    model, geom = fresh()
    transient(model, values='100 200')
    java = model.java
    (model/'datasets'/'sweep//Parametric Solutions 1').remove()
    with pytest.raises(RuntimeError, match=r'holds only the last value of a '
                       r'parametric sweep; its solution (sol\d) has no '
                       r'dataset for geometry'):
        outer_values(geom)
    sweep = [str(tag) for tag in java.sol().tags()
             if len(java.sol(tag).getSolutioninfo().getOuterSolnum())][0]
    # the advice in the message
    ds = mk.set((model/'datasets').create('Solution').java, solution=sweep,
                geom=geom.tag())
    assert outer_values(geom, dataset=str(ds.tag())) == \
        pytest.approx([{'Th': 100 + KELVIN}, {'Th': 200 + KELVIN}])
    java.result().dataset().remove(str(ds.tag()))
    java.sol().remove(sweep)
    with pytest.raises(RuntimeError, match=r'holds one value of a parametric '
                       r'sweep whose other values are gone \(Th=473.15\); '
                       r"run model.solve\('sweep'\) again"):
        outer_values(geom)


def test_by_study(fresh):
    model, geom = fresh()
    study = transient(model, values='100 200')
    other = (model/'studies').create(name='other')
    other.create('Stationary')
    with pytest.raises(ValueError, match=r'Study "other" has no solved '
                                         r'dataset for geometry'):
        outer_values(geom, dataset=other)
    with pytest.raises(LookupError, match=r'No dataset "nothing"; the model '
                       r"has .*\. No study \"nothing\" either; the model has "
                       r"'sweep', 'other'\."):
        outer_values(geom, dataset='nothing')
    model.solve('other')
    with pytest.raises(ValueError, match=r'several solved datasets: .*; pass '
                       r"dataset= one of these, or the study, e.g. "
                       r"dataset='sweep'\.$"):
        outer_values(geom)
    assert outer_values(geom, dataset='other') == []
    assert len(outer_values(geom, dataset=study)) == 2
    # the sweep switched off and solved again: the old sweep stays
    (study/'Parametric Sweep').toggle('off')
    model.solve('sweep')
    for dataset in (None, study):
        with pytest.raises(ValueError, match=r'several solved datasets: .* '
                           r'Some come from the same study, .* '
                           r"model.java.sol\(\).remove\('sol\d'\)"):
            outer_values(geom, dataset=dataset)
        with pytest.raises(ValueError, match='several solved datasets'):
            mk.average(geom, 'domain', 'T', dataset=dataset, outer=1,
                       step='last')
    assert outer_values(geom, dataset='dset1') == []


###########
# Failure #
###########

def failing(fresh, values=None):
    """A plate whose conductivity divides by zero at kk = 50."""
    model, geom = fresh()
    model.parameter('kk', '50')
    (model/'materials'/'steel'/'Basic').property('thermalconductivity',
                                                ['kk/(kk-50)'])
    study = (model/'studies').create(name='s')
    study.create('Transient').property('tlist', 'range(0,1,2)')
    if values:
        sweep(study, values, 'kk', '')
    with pytest.raises(Exception):
        model.solve('s')
    return model, geom, study


@pytest.mark.parametrize('values', [None, '50 60', '45 50 60'])
def test_failed(fresh, values):
    model, geom, study = failing(fresh, values)
    message = r'The last solve of study "s" failed: .* \(solution sol1\)'
    with pytest.raises(RuntimeError, match=message):
        outer_values(geom)
    with pytest.raises(RuntimeError, match=message):
        mk.average(geom, 'domain', 'T', step='last')
    for tag in model.java.result().dataset().tags():
        with pytest.raises(RuntimeError, match=message):
            outer_values(geom, dataset=str(tag))


def test_failed_then_fixed(fresh):
    model, geom, study = failing(fresh, '45 50 60')
    other = (model/'studies').create(name='other')
    other.create('Stationary')
    model.parameter('kk', '60')
    model.solve('other')
    with pytest.raises(RuntimeError, match=r'failed: .* Other solved '
                       r'datasets: "other//Solution \d+" \(dset\d\); pass '
                       r'dataset= one of these\.'):
        outer_values(geom)
    assert outer_values(geom, dataset='other') == []
    (model/'materials'/'steel'/'Basic').property('thermalconductivity',
                                                ['45'])
    model.solve('s')
    assert outer_values(geom, dataset='s') == \
        pytest.approx([{'kk': 45}, {'kk': 50}, {'kk': 60}])


###########
# Numbers #
###########

def test_numbers(swept):
    model, geom, study, faces = swept
    hot = faces['hot']
    found = mk.average(geom, 'boundary', 'T', hot, unit='degC',
                       outer='all', step='last')
    assert found == pytest.approx([100, 200, 300])
    one = mk.average(geom, 'boundary', 'T', hot, unit='degC', outer=2,
                     step='last')
    assert isinstance(one, float)
    assert one == pytest.approx(200)
    assert mk.average(geom, 'boundary', 'T', hot, outer=[1],
                      step='last').shape == (1,)
    every = mk.average(geom, 'domain', 'T', outer='all', step='all')
    assert every.shape == (3, 3)
    assert numpy.all(numpy.diff(every[:, -1]) > 0)
    assert mk.average(geom, 'domain', 'T', outer=[3, 1, 3], step='last') \
        == pytest.approx(every[[2, 0, 2], -1])
    assert mk.average(geom, 'domain', 'T', outer='last', step=[3, 2]) == \
        pytest.approx(every[2, [2, 1]])
    values, where = mk.maximum(geom, 'domain', 'T', unit='degC',
                               outer='all', step='last', position=True)
    assert values == pytest.approx([100, 200, 300])
    assert where.shape == (3, 3)
    assert where[:, 0] == pytest.approx([0, 0, 0])
    values, where = mk.minimum(geom, 'domain', 'T', outer='all',
                               step='all', position=True)
    assert values.shape == (3, 3)
    assert where.shape == (3, 3, 3)
    # COMSOL's outer-loop interpolation reads the first value only
    inside = mk.value(geom, 'T', (0.002, 0.025, 0.005), unit='degC',
                      outer='all', step='last')
    assert inside.shape == (3,)
    assert numpy.all(numpy.diff(inside) > 10)
    assert mk.value(geom, 'T', (0.002, 0.025, 0.005), unit='degC', outer=2,
                    step='last') == pytest.approx(inside[1])
    both = mk.value(geom, 'T', [(0, 0, 0), (0.002, 0.025, 0.005)],
                    unit='degC', outer='all', step='all')
    assert both.shape == (2, 3, 3)
    assert both[0, :, -1] == pytest.approx([100, 200, 300])
    assert both[1, :, -1] == pytest.approx(inside)


def test_number_errors(swept):
    model, geom, study, faces = swept
    with pytest.raises(ValueError, match=r'holds a parametric sweep over 3 '
                       r'values \(1: Th=373.15 \(100 degC\); 2: Th=473.15 '
                       r'\(200 degC\); 3: Th=573.15 \(300 degC\)\); pass '
                       r"outer= .*mk.outer_values\(geom\)"):
        mk.average(geom, 'domain', 'T', step='last')
    with pytest.raises(ValueError, match='has 3 outer values, not 4'):
        mk.average(geom, 'domain', 'T', outer=4, step='last')
    with pytest.raises(ValueError, match=r'at outer=2 \(Th=473.15 \(200 '
                       r"degC\)\) has 3 steps; pass step='last'.* "
                       r'\(mk.step_values\(geom, outer=2\) gives them: '
                       r't = 0, 1, 2\)'):
        mk.average(geom, 'domain', 'T', outer=2)
    with pytest.raises(ValueError, match='has 3 steps, not 5'):
        mk.value(geom, 'T', (0, 0, 0), outer=2, step=5)
    with pytest.raises(ValueError, match='outer counts from 1'):
        mk.average(geom, 'domain', 'T', outer=0, step='last')
    with pytest.raises(TypeError, match='outer must be'):
        mk.average(geom, 'domain', 'T', outer=1.5, step='last')
    with pytest.raises(ValueError, match=r'COMSOL evaluated "T" in .*K, '
                                         r"not 'kg'"):
        mk.average(geom, 'domain', 'T', unit='kg', outer='all',
                   step='last')
    with pytest.raises(ValueError, match=r'Point 1 of 1 is outside .* at '
                                         r'outer=1 \(Th=373.15'):
        mk.value(geom, 'T', (1, 1, 1), outer='all', step='last')
    assert numpy.isnan(mk.value(geom, 'T', (1, 1, 1), outer='all',
                                step='last', outside='nan')).all()


def test_step_values(swept):
    model, geom, study, faces = swept
    found = step_values(geom)
    assert list(found) == ['t']
    assert found['t'] == pytest.approx([0, 1, 2])
    assert step_values(geom, outer=2)['t'] == pytest.approx([0, 1, 2])
    assert step_values(geom, outer='all')['t'].shape == (3, 3)
    assert step_values(geom, outer=[3, 1])['t'].shape == (2, 3)
    with pytest.raises(ValueError, match='holds only the last value'):
        step_values(geom, dataset='dset1')


def test_free_time_steps(fresh):
    model, geom = fresh()
    study = (model/'studies').create(name='free')
    study.create('Transient').property('tlist', 'range(0,10,100)')
    sweep(study, '100 1000')
    model.solve('free')
    java = model.java
    for tag in java.sol('sol1').feature().tags():
        solver = java.sol('sol1').feature(tag)
        if str(solver.getType()) == 'Time':
            solver.set('tout', 'tsteps')
            solver.set('tstepsbdf', 'free')
    model.solve('free')
    first = step_values(geom, outer=1)['t']
    second = step_values(geom, outer=2)['t']
    assert len(first) != len(second)
    assert first[-1] == second[-1] == pytest.approx(100)
    assert mk.average(geom, 'domain', 'T', outer='all',
                      step='last').shape == (2,)
    assert mk.average(geom, 'domain', 'T', outer=2,
                      step=len(second)) == pytest.approx(
        mk.average(geom, 'domain', 'T', outer=2, step='last'))
    for step in ('all', 2, [1, 2]):
        with pytest.raises(ValueError, match=r'outer values 1 and 2 of .* '
                           r'have different steps \(outer=1: \d+ steps, '
                           r"outer=2: \d+ steps\); pass step='first'"):
            mk.average(geom, 'domain', 'T', outer='all', step=step)
    with pytest.raises(ValueError, match=r"different steps .*; pass outer=k "
                                         r'for one value at a time\.'):
        step_values(geom)
    with pytest.raises(ValueError, match='outer=k for one value at a time'):
        step_values(geom, outer='all')


def test_eigenfrequencies(client):
    model = client.create('beam')
    try:
        model.parameter('E0', '200[GPa]')
        geom = mk.geometry(model, 3)
        mk.block(geom, (0.1, 0.01, 0.01))
        model.build(geom)
        steel = (model/'materials').create('Common')
        for key, value in (('youngsmodulus', 'E0'), ('poissonsratio', '0.3'),
                           ('density', '7850')):
            (steel/'Basic').property(key, [value])
        solid = (model/'physics').create('SolidMechanics', geom)
        solid.create('Fixed', 2).select(mk.sel.box(geom, 'boundary', x=0))
        (model/'meshes').create(geom)
        study = (model/'studies').create(name='eigen')
        steps = study.create('Eigenfrequency')
        steps.property('neigs', 2)
        steps.property('shift', '100')
        sweep(study, '100[GPa] 200[GPa]', 'E0', 'Pa')
        model.solve()
        found = step_values(geom, outer=1)
        assert list(found) == ['lambda', 'freq']
        assert found['freq'].dtype == float
        assert found['lambda'] == pytest.approx(-2j*math.pi*found['freq'])
        assert mk.average(geom, 'domain', 'solid.freq', outer=1,
                          step='all') == pytest.approx(found['freq'])
        # eigenvalues differ between values by nature: their number counts
        every = mk.average(geom, 'domain', 'solid.freq', outer='all',
                           step='all')
        assert every.shape == (2, 2)
        assert every[1] == pytest.approx(every[0]*math.sqrt(2))
        assert step_values(geom, outer='all')['freq'] == \
            pytest.approx(every)
        with pytest.raises(ValueError, match='different steps'):
            step_values(geom)
    finally:
        client.remove(model)


def test_material_sweep(fresh):
    model, geom = fresh()
    materials = mk.component_of(geom).java.material()
    for tag in list(materials.tags()):
        materials.remove(tag)
    switch = materials.create('sw1', 'Switch')
    switch.selection().all()
    for tag, conductivity in (('ma', '10'), ('mb', '90')):
        material = switch.feature().create(tag, 'Common')
        group = material.propertyGroup('def')
        for key, value in (('thermalconductivity', conductivity),
                           ('density', '7850'), ('heatcapacity', '475')):
            group.set(key, value)
    study = (model/'studies').create(name='materials')
    study.create('Stationary')
    swept = study.java.create('matsw', 'MaterialSweep')
    swept.set('pname', ['matsw.comp1.sw1'])
    swept.set('plistarr', ['1 2'])
    model.solve()
    assert outer_values(geom) == [{'matsw.comp1.sw1': 1.0},
                                  {'matsw.comp1.sw1': 2.0}]
    with pytest.raises(ValueError, match=r'1: matsw.comp1.sw1=1.0 \[.*\]; '
                                         r'2: matsw.comp1.sw1=2.0 \['):
        mk.integral(geom, 'domain', '1')
    hot = mk.sel.box(geom, 'boundary', x=0)
    flux = mk.integral(geom, 'boundary', 'ht.ntflux', hot, outer='all')
    assert flux == pytest.approx([-10/45*18, -90/45*18])
    assert mk.integral(geom, 'boundary', 'ht.ntflux', hot,
                       outer={'matsw.comp1.sw1': 2}) == pytest.approx(-36)
    with pytest.raises(ValueError, match='takes the number of the case'):
        mk.integral(geom, 'boundary', 'ht.ntflux', hot,
                    outer={'matsw.comp1.sw1': '2'})


############
# Geometry #
############

def test_remeshed(fresh):
    model, geom = fresh()
    transient(model, values='100 200')
    (model/'meshes'/'mesh').java.autoMeshSize(3)
    model.mesh()
    hot = mk.sel.box(geom, 'boundary', x=0)
    assert mk.average(geom, 'boundary', 'T', hot, unit='degC',
                      outer='all', step='last') == pytest.approx([100, 200])


def test_mesh_sweep(fresh):
    model, geom = fresh()
    model.parameter('hm', '0.02')
    size = (model/'meshes'/'mesh').create('Size')
    size.property('custom', 'on')
    size.property('hmaxactive', True)
    size.property('hmax', 'hm')
    (model/'meshes'/'mesh').create('FreeTet')
    study = (model/'studies').create(name='mesh sweep')
    study.create('Stationary')
    sweep(study, '0.02 0.01', 'hm', 'm')
    model.solve()
    assert outer_values(geom) == pytest.approx([{'hm': 0.02}, {'hm': 0.01}])
    hot = mk.sel.box(geom, 'boundary', x=0)
    assert mk.average(geom, 'boundary', 'T', hot, unit='degC',
                      outer='all') == pytest.approx([100, 100])


def test_stale(fresh):
    model, geom = fresh()
    transient(model, values='100 200')
    block = geom/'Block 1'
    block.property('size', ['0.2', '0.05', '0.01'])
    with pytest.raises(RuntimeError, match=r'Geometry ".*" is not built; '
                       r'run model.build\(geom\), and if it changed'):
        mk.average(geom, 'domain', 'T', outer=1, step='last')
    model.build(geom)
    model.mesh()
    with pytest.raises(RuntimeError, match=r'changed since the solve; run '
                       r'model.build\(geom\), model.mesh\(\) and '
                       r'model.solve\(\)'):
        mk.average(geom, 'domain', 'T', outer=1, step='last')


##################
# Without COMSOL #
##################

@pytest.mark.parametrize('built, requested, others, expected', [
    (True, ['same', 'same'], 'differ', 'full'),
    (True, ['same', 'unknown'], 'same', 'unknown'),
    (False, ['unknown'], 'same', 'unknown'),
    (True, ['same', 'different'], 'differ', 'sweep'),
    (True, ['different'], 'same', 'stale'),
    (True, ['different'], 'single', 'stale-single'),
    (False, ['different'], 'differ', 'sweep'),
    (False, ['different'], 'same', 'unbuilt'),
    (False, ['different'], 'single', 'unbuilt-single')])
def test_decide(built, requested, others, expected):
    asked = []

    def compare():
        asked.append(True)
        return others
    assert mk._sweep.decide(built, requested, compare) == expected
    # the other values are only read when needed
    assert bool(asked) == (expected not in ('full', 'unknown'))


def test_same_points():
    same = mk._sweep.same_points
    points = numpy.array([[0, 0, 0], [0.1, 0, 0], [0, 0.05, 0.01]])
    assert same(points, points[[2, 0, 1]])
    assert same(points, points + 1e-15)
    assert not same(points, points + [1e-6, 0, 0])
    assert not same(points, points[:2])
    # rows that sort apart within the tolerance
    near = numpy.array([[1, 2], [1 + 1e-14, 1]])
    assert same(near, numpy.array([[1 + 1e-14, 2], [1, 1]]))


class Solution:
    """Stands in for a Java solution with steps."""

    def __init__(self, names, real, imag=None):
        self.names, self.real, self.imag = names, real, imag

    def getPNames(self):
        return self.names

    def getPVals(self):
        return self.real

    def getPValsImag(self):
        return self.imag

    def getSolutioninfo(self):
        count = max(1, len(self.real)//max(1, len(self.names)))
        info = type('Info', (), {})()
        info.getSolnum = lambda *args: list(range(1, count + 1))
        return info


class Model:
    """Stands in for a Java model with solutions by tag."""

    def __init__(self, **solutions):
        self.solutions = solutions

    def sol(self, tag):
        return self.solutions[tag]


def test_same_steps():
    same = mk._sweep.same_steps
    model = Model(a=Solution(['t'], [0, 1, 2]), b=Solution(['t'], [0, 1, 2]),
                  c=Solution(['t'], [0, 1, 2.5]), d=Solution(['t'], [0, 1]),
                  e=Solution(['t'], [0, 1, 2 + 1e-12]),
                  f=Solution(['lambda'], [0, 0], [-1, -2]),
                  g=Solution(['lambda'], [0, 0], [-3, -4]),
                  h=Solution([''], [0]), i=Solution([''], [0]),
                  j=Solution(['Th', 'k'], [1, 10, 1, 90]),
                  k=Solution(['k', 'Th'], [10, 1, 90, 1]))
    assert same(model, ['a', 'b', 'e'])
    assert not same(model, ['a', 'c'])
    assert not same(model, ['a', 'd'])
    assert same(model, ['f', 'g'])
    assert not same(model, ['f', 'g'], strict=True)
    assert same(model, ['f'], strict=True)
    assert same(model, ['h', 'i'], strict=True)
    assert same(model, ['j', 'j'])
    assert not same(model, ['j', 'k'])


def test_named_steps():
    named = mk._sweep._step_table
    table = named(Solution(['Th', 'k'], [1, 10, 2, 90, 3, 10]))
    assert table['Th'] == pytest.approx([1, 2, 3])
    assert table['k'] == pytest.approx([10, 90, 10])
    assert named(Solution([''], [0])) == {}
    with pytest.raises(RuntimeError, match='3 step values for 2 names'):
        named(Solution(['Th', 'k'], [1, 10, 2]))


def test_listed():
    labels = [f'Th={n}' for n in range(1, 11)]
    assert mk._sweep.listed(labels[:3]) == '1: Th=1; 2: Th=2; 3: Th=3'
    assert mk._sweep.listed(labels) == ('1: Th=1; 2: Th=2; 3: Th=3; 4: Th=4; '
                                        '5: Th=5; ...; 9: Th=9; 10: Th=10')


def title(expected, candidates, number=2, switches=()):
    """A title to check, as `Sweep.title` makes it."""
    return mk._sweep.Title(number, expected, candidates, list(switches),
                           'outer=k')


@pytest.mark.parametrize('indicator, number, expected, candidates, '
                         'switches, problem', [
    ('Th(2)=1000 degC Time=21.6 s', 2, {'Th': 1000}, {'Th': [100, 1000]},
     (), None),
    ('Th(2)=1000 degC Time=21.6 s', 1, {'Th': 100}, {'Th': [100, 1000]},
     (), 'wrong'),
    ('Th=100 degC, k=90 W/m/K Time=10 s', 3, {'Th': 100, 'k': 90},
     {'Th': [100, 100, 200, 200], 'k': [10, 90, 10, 90]}, (), None),
    ('Th=100 degC, k=90 W/m/K Time=10 s', 1, {'Th': 100, 'k': 10},
     {'Th': [100, 100, 200, 200], 'k': [10, 90, 10, 90]}, (), 'wrong'),
    ('k=10 W/m/K, Th=200 degC Time=10 s', 2, {'k': 10, 'Th': 200},
     {'k': [10, 10], 'Th': [100, 200]}, (), None),
    ('a(2)=2 lambda(3)=32.083 rad/s', 2, {'a': 2}, {'a': [1, 2]}, (), None),
    ('Th(2)=200 degC k(1)=10 W/m/K', 2, {'Th': 200}, {'Th': [100, 200]},
     (), None),
    ('k(2)=90 W/m/K Th(1)=100 degC', 2, {'k': 90}, {'k': [10, 90]}, (),
     None),
    ('hm(2)=0.005 m', 2, {'hm': 0.005}, {'hm': [0.01, 0.005]}, (), None),
    ('dT(4)=5.5511E-17 K Time=10 s', 4, {'dT': 5.551115123125783e-17},
     {'dT': [-0.2, -0.1, 0.1, 5.551115123125783e-17]}, (), None),
    ('W(2)=133.33 mm', 2, {'W': 133.33333333}, {'W': [100, 133.33333333]},
     (), None),
    ('Th(2)=100.12 degC', 2, {'Th': 100.124},
     {'Th': [100.123456789, 100.124]}, (), None),
    ('p0(2)=2 kPa freq(1)=100 Hz', 2, {'p0': 2}, {'p0': [1, 2]}, (), None),
    ('Th(1)=300 degC Time=10 s', 1, {'Th': 300}, {'Th': [300]}, (), None),
    ('Th(1)=300 degC Time=10 s', 1, {'Th': 200}, {'Th': [200]}, (),
     'wrong'),
    ('Th(2)=200 degC Time=1 s', 2, {'Th': 200}, {'Th': [100, 200, 300]},
     (), None),
    ('Material Switch 1(2)=Material 2 Time=10 s', 2, {}, {},
     [('Material Switch 1', 'Material 2')], None),
    ('Material Switch 1(2)=Material 2 Time=10 s', 2, {}, {},
     [('Material Switch 1', 'Material')], 'wrong'),
    ('Th=200 degC, Material Switch 1=Material Time=10 s', 3, {'Th': 200},
     {'Th': [100, 100, 200, 200]}, [('Material Switch 1', 'Material')],
     None),
    ('Th=200 degC, Material Switch 1=Material Time=10 s', 4, {'Th': 200},
     {'Th': [100, 100, 200, 200]}, [('Material Switch 1', 'Material 2')],
     'wrong'),
    ('Th=200 degC, Material Switch 1=Material 2', 4, {'Th': 200},
     {'Th': [100, 100, 200, 200]}, [('Material Switch 1', 'Material 2')],
     None),
    ('Function Switch 1(2)=Analytic 2 Time=10 s', 2, {}, {},
     [('Function Switch 1', 'Analytic 2')], None),
    ('Time=10 s', 2, {'Th': 200}, {'Th': [100, 200]}, (), 'missing'),
    ('', 2, {}, {}, [('Material Switch 1', 'Material 2')], 'missing')])
def test_title(indicator, number, expected, candidates, switches, problem):
    found = mk._sweep.title_problem(
        indicator, title(expected, candidates, number, switches))
    assert found == problem


##########################
# Sweeps of the geometry #
##########################

@pytest.fixture(scope='module')
def widths(client):
    """
    The heat plate swept over its width W = 0.1 and 0.15 m, stationary:
    T = 100 degC at x = 0 and 20 degC at x = W, so the heat through it
    is 18 and 12 W.
    """
    model = client.create('widths')
    model.parameter('W', '0.1[m]')
    model.parameter('Th', '100[degC]')
    geom = mk.geometry(model, 3)
    block = mk.block(geom, ('W', 0.05, 0.01))
    model.build(geom)
    faces = {'hot': mk.sel.box(geom, 'boundary', x=0, name='hot'),
             'cold': mk.sel.box(geom, 'boundary', x='W', name='cold'),
             # before the solve: its values hold it then
             'block': mk.sel.result(geom, block, 'domain')}
    steel = (model/'materials').create('Common', name='steel')
    for key, value in (('thermalconductivity', '45'), ('density', '7850'),
                       ('heatcapacity', '475')):
        (steel/'Basic').property(key, [value])
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    for face, temperature in (('hot', 'Th'), ('cold', '20[degC]')):
        boundary = heat.create('TemperatureBoundary', 2, name=face)
        boundary.select(faces[face])
        boundary.property('T0', temperature)
    (model/'meshes').create(geom, name='mesh')
    study = (model/'studies').create(name='widths')
    study.create('Stationary')
    sweep(study, '0.1 0.15', 'W', 'm')
    model.solve()
    yield model, geom, block, faces
    client.remove(model)


def test_geometry_sweep(widths):
    model, geom, block, faces = widths
    assert outer_values(geom) == pytest.approx([{'W': 0.1}, {'W': 0.15}])
    assert mk.integral(geom, 'boundary', 'ht.ntflux', faces['hot'],
                       unit='W', outer='all') == pytest.approx([-18, -12])
    assert mk.integral(geom, 'boundary', 'ht.ntflux', faces['cold'],
                       unit='W', outer=2) == pytest.approx(12)
    assert mk.average(geom, 'domain', 'T', unit='degC', outer='all') == \
        pytest.approx([60, 60])
    assert mk.integral(geom, 'domain', '1', outer='all') == \
        pytest.approx([5e-5, 7.5e-5])
    assert mk.integral(geom, 'domain', '1', faces['block'], outer='all') == \
        pytest.approx([5e-5, 7.5e-5])
    ends = mk.sel.union(geom, 'boundary', [faces['hot'], faces['cold']])
    assert mk.average(geom, 'boundary', 'T', ends, unit='degC',
                      outer='all') == pytest.approx([60, 60])
    values, where = mk.maximum(geom, 'domain', 'T', unit='degC',
                               outer='all', position=True)
    assert values == pytest.approx([100, 100])
    assert where[:, 0] == pytest.approx([0, 0])
    # a point beyond the narrower plate
    found = mk.value(geom, 'T', (0.12, 0.025, 0.005), unit='degC',
                     outer='all', outside='nan')
    assert numpy.isnan(found[0])
    assert found[1] == pytest.approx(100 - 80*0.12/0.15, abs=0.5)
    with pytest.raises(ValueError, match=r'Point 1 of 1 is outside .* at '
                                         r'outer=1 \(W=0.1 \(0.1 m\)\)'):
        mk.value(geom, 'T', (0.12, 0.025, 0.005), outer='all')


def test_geometry_sweep_refusals(widths, tmp_path):
    model, geom, block, faces = widths
    with pytest.raises(ValueError, match=r'changes the geometry, so entity '
                       r'numbers stand for other entities in some values\. '
                       r"Pass a selection node or None: e.g. mk.sel.box\("
                       r"geom, 'boundary', x='W'\).* \(none is now\)"):
        mk.average(geom, 'boundary', 'T', 1, outer='all')
    fixed = mk.sel.box(geom, 'boundary', x=0.1, name='fixed')
    with pytest.raises(ValueError, match=r'"selections/fixed" is empty at '
                       r'outer=2 \(W=0.15 \(0.15 m\)\): .* fixed '
                       r"coordinates .* x='W'"):
        mk.average(geom, 'boundary', 'T', fixed, outer='all')
    assert mk.average(geom, 'boundary', 'T', fixed, unit='degC',
                      outer=1) == pytest.approx(20)
    # made after the solve: the values' geometries do not hold it
    later = mk.sel.result(geom, block, 'boundary')
    with pytest.raises(ValueError, match=r'is empty at outer=1 .* are empty '
                                         r'in values solved before they were '
                                         r'made: solve again'):
        mk.integral(geom, 'boundary', '1', later, outer='all')
    component = mk.component_of(geom).java.selection()
    explicit = component.create('ex1', 'Explicit')
    explicit.geom(geom.tag(), 2)
    explicit.set([1])
    nodes = {'explicit': model/'selections'/str(explicit.label())}
    on_explicit = mk.sel.box(geom, 'boundary', x=(-1, 1), name='on explicit')
    mk.set(on_explicit, inputent='selections', input=[nodes['explicit']])
    with pytest.raises(ValueError, match=r'may pick other entities for some '
                       r'values of dataset .*: "Explicit 1" is an explicit '
                       r'selection, a list of entity numbers\. Pass'):
        mk.average(geom, 'boundary', 'T', nodes['explicit'], outer='all')
    with pytest.raises(ValueError, match=r'"on explicit" takes "Explicit 1" '
                                         r'as input; "Explicit 1" is an '
                                         r'explicit selection'):
        mk.average(geom, 'boundary', 'T', on_explicit, outer='all')
    with pytest.raises(ValueError, match='changes the geometry, and mk.plot'):
        mk.plot(geom, 'T', tmp_path/'T.png', outer=1)


def test_geometry_sweep_rebuilt(widths):
    model, geom, block, faces = widths
    model.build(geom)
    hot = mk.sel.entities(geom, faces['hot'])
    # the geometry as built is that of W = 0.1, the first value
    assert mk.average(geom, 'boundary', 'T', hot, unit='degC', outer=1) == \
        pytest.approx(100)
    with pytest.raises(ValueError, match=r'Numbers work one value at a '
                       r'time, for outer=1 \(W=0.1 \(0.1 m\)\), solved on '
                       r'the geometry as built\.'):
        mk.average(geom, 'boundary', 'T', hot, outer='all')
    model.mesh()
    assert mk.average(geom, 'boundary', 'T', hot, unit='degC', outer=1) == \
        pytest.approx(100)
    assert mk.average(geom, 'boundary', 'T', faces['hot'], unit='degC',
                      outer='all') == pytest.approx([100, 100])


def test_part_without_physics(client):
    model = client.create('part')
    try:
        model.parameter('Th', '100[degC]')
        geom = mk.geometry(model, 3)
        first = mk.block(geom, (0.1, 0.05, 0.01))
        mk.block(geom, (0.1, 0.05, 0.01), (0.2, 0, 0))
        model.build(geom)
        steel = (model/'materials').create('Common', name='steel')
        for key, value in (('thermalconductivity', '45'),
                           ('density', '7850'), ('heatcapacity', '475')):
            (steel/'Basic').property(key, [value])
        heat = (model/'physics').create('HeatTransfer', geom)
        heat.java.selection().set([1])
        boundary = heat.create('TemperatureBoundary', 2)
        boundary.select(mk.sel.box(geom, 'boundary', x=0))
        boundary.property('T0', 'Th')
        (model/'meshes').create(geom)
        study = (model/'studies').create(name='part')
        study.create('Transient').property('tlist', '0 1')
        sweep(study, '100 200')
        model.solve()
        with pytest.raises(ValueError, match='covers part of the geometry '
                                             'only'):
            mk.average(geom, 'domain', 'T', 1, outer='all', step='last')
        solved = mk.sel.result(geom, first, 'domain')
        assert mk.maximum(geom, 'domain', 'T', solved, unit='degC',
                          outer='all', step='last') == \
            pytest.approx([100, 200])
        with pytest.raises(RuntimeError, match='no solution at outer=1'):
            mk.average(geom, 'domain', 'T', outer='all', step='last')
    finally:
        client.remove(model)


def test_axisymmetric_width_sweep(client):
    model = client.create('axisymmetric widths')
    try:
        model.parameter('W', '0.1[m]')
        geom = mk.geometry(model, 2)
        geom.java.axisymmetric(True)
        mk.rectangle(geom, ('W', 0.05), (0.01, 0))
        model.build(geom)
        hot = mk.sel.box(geom, 'boundary', x=0.01, name='inner')
        cold = mk.sel.box(geom, 'boundary', x='W+0.01', name='outer')
        steel = (model/'materials').create('Common', name='steel')
        for key, value in (('thermalconductivity', '45'),
                           ('density', '7850'), ('heatcapacity', '475')):
            (steel/'Basic').property(key, [value])
        heat = (model/'physics').create('HeatTransfer', geom)
        for face, temperature in ((hot, '100[degC]'), (cold, '20[degC]')):
            boundary = heat.create('TemperatureBoundary', 1)
            boundary.select(face)
            boundary.property('T0', temperature)
        (model/'meshes').create(geom)
        study = (model/'studies').create(name='widths')
        study.create('Stationary')
        sweep(study, '0.1 0.15', 'W', 'm')
        model.solve()
        # the revolved volume, pi*((W + 0.01)^2 - 0.01^2)*0.05
        assert mk.integral(geom, 'domain', '1', outer='all') == \
            pytest.approx([math.pi*0.012*0.05, math.pi*0.0255*0.05])
        assert mk.average(geom, 'boundary', 'T', cold, unit='degC',
                          outer='all') == pytest.approx([20, 20])
        with pytest.raises(ValueError, match='entity numbers stand for '
                                             'other entities'):
            mk.average(geom, 'boundary', 'T', 1, outer='all')
    finally:
        client.remove(model)


def test_cumulative_and_geometry_selections(client):
    model = client.create('mm widths')
    try:
        model.parameter('W', '100[mm]')
        geom = mk.geometry(model, 3, length_unit='mm')
        holes = mk.sel.cumulative(geom, 'holes', 'domain', create=True)
        mk.block(geom, ('W', 50, 10))
        mk.cylinder(geom, 5, 10, (20, 25, 0), contributeto=holes)
        inside = mk.sel.box(geom, 'boundary', x=0, where='geometry',
                            name='inside')
        model.build(geom)
        hot = mk.sel.box(geom, 'boundary', x=0, name='hot')
        cold = mk.sel.box(geom, 'boundary', x='W', name='cold')
        steel = (model/'materials').create('Common', name='steel')
        for key, value in (('thermalconductivity', '45'),
                           ('density', '7850'), ('heatcapacity', '475')):
            (steel/'Basic').property(key, [value])
        heat = (model/'physics').create('HeatTransfer', geom)
        for face, temperature in ((hot, '100[degC]'), (cold, '20[degC]')):
            boundary = heat.create('TemperatureBoundary', 2)
            boundary.select(face)
            boundary.property('T0', temperature)
        (model/'meshes').create(geom)
        study = (model/'studies').create(name='widths')
        study.create('Stationary')
        sweep(study, '100 150', 'W', 'mm')
        model.solve()
        assert outer_values(geom) == pytest.approx([{'W': 0.1},
                                                    {'W': 0.15}])
        assert mk.integral(geom, 'boundary', 'ht.ntflux', cold, unit='W',
                           outer='all') == pytest.approx([18, 12])
        assert mk.integral(geom, 'domain', '1', holes, outer='all') == \
            pytest.approx([math.pi*25e-6*0.01]*2, rel=0.01)
        with pytest.raises(ValueError, match=r'"inside" is made in the '
                                             r'geometry sequence; make it in '
                                             r'the component instead '
                                             r'\(where=None\)'):
            mk.integral(geom, 'boundary', '1', inside, outer='all')
    finally:
        client.remove(model)


############
# By value #
############

def test_by_value(swept):
    model, geom, study, faces = swept
    hot = faces['hot']

    def read(outer):
        return mk.average(geom, 'boundary', 'T', hot, unit='degC',
                          outer=outer, step='last')
    assert read({'Th': 473.15}) == pytest.approx(200)
    assert read({'Th': '200[degC]'}) == pytest.approx(200)
    assert read({'Th': '392[degF]'}) == pytest.approx(200)
    assert read([{'Th': '300[degC]'}, {'Th': 373.15}]) == \
        pytest.approx([300, 100])
    assert read(mk.outer_values(geom)) == pytest.approx([100, 200, 300])
    assert read(mk.outer_values(geom)[1]) == pytest.approx(200)
    assert step_values(geom, outer={'Th': '300[degC]'})['t'] == \
        pytest.approx([0, 1, 2])
    for given, message in (
            ('200[kg]', r"COMSOL cannot read '200\[kg\]' as a value of Th "
                        r'in K: .*Inconsistent unit'),
            ('200[xyz]', 'Unknown unit'),
            ('473.15', r"cannot read '473.15'.* Give the unit in brackets")):
        with pytest.raises(ValueError, match=message):
            read({'Th': given})
    with pytest.raises(ValueError, match=r"has no value with Th=200; it has "
                       r'1: Th=373.15 \(100 degC\);.* Numbers are in SI '
                       r"units; give the unit as a string instead, e.g. "
                       r"'200\[degC\]'"):
        read({'Th': 200})
    with pytest.raises(ValueError, match=r"has no outer parameter 'T'; its "
                                         r'outer parameters are Th\.'):
        read({'T': 1})
    with pytest.raises(ValueError, match='holds t as steps of each outer '
                                         'value'):
        read({'t': 1})
    with pytest.raises(TypeError, match='names no parameter'):
        read({})
    with pytest.raises(TypeError, match='mixes numbers and values by name'):
        read([1, {'Th': 373.15}])
    with pytest.raises(TypeError, match='takes a number in SI units'):
        read({'Th': True})
    with pytest.raises(TypeError, match=r"or values by name such as "):
        read(1.5)


def test_two_parameters_by_value(fresh):
    model, geom = fresh()
    model.parameter('k', '45[W/(m*K)]')
    (model/'materials'/'steel'/'Basic').property('thermalconductivity',
                                                ['k'])
    study = (model/'studies').create(name='both')
    study.create('Transient').property('tlist', '0 1')
    both = study.create('Parametric')
    both.property('pname', ['Th', 'k'])
    both.property('plistarr', ['100 200', '10 90'])
    both.property('punit', ['degC', 'W/(m*K)'])
    both.java.set('sweeptype', 'filled')
    model.solve()
    found = outer_values(geom)
    assert len(found) == 4
    second = {'Th': '100[degC]', 'k': 90}
    k = found.index(next(row for row in found
                         if row['k'] == pytest.approx(90)
                         and row['Th'] == pytest.approx(373.15))) + 1
    assert mk.integral(geom, 'domain', '1', outer=second, step='last') == \
        pytest.approx(mk.integral(geom, 'domain', '1', outer=k,
                                  step='last'))
    with pytest.raises(ValueError, match=r"Th='100\[degC\]' fits several "
                                         r'values of .*give more parameters'):
        mk.integral(geom, 'domain', '1', outer={'Th': '100[degC]'},
                    step='last')


def test_repeated_value(fresh):
    model, geom = fresh()
    transient(model, values='100 100')
    with pytest.raises(ValueError, match=r'has Th=373.15 more than once '
                                         r'\(outer=1 and 2\); pass outer= '
                                         r'one of these numbers'):
        mk.average(geom, 'domain', 'T', outer={'Th': 373.15}, step='last')


def test_width_by_value(widths):
    model, geom, block, faces = widths
    assert mk.integral(geom, 'domain', '1', outer={'W': '150[mm]'}) == \
        pytest.approx(7.5e-5)
    assert mk.integral(geom, 'domain', '1', outer={'W': 0.1}) == \
        pytest.approx(5e-5)


def test_last_value_kept(fresh, tmp_path):
    model, geom = fresh()
    study = (model/'studies').create(name='last')
    study.create('Transient').property('tlist', '0 1')
    sweep(study, '100 200 300').java.set('keepsol', 'last')
    model.solve()
    assert outer_values(geom) == pytest.approx([{'Th': 300 + KELVIN}])
    hot = mk.sel.box(geom, 'boundary', x=0)
    # one value: outer may be left out
    assert mk.average(geom, 'boundary', 'T', hot, unit='degC',
                      step='last') == pytest.approx(300)
    assert mk.plot(geom, 'T', tmp_path/'T.png', outer='last',
                   step='last').exists()


def test_last_value_kept_of_geometry(fresh):
    model, geom = fresh()
    model.parameter('W', '0.1[m]')
    (geom/'Block 1').property('size', ['W', '0.05', '0.01'])
    model.build(geom)
    study = (model/'studies').create(name='last')
    study.create('Stationary')
    sweep(study, '0.1 0.15', 'W', 'm').java.set('keepsol', 'last')
    model.solve()
    with pytest.raises(RuntimeError, match=r'is not built; set the '
                       r"parameters to the sweep's value \(W=0.15 \(0.15 "
                       r'm\), see mk.outer_values\(geom\)\)'):
        mk.average(geom, 'domain', 'T', 1)
    model.build(geom)
    with pytest.raises(RuntimeError, match='differs from the one dataset .* '
                                           'was solved on.* must keep all '
                                           'solutions to be read'):
        mk.average(geom, 'domain', 'T', 1)
