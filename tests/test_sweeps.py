"""
Tests of parametric sweeps that COMSOL stores as an outer loop: which
datasets the helpers take or refuse, and `mk.outer_values`.

The plate of test_results, 0.1 x 0.05 x 0.01 m, held at Th at x = 0 and
20 degC at x = 0.1, swept over Th around a short time-dependent study.
"""
import pytest

import mphkit as mk
from test_results import leftovers, plate

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


@pytest.fixture(scope='module')
def swept(client):
    """The plate swept over Th = 100, 200, 300 degC, time-dependent."""
    model, geom, faces = plate(client, 'swept plate', study=False)
    study = transient(model)
    yield model, geom, study
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
    model, geom, study = swept
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
    sweep(study, '100 200')
    model.solve()
    with pytest.raises(ValueError, match=r'holds its sweep over Th as steps, '
                                         r'not as outer values: pass step='):
        outer_values(geom)


############
# Datasets #
############

def test_copy(swept):
    model, geom, study = swept
    with pytest.raises(ValueError, match=r'Dataset "sweep//Solution 1" '
                       r"\(dset1\) holds only the last value of a parametric "
                       r"sweep; pass dataset='dset2', .* with outer='last' "
                       r'\(Th=573.15 \(300 degC\)\)'):
        outer_values(geom, dataset='dset1')
    with pytest.raises(ValueError, match='holds only the last value'):
        mk.average(geom, 'domain', 'T', dataset='dset1')


def test_child(swept):
    model, geom, study = swept
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
