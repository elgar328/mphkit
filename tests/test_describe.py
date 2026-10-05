"""
Checks mk.describe: the plain functions without COMSOL, and the summary of
models built here (defaults, selections by location, hidden nodes, meshes,
studies, solvers) together with leaving the model as it was.
"""
import json
import math
import re
import textwrap
from pathlib import Path

import pytest

import mphkit as mk
from conftest import model_state, read
from mphkit import _comsol, _describe

root = Path(__file__).parents[1]


###################
# Plain functions #
###################

def test_json_value():
    import numpy
    assert _describe.json_value(math.nan) == 'NaN'
    assert _describe.json_value(-math.inf) == '-Infinity'
    assert _describe.json_value((1.0, math.inf)) == [1.0, 'Infinity']
    assert _describe.json_value(numpy.array([[1, 2]])) == [[1, 2]]
    assert _describe.json_value('Th') == 'Th'
    value = {'a': _describe.json_value([math.nan])}
    assert json.loads(json.dumps(value)) == value


def test_differs():
    properties, defaults, unknown = _describe.differs(
        {'T0': 'Th', 'h': '0', 'new': 1}, {'T0': '293.15[K]', 'h': '0'})
    assert properties == {'T0': 'Th', 'new': 1}
    assert defaults == {'T0': '293.15[K]'}
    assert unknown == ['new']


def test_step_map():
    own = ['ht', 'on', 'ec', 'off', 'frame:spatial1', 'on']
    default = ['ht', 'on', 'ec', 'on', 'frame:spatial1', 'on', 'comp1', 'on']
    # an older model lacks the component's pair: no difference
    assert _describe.step_map(own, default) == ({'ec': 'off'}, {'ec': 'on'})
    assert _describe.step_map(default, default) == ({}, {})
    assert _describe.pairs_of(['a', 'b', 'c']) is None
    assert _describe.pairs_of(['a', '1', 'a', '2']) is None
    assert _describe.pairs_of(['solid', 'solid2/bndl1']) == \
        {'solid': 'solid2/bndl1'}


def test_owner():
    components = {'comp1'}
    assert _describe.owner('root', components) is None
    assert _describe.owner('root.comp1', components) == 'comp1'
    # made by a physics interface: builder_* and Derived operators
    assert _describe.owner('root.comp1.ht', components) is False
    # coordinate systems and materials end with their own tag
    assert _describe.owner('root.comp1.sys1', components, 'sys1') == 'comp1'
    assert _describe.owner('root.mat9', components, 'mat9') is None
    # a material inside a switch, and a component a layered material made
    assert _describe.owner('root.comp1.sw1.mat9', components, 'mat9') \
        is False
    assert _describe.owner('root.mat4_xdim', components) is False
    assert _describe.hidden_component('mat4_xdim', {'mat4'})
    assert not _describe.hidden_component('comp_xdim', {'mat4'})


def test_appearance():
    assert _describe.appearance('family')
    assert _describe.appearance('customspecular')
    assert _describe.appearance('noisefreq')
    assert not _describe.appearance('thickness')
    assert not _describe.appearance('sys')


def test_solver_changes():
    model_nodes = {
        'st1': ('StudyStep', 'Compile Equations', {'study': 'std1'}),
        'v1': ('Variables', 'Dependent Variables 1', {'initsol': 'sol1'}),
        's1': ('Stationary', 'Stationary Solver 1', {'stol': 1e-6}),
        's1/su1': ('StoreSolution', 'Store 1', {'sol': 'sol2'}),
        's1/se1': ('Segregated', 'Segregated 1', {}),
        's1/l1': ('Lists', 'Lists 1', {'both': ['sol1', 'x'],
                                       'stores': ['sol2', 'y']}),
    }
    automatic = {
        'st1': ('StudyStep', 'Compile Equations', {'study': 'std1'}),
        'v1': ('Variables', 'Dependent Variables 1', {'initsol': 'mkdesc1'}),
        's1': ('Stationary', 'Stationary Solver 1', {'stol': 1e-3}),
        's1/su1': ('StoreSolution', 'Store 1', {'sol': 'sol3'}),
        's1/fc1': ('FullyCoupled', 'Fully Coupled 1', {}),
        # the temporary tag and other solution tags inside lists too
        's1/l1': ('Lists', 'Lists 1', {'both': ['mkdesc1', 'x'],
                                       'stores': ['sol3', 'y']}),
    }
    found = _describe.solver_changes(model_nodes, automatic,
                                     {'mkdesc1': 'sol1'},
                                     {'sol1', 'sol2', 'sol3', 'mkdesc1'})
    assert found == [
        {'path': 's1', 'labels': 'Stationary Solver 1', 'type': 'Stationary',
         'change': 'property', 'properties': {'stol': [1e-6, 1e-3]}},
        {'path': 's1/se1', 'labels': 'Segregated 1', 'type': 'Segregated',
         'change': 'only_in_model'},
        {'path': 's1/fc1', 'labels': 'Fully Coupled 1',
         'type': 'FullyCoupled', 'change': 'only_in_automatic'}]


class Fake:
    """A node with string properties and no selection or subnodes."""

    def __init__(self, kind, **values):
        self.kind, self.values = kind, values

    def tag(self):
        return 'f1'

    def getType(self):
        return self.kind

    def label(self):
        return self.kind

    def isActive(self):
        return True

    def properties(self):
        return list(self.values)

    def getValueType(self, name):
        return 'StringArray' if isinstance(self.values[name], list) \
            else 'String'

    def getStringArray(self, name):
        return self.values[name]

    def getString(self, name):
        return self.values[name]

    def selection(self, name=None):
        raise RuntimeError('no selection')

    def getExtraSelectionNames(self):
        return []


def reader(context=''):
    """A reader without a model, for the parts that need none."""
    found = object.__new__(_describe._Reader)
    found.notes, found.context, found.java = [], context, None

    class Geometry:
        def getSDim(self):
            return 3

    found.geometries = {'geom1': Geometry()}
    return found


def test_step_maps_on_steps_only():
    own = Fake('Stationary', activate=['ht', 'on', 'ec', 'off'])
    new = Fake('Stationary', activate=['ht', 'on', 'ec', 'on'])
    other = reader().node(own, new, 'f1')
    assert other['properties'] == {'activate': ['ht', 'on', 'ec', 'off']}
    step = reader('study').node(own, new, 'std1/f1')
    assert step['properties'] == {'activate': {'ec': 'off'}}
    assert step['defaults'] == {'activate': {'ec': 'on'}}
    unknown = reader('study').node(own, None, 'std1/f1')
    assert unknown['properties'] == {'activate': {'ht': 'on', 'ec': 'off'}}


def test_levels():
    class Selection:
        def geom(self):
            return 'geom1'

    selection = Selection()
    mesh, other = reader('mesh'), reader()
    assert mesh.described(selection, [], 'p') == {'level': 'remaining'}
    assert mesh.described(selection, [0, 1, 2, 3], 'p') == \
        {'level': 'remaining'}
    assert other.described(selection, [0, 1, 2, 3], 'p') == \
        {'level': 'geometry'}
    assert other.described(selection, [], 'p') == {'level': 'none'}
    several = other.described(selection, [2, 3], 'p')
    assert several == {'level': 'several', 'levels': ['boundary', 'domain'],
                       'entities': 'unknown'}
    assert other.notes[0]['kind'] == 'levels_unknown'


def test_set_back_step_by_step():
    # a step that fails is noted, the others still run
    class List:
        def __init__(self, tags=()):
            self.items = list(tags)

        def tags(self):
            return list(self.items)

        def remove(self, tag):
            if tag == 'stuck':
                raise RuntimeError('Object cannot be removed.')
            self.items.remove(tag)

    class Model:
        def __init__(self):
            self.lists = {name: List() for name in
                          ('sol', 'study', 'mesh', 'variable', 'cpl',
                           'func')}
            self.lists['variable'] = List(['var1'])

        def __getattr__(self, name):
            return lambda: self.lists[name]

    model = Model()
    with _comsol.compiled_traces_removed(model) as changed:
        model.lists['variable'].items += ['stuck', 'iexpr1']
        model.lists['cpl'].items += ['maxOp1', 'builder_integrate9']
    assert model.lists['variable'].items == ['var1', 'stuck']
    assert model.lists['cpl'].items == ['builder_integrate9']
    assert changed == ["could not set back the new node 'stuck': "
                       'Object cannot be removed.']


##########
# Models #
##########

def plate(client, name='plate'):
    """
    A 100 × 50 × 10 mm plate with a hole: temperature on the face at x = 0
    and a periodic condition on it and the face at x = 100 (which
    overrides the temperature there), convection on top, an integration
    operator, steel, a mesh and a stationary study.
    """
    model = client.create(name)
    model.parameter('Th', '100[degC]')
    model.parameter('h0', '10[W/(m^2*K)]')
    geom = mk.geometry(model, 3, length_unit='mm')
    block = mk.block(geom, (100, 50, 10))
    hole = mk.cylinder(geom, 5, 10, (30, 25, 0))
    mk.difference(geom, block, [hole])
    model.build(geom)
    left = mk.sel.box(geom, 'boundary', x=0, name='left')
    right = mk.sel.box(geom, 'boundary', x=100)
    top = mk.sel.box(geom, 'boundary', z=10)
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    hot = heat.create('TemperatureBoundary', 2, name='hot')
    hot.select(left)
    hot.property('T0', 'Th')
    cooling = heat.create('HeatFluxBoundary', 2, name='cooling')
    cooling.select(top)
    cooling.property('HeatFluxType', 'ConvectiveHeatFlux')
    cooling.property('h', 'h0')
    periodic = heat.create('PeriodicHeat', 2, name='periodic')
    periodic.java.selection().set(
        mk.sel.entities(geom, left) + mk.sel.entities(geom, right))
    component = mk.component_of(geom).java
    operator = component.cpl().create('intop1', 'Integration')
    operator.selection().geom(str(geom.java.tag()), 2)
    operator.selection().set(mk.sel.entities(geom, top))
    mk.material(geom, 'Structural steel')
    mesh = (model/'meshes').create(geom, name='mesh')
    mesh.create('FreeTet')
    model.mesh()
    (model/'studies').create(name='static').create('Stationary')
    return model, geom


@pytest.fixture(scope='module')
def solved(client):
    """The plate, solved, and its description."""
    model, geom = plate(client, 'described')
    model.solve()
    yield model, geom, mk.describe(model, solver=True)
    client.remove(model)


def features(described, type, component=0, physics=0):
    nodes = described['components'][component]['physics'][physics]
    return [n for n in nodes['features'] if n['type'] == type]


def find(described, type, **kwargs):
    [node] = features(described, type, **kwargs)
    return node


def blocks(model, count=2):
    geom = mk.geometry(model, 3)
    for n in range(count):
        mk.block(geom, (1, 1, 1), (n, 0, 0))
    model.build(geom)
    return geom


def test_plate(solved):
    model, geom, described = solved
    hot = find(described, 'TemperatureBoundary')
    assert hot['properties'] == {'T0': 'Th'}
    assert hot['defaults'] == {'T0': '293.15[K]'}
    assert hot['path'] == 'comp1/ht/temp1'
    [face] = hot['selection']['entities']
    assert face['x'] == [0, 0]
    assert face['y'] == pytest.approx([0, 50])
    assert face['size'] == pytest.approx(500)
    assert hot['selection']['level'] == 'boundary'
    assert hot['selection']['named'] == 'left'
    cooling = find(described, 'HeatFluxBoundary')
    assert cooling['properties'] == {'HeatFluxType': 'ConvectiveHeatFlux',
                                     'h': 'h0'}
    for default in ('SolidHeatTransferModel', 'init', 'ThermalInsulation'):
        assert find(described, default)['properties'] == {}
    assert described['parameters']['Th'] == {'expression': '100[degC]',
                                             'value': 373.15}
    [geometry] = described['components'][0]['geometries']
    assert geometry['length_unit'] == 'mm'
    assert len(geometry['entities']['boundary']) == \
        geom.java.getNBoundaries()
    # solved: the first compile set solnum and notsolnum to 'auto'
    [step] = described['studies'][0]['steps']
    assert step['properties'] == {}
    assert described['mphkit'] == mk.__version__
    assert described['comsol'].startswith('COMSOL')
    assert described['saved_with'].startswith('COMSOL')
    assert described['format'] == 1
    assert described['notes'] == []


def test_structure(solved):
    model, geom, described = solved

    def nodes(items):
        for item in items:
            yield item
            yield from nodes(item.get('features', []))

    component = described['components'][0]
    every = list(nodes(component['physics'][0]['features'])) + \
        list(nodes(component['multiphysics'])) + \
        list(nodes(component['meshes'][0]['features'])) + \
        list(nodes(described['studies'][0]['steps'])) + \
        list(nodes(described['couplings']))
    for node in every:
        for key in ('tag', 'path', 'type', 'label', 'active', 'properties',
                    'defaults', 'selection', 'selections', 'features'):
            assert key in node, (key, node['path'])
        if node.get('all_properties'):
            assert node['defaults'] == {}
        else:
            assert set(node['defaults']) == \
                set(node['properties']) - set(node.get('unknown_defaults',
                                                       []))


def test_json(solved):
    model, geom, described = solved
    assert json.loads(json.dumps(described)) == described


def test_hidden_nodes(solved):
    # a physics interface makes builder_* operators and functions, a
    # periodic condition 'Derived' operators: COMSOL's own
    model, geom, described = solved
    assert 'builder_' not in json.dumps(described)
    assert [c['tag'] for c in described['couplings']] == ['intop1']
    assert described['couplings'][0]['component'] == 'comp1'


def test_applied(solved):
    # the periodic condition after it overrides the temperature at x = 0
    model, geom, described = solved
    hot = find(described, 'TemperatureBoundary')
    assert hot['selection']['applied'] == []
    periodic = find(described, 'PeriodicHeat')
    assert 'destinationDomains' in periodic['selections']
    assert 'applied' not in periodic['selection']
    # insulation is left on the 10 faces less x = 0, x = 100 and the top
    insulation = find(described, 'ThermalInsulation')
    assert insulation['selection']['entities'] == 'all'
    applied = insulation['selection']['applied']
    assert len(applied) == 7
    assert not any(face['x'] in ([0, 0], [100, 100]) for face in applied)
    assert not any(face['z'] == [10, 10] for face in applied)


def test_leaves_nothing(client, solved):
    model, geom, described = solved
    before = model_state(model)
    assert mk.describe(model, solver=True) == described
    assert model_state(model) == before
    # without solver=True nothing is compiled
    plain = mk.describe(model)
    assert model_state(model) == before
    assert plain['studies'][0]['solver']['status'] == 'not compared'
    assert plain['notes'] == []


def test_two_studies_and_sweep(client):
    model = client.create('studies')
    try:
        model.parameter('V', '1[V]')
        geom = blocks(model, 1)
        current = (model/'physics').create('ConductiveMedia', geom)
        potential = current.create('ElectricPotential', 2)
        potential.select(mk.sel.box(geom, 'boundary', x=0))
        potential.property('V0', 'V')
        current.create('Ground', 2).select(mk.sel.box(geom, 'boundary', x=1))
        mk.material(geom, 'Copper')
        (model/'meshes').create(geom).create('FreeTet')
        model.mesh()
        first = (model/'studies').create(name='first')
        first.create('Stationary')
        sweep = first.create('Parametric')
        mk.set(sweep, pname=['V'], plistarr=['1 2'], punit=['V'])
        (model/'studies').create(name='second').create('Stationary')
        model.solve()
        before = model_state(model)
        described = mk.describe(model, solver=True)
        assert model_state(model) == before
        assert [s['solver']['status'] for s in described['studies']] == \
            ['compared', 'compared']
        assert [s['solver']['changes'] for s in described['studies']] == \
            [[], []]
    finally:
        client.remove(model)


def test_not_built(model):
    geom = blocks(model)
    geom.java.feature('blk1').set('size', ['2', '1', '1'])
    with pytest.raises(RuntimeError, match='run model.build'):
        mk.describe(model)


def test_defaults_unknown(client, solved, monkeypatch):
    # a node that cannot be made for defaults lists all its properties;
    # the ones after it are fine, and nothing is left behind
    model, geom, described = solved
    made = []
    create = _describe._create_physics

    def failing(container, java):
        if str(java.tag()) != 'temp1':
            return create(container, java)
        # fails half-made, as some features do
        made.append(container)
        container.create('temp1', 'TemperatureBoundary', 2)
        raise RuntimeError('made to fail')

    monkeypatch.setattr(_describe, '_create_physics', failing)
    before = model_state(model)
    found = mk.describe(model)
    assert model_state(model) == before
    assert made
    hot = find(found, 'TemperatureBoundary')
    assert hot['all_properties'] is True
    assert hot['properties']['T0'] == 'Th'
    assert hot['defaults'] == {}
    assert any(note['path'] == 'comp1/ht/temp1'
               and note['kind'] == 'defaults_unknown'
               for note in found['notes'])
    assert find(found, 'HeatFluxBoundary') == \
        find(described, 'HeatFluxBoundary')


def test_solver_failing(client, solved, monkeypatch):
    model, geom, described = solved

    read = _describe._Reader.solver_nodes

    def failing(self, sequence, *args, **kwargs):
        if str(sequence.tag()).startswith(_describe.SCRATCH):
            raise RuntimeError('made to fail')
        return read(self, sequence, *args, **kwargs)

    monkeypatch.setattr(_describe._Reader, 'solver_nodes', failing)
    before = model_state(model)
    found = mk.describe(model, solver=True)
    assert model_state(model) == before
    solver = found['studies'][0]['solver']
    assert solver['status'] == 'not compared'
    assert solver['sequence'] == 'sol1'
    assert 'made to fail' in solver['reason']


def test_solver(model):
    geom = blocks(model, 1)
    heat = (model/'physics').create('HeatTransfer', geom)
    heat.create('TemperatureBoundary', 2).select(
        mk.sel.box(geom, 'boundary', x=0))
    mk.material(geom, 'Copper')
    (model/'meshes').create(geom).create('FreeTet')
    model.mesh()
    study = (model/'studies').create(name='static')
    study.create('Stationary')
    solver = mk.describe(model)['studies'][0]['solver']
    assert solver == {'status': 'automatic', 'sequence': None}
    model.solve()
    solver = mk.describe(model, solver=True)['studies'][0]['solver']
    assert solver == {'status': 'compared', 'sequence': 'sol1',
                      'changes': []}
    model.java.sol('sol1').feature('s1').set('stol', '1e-6')
    [change] = mk.describe(model, solver=True)['studies'][0]['solver'][
        'changes']
    assert (change['path'], change['change']) == ('s1', 'property')
    # a changed solver setting also marks the node as set by the user
    assert change['properties']['stol'][0] == '1e-6' or \
        change['properties']['stol'][0] == 1e-6
    assert sorted(change['properties']) == ['control', 'stol'], \
        change['properties']
    assert mk.describe(model)['studies'][0]['solver'] == {
        'status': 'not compared', 'sequence': 'sol1',
        'reason': 'not asked for: mk.describe(model, solver=True) compares '
                  'it'}
    # without a built mesh, a temporary sequence would build it silently
    mesh = model.java.mesh(model.java.mesh().tags()[0])
    mesh.clearMesh()
    solver = mk.describe(model, solver=True)['studies'][0]['solver']
    assert solver['status'] == 'not compared'
    assert 'run model.mesh()' in solver['reason']
    assert mesh.getNumElem() == 0
    # a disabled study is not compared either (COMSOL 6.4 refuses to
    # disable a study through the API: "Object cannot be disabled")
    model.mesh()

    class Disabled:
        def isActive(self):
            return False

    sequences = [str(t) for t in model.java.sol().tags()]
    reader = _describe._Reader(model)
    assert reader.solver(Disabled(), 'std1', True) == {
        'status': 'not compared', 'sequence': 'sol1',
        'reason': 'study disabled'}
    assert [str(t) for t in model.java.sol().tags()] == sequences


def test_physics_on_part(model):
    # default features select all boundaries also with physics on one
    # domain of two; where they apply is what is left
    geom = blocks(model)
    heat = (model/'physics').create('HeatTransfer', geom)
    heat.java.selection().set([1])
    described = mk.describe(model)
    physics = described['components'][0]['physics'][0]
    assert len(physics['selection']['entities']) == 1
    insulation = find(described, 'ThermalInsulation')
    assert insulation['selection']['entities'] == 'all'
    assert len(insulation['selection']['applied']) == 6


def test_derived_named_selection(model):
    # a derived named selection: inputEntities() gives its input (the
    # domains), not the boundaries it selects
    geom = blocks(model, 3)
    right = mk.sel.box(geom, 'domain', x=(1, 3))
    around = mk.sel.adjacent(geom, right, name='around')
    heat = (model/'physics').create('HeatTransfer', geom)
    hot = heat.create('TemperatureBoundary', 2)
    hot.java.selection().named(str(around.tag()))
    selected = find(mk.describe(model), 'TemperatureBoundary')['selection']
    assert selected['named'] == 'around'
    assert len(selected['entities']) == len(mk.sel.entities(geom, around))
    assert 'applied' not in selected
    # the outer faces of domains 2 and 3: not the face between them
    assert not any(place['x'] == [2, 2] for place in selected['entities'])


def test_activate(model):
    geom = blocks(model, 1)
    (model/'physics').create('HeatTransfer', geom)
    (model/'physics').create('ConductiveMedia', geom)
    step = (model/'studies').create().create('Stationary')
    activate = [str(v) for v in step.java.getStringArray('activate')]
    activate[activate.index('ec') + 1] = 'off'
    step.java.set('activate', activate)
    [described] = mk.describe(model)['studies'][0]['steps']
    assert described['properties'] == {'activate': {'ec': 'off'}}
    assert described['defaults'] == {'activate': {'ec': 'on'}}


def test_identifier(model):
    # a renamed interface has other default expressions ('heat.helem')
    geom = blocks(model, 1)
    heat = (model/'physics').create('HeatTransfer', geom)
    heat.java.identifier('heat')
    [physics] = mk.describe(model)['components'][0]['physics']
    assert physics['identifier'] == 'heat'
    assert physics['settings'] == {}
    assert all(n['properties'] == {} for n in physics['features'])


def test_mesh(model):
    geom = blocks(model)
    mesh = (model/'meshes').create(geom)
    size = mesh.create('Size')
    size.java.selection().geom(str(geom.java.tag()), 2)
    size.java.selection().set([1])
    mk.set(size, hmax=0.15)        # 0.2 is the default here
    first = mesh.create('FreeTet')
    first.java.selection().geom(str(geom.java.tag()), 3)
    first.java.selection().set([1])
    rest = mesh.create('FreeTet')
    rest.java.selection().remaining()
    model.mesh()
    [described] = mk.describe(model)['components'][0]['meshes']
    assert described['automatic'] is False
    default, size, first, rest = described['features']
    assert default['type'] == 'MeshSizeDefault'
    assert default['properties'] == {}
    assert default['selection'] is None
    assert size['properties'] == {'hmax': 0.15, 'custom': 'on'}
    assert first['selection']['level'] == 'domain'
    assert len(first['selection']['entities']) == 1
    assert rest['selection'] == {'level': 'remaining'}


def test_whole_geometry(model):
    # the whole geometry and what is left read alike once meshed
    geom = blocks(model, 1)
    left = (model/'meshes').create(geom)
    left.create('FreeTet')
    whole = (model/'meshes').create(geom)
    whole.create('FreeTet').java.selection().geom(str(geom.java.tag()))
    for meshed in (False, True):
        if meshed:
            model.mesh()
        for described in mk.describe(model)['components'][0]['meshes']:
            assert described['features'][1]['selection'] == \
                {'level': 'remaining'}


@pytest.mark.parametrize('kind', ['1D', '2D', 'axisymmetric', 'shell'])
def test_other_geometries(model, kind):
    if kind == '1D':
        geom = mk.geometry(model, 1)
        mk.interval(geom, [0, 2])
        axes = 1
    elif kind == 'shell':
        geom = mk.geometry(model, 3)
        plane = mk.workplane(geom)
        mk.rectangle(plane, (2, 1))
        axes = 3
    else:
        geom = mk.geometry(model, 2, axisymmetric=kind == 'axisymmetric')
        mk.rectangle(geom, (2, 1))
        axes = 2
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    (model/'meshes').create(geom)
    described = mk.describe(model)
    [geometry] = described['components'][0]['geometries']
    assert geometry['axisymmetric'] == (kind == 'axisymmetric')
    for place in geometry['entities']['domain']:
        assert len([k for k in place if k in 'xyz']) == axes
    assert described['components'][0]['meshes'][0]['features'][0][
        'properties'] == {}
    assert heat.java.tag() == described['components'][0]['physics'][0][
        'tag']


def test_derived_variables(solved):
    # solving makes 'Derived Variables' nodes, gone once saved and loaded
    model, geom, described = solved
    tags = [str(t) for t in model.java.variable().tags()]
    assert any(tag.startswith('iexpr') for tag in tags)
    assert described['variables'] == []


def test_saved(client, solved, tmp_path):
    model, geom, described = solved
    model.save(tmp_path/'saved.mph')
    loaded = client.load(tmp_path/'saved.mph')
    try:
        # not solved since loaded: comparing the solvers compiles the
        # equations for the first time
        before = model_state(loaded)
        assert mk.describe(loaded, solver=True) == described
        assert model_state(loaded) == before
    finally:
        client.remove(loaded)


def test_materials(model):
    geom = blocks(model, 3)
    mk.material(geom, 'Air', [1])
    java = model.java
    used = java.material().create('gm1', 'Common', '')
    used.propertyGroup('def').set('thermalconductivity', ['5'])
    component = mk.component_of(geom).java
    component.material().create('lnk1', 'Link')
    component.material('lnk1').set('link', 'gm1')
    component.material('lnk1').selection().set([2])
    switch = component.material().create('sw1', 'Switch')
    switch.feature().create('mat9', 'Common')
    switch.feature('mat9').propertyGroup('def').set('density', '1')
    switch.selection().set([3])
    materials = {m['tag']: m for m in mk.describe(model)['materials']}
    assert sorted(materials) == ['gm1', 'lnk1', 'mat1', 'sw1']
    air = materials['mat1']
    assert air['component'] == 'comp1'
    assert 'k' in air['groups']['def']['functions']
    assert air['groups']['def']['properties']['thermalconductivity'] == \
        ['k(T)']
    assert not any(_describe.appearance(n) for n in air['properties'])
    assert 'component' not in materials['gm1']
    assert materials['gm1']['groups']['def']['properties'] == \
        {'thermalconductivity': ['5']}
    assert materials['lnk1']['properties']['link'] == 'gm1'
    [member] = materials['sw1']['features']
    assert member['tag'] == 'mat9'
    assert member['groups']['def']['properties'] == {'density': '1'}


def test_definitions(model):
    model.parameter('a', '1[m]')
    java = model.java
    java.param().group().create('par2')
    java.param('par2').set('b', '2[mm]')
    model.parameter('z', '1+2*i')
    geom = blocks(model, 1)
    component = mk.component_of(geom).java
    java.variable().create('gvar').set('G', '2')
    component.variable().create('cvar').set('C', 'x')
    java.func().create('gan', 'Analytic')
    component.func().create('can', 'Analytic')
    component.common().create('dd1', 'DeformingDomain')
    (model/'physics').create('HeatTransfer', geom)
    described = mk.describe(model)
    assert described['parameters']['b'] == {'expression': '2[mm]',
                                            'value': 0.002}
    assert described['parameters']['z']['value'] is None
    variables = {v['tag']: v for v in described['variables']}
    assert sorted(variables) == ['cvar', 'gvar']
    assert 'component' not in variables['gvar']
    assert variables['gvar']['selection'] == {'level': 'global'}
    assert variables['cvar']['component'] == 'comp1'
    assert variables['cvar']['selection'] == {'level': 'global'}
    assert variables['cvar']['variables'] == {'C': 'x'}
    functions = {f['tag']: f for f in described['functions']}
    assert sorted(functions) == ['can', 'gan']
    assert functions['can']['component'] == 'comp1'
    assert 'component' not in functions['gan']
    # the model's list of definitions holds the component's too
    [deforming] = [d for d in described['definitions']
                   if d['type'] == 'DeformingDomain']
    assert deforming['component'] == 'comp1'
    assert deforming['path'] == 'comp1/dd1'


def test_multiphysics_and_coordinates(model):
    geom = blocks(model, 1)
    (model/'physics').create('HeatTransfer', geom)
    (model/'physics').create('ConductiveMedia', geom)
    component = mk.component_of(geom).java
    heating = component.multiphysics().create(
        'emh1', 'ElectromagneticHeating', str(geom.java.tag()), 3)
    heating.selection().all()
    rotated = component.coordSystem().create('sys2', 'Rotated')
    rotated.set('angle', ['0', '0', '30[deg]'])
    described = mk.describe(model)
    [coupling] = described['components'][0]['multiphysics']
    assert coupling['type'] == 'ElectromagneticHeating'
    assert coupling['selection'] == {'level': 'domain', 'entities': 'all'}
    assert 'all_properties' not in coupling
    systems = {c['tag']: c for c in described['coordinates']}
    assert systems['sys2']['properties'] == {'angle': ['0', '0', '30[deg]']}
    assert systems['sys2']['component'] == 'comp1'


def test_model_changed(solved, monkeypatch):
    # what COMSOL changes for good goes to the notes
    model, geom, described = solved
    calls = []
    original = _comsol._sequence_nodes

    def nodes(java):
        calls.append(1)
        found = original(java)
        return found + ['sol1/new'] if len(calls) > 1 else found

    monkeypatch.setattr(_comsol, '_sequence_nodes', nodes)
    found = mk.describe(model, solver=True)
    [note] = [n for n in found['notes'] if n['kind'] == 'model_changed']
    assert 'sol1/new' in note['message']


def test_pairs(model):
    geom = blocks(model)
    model.java.component('comp1').geom('geom1').feature('fin').set(
        'action', 'assembly')
    model.build(geom)
    [pair] = mk.describe(model)['components'][0]['pairs']
    assert pair['type'] == 'Identity'
    assert pair['active'] is True
    [source] = pair['source']['entities']
    [destination] = pair['destination']['entities']
    # in an assembly, both faces of a pair have the same place
    assert source['x'] == destination['x'] == [1, 1]


def test_readme(solved):
    model, geom, described = solved
    readme = read(root/'README.md')
    code = re.search(r'A model rebuilt by a script.*?```python\n(.*?)```',
                     readme, re.S).group(1)
    assert len(code.splitlines()) == 2, 'update the checks with the README'
    namespace = {'mk': mk, 'old': model, 'model': model}
    exec(code, namespace)
    assert namespace['old_settings'] == namespace['new_settings'] == \
        mk.describe(model)


def test_docstring_example(solved, monkeypatch, tmp_path):
    # the example at the top of help(mk.describe) runs as written
    model, geom, described = solved
    code = re.search(r'```python\n(.*?)```', mk.describe.__doc__,
                     re.S).group(1)
    code = textwrap.dedent(code)    # Python 3.13 dedents docstrings
    monkeypatch.chdir(tmp_path)
    namespace = {'mk': mk, 'json': json, 'model': model}
    exec(code, namespace)
    assert namespace['hot']['tag'] == 'temp1'
    assert namespace['hot']['properties'] == {'T0': 'Th'}
    assert json.loads((tmp_path/'old.json').read_text('utf-8')) == \
        namespace['d']
