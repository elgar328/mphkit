"""
Checks mk.compare: the plain functions on made-up descriptions without
COMSOL, then pairs of models built here that differ in one way each.
"""
import copy
import json

import pytest

import mphkit as mk
from mphkit import _compare, _describe
from mphkit._compare import Table, Tolerance, match_tables, same_value

def kinds(items):
    """The kinds of the items that are differences."""
    return [i['kind'] for i in items if i['kind'] != 'note']


###################
# Made-up models  #
###################

def box(x, y, z, size=None):
    found = {'x': list(x), 'y': list(y), 'z': list(z)}
    if size is not None:
        found['size'] = size
    return found


def cube_faces(low=0.0, high=1.0):
    """The six faces of a cube."""
    lo, hi = (low, low), (high, high)
    span = (low, high)
    area = (high - low)**2
    return [box(lo, span, span, area), box(hi, span, span, area),
            box(span, lo, span, area), box(span, hi, span, area),
            box(span, span, lo, area), box(span, span, hi, area)]


def geometry(domains, boundaries, tag='geom1', unit='m', scale=1.0,
             finalize='union', representation=None):
    lows = [min(d[a][0] for d in domains) for a in 'xyz']
    highs = [max(d[a][1] for d in domains) for a in 'xyz']
    return {'tag': tag, 'label': 'Geometry 1', 'dimension': 3,
            'axisymmetric': False, 'length_unit': unit,
            'length_scale': scale, 'finalize': finalize, 'voids': 0,
            'representation': representation,
            'bounding_box': {a: [lo, hi] for a, lo, hi in
                             zip('xyz', lows, highs)},
            'entities': {'domain': domains, 'boundary': boundaries,
                         'edge': [], 'point': []}}


def node(tag, kind, selection=None, properties=None, defaults=None,
         **extra):
    return {'tag': tag, 'path': f'comp1/ht/{tag}', 'type': kind,
            'label': kind, 'active': True,
            'properties': properties or {}, 'defaults': defaults or {},
            'selection': selection, 'selections': {}, 'features': [],
            **extra}


def described(features=(), geometries=None, parameters=None, **extra):
    """A description in the current format with one component and one
    interface."""
    if geometries is None:
        geometries = [geometry([box((0, 1), (0, 1), (0, 1), 1.0)],
                               cube_faces())]
    interface = {'tag': 'ht', 'path': 'comp1/ht', 'identifier': 'ht',
                 'type': 'HeatTransfer', 'label': 'Heat', 'active': True,
                 'settings': {}, 'defaults': {}, 'selection':
                 {'level': 'domain', 'geometry': 'geom1',
                  'entities': 'all'},
                 'features': list(features)}
    found = {'format': _describe.FORMAT, 'parameters': parameters or {},
             'functions': [],
             'variables': [], 'couplings': [], 'coordinate_systems': [],
             'materials': [], 'definitions': [], 'probes': [],
             'components': [{'tag': 'comp1', 'label': 'Component 1',
                             'geometries': geometries, 'pairs': [],
                             'physics': [interface], 'multiphysics': [],
                             'meshes': []}],
             'studies': [], 'solutions': [], 'notes': []}
    found.update(extra)
    return found


def faces(*numbers, geometries=None):
    """A boundary selection of cube faces (1 to 6)."""
    every = cube_faces()
    return {'level': 'boundary', 'geometry': 'geom1',
            'entities': [every[n - 1] for n in numbers]}


###################
# Plain functions #
###################

def test_spaced():
    assert _compare.spaced(' 1 +  2 ') == '1+2'
    assert _compare.spaced('0 10 20') == '0 10 20'
    assert _compare.spaced('0 10 20') != _compare.spaced('0 1020')
    assert _compare.spaced('range(0, 0.1, 1)') == 'range(0,0.1,1)'
    assert _compare.spaced('0 -1') == '0 -1'
    assert _compare.spaced('a - 1') == 'a-1'


@pytest.mark.parametrize('text, expected', [
    # as COMSOL lists them (measured: the output times of a solve)
    ('range(0,0.3,1)', [0, 0.3, 0.6, 0.9]),
    ('range(0,0.25,1)', [0, 0.25, 0.5, 0.75, 1]),
    ('range(0,0.1,0.35)', [0, 0.1, 0.2, 0.3]),
    ('range(1,-0.3,0)', [1, 0.7, 0.4, 0.1]),
    ('range(-0.3,0.1,0)', [-0.3, -0.2, -0.1, 0]),
    ('0 0.1 0.2', [0, 0.1, 0.2]),
    ('0,0.1', [0, 0.1]),
    ('0 range(1,1,3)', [0, 1, 2, 3]),
    ('range(0,dt,1)', None), ('1[s] 2[s]', None), ('x', None),
    # given up before making the values
    ('range(0,1,inf)', None), ('range(0,nan,1)', None),
    ('range(0,1e-7,1)', None),
    ('range(0,1,999999) range(0,1,999999)', None)])
def test_numbers(text, expected):
    found = _compare.numbers(text)
    if expected is None:
        assert found is None
    else:
        assert found == pytest.approx(expected, abs=1e-12)


def si(value, unit):
    return {'value': value, 'unit': unit}


def test_same_value():
    parameters = frozenset({'Th', 'h0'})
    assert same_value(' 1 + 2', '1+2') is None
    assert same_value('0 10 20', '0 1020') == 'property'
    # the same SI value and unit, no parameters: equal
    assert same_value('100[degC]', '373.15[K]', si(373.15, 'K'),
                      si(373.15, 'K'), parameters) is None
    # a parameter or a missing unit matters
    assert same_value('Th', '100[degC]', si(373.15, 'K'), si(373.15, 'K'),
                      parameters) == 'expression'
    assert same_value('373.15', '100[degC]', si(373.15, '1'),
                      si(373.15, 'K'), parameters) == 'expression'
    assert same_value('ht.Th', '100[degC]', si(373.15, 'K'),
                      si(373.15, 'K'), parameters) is None
    assert same_value('2*Th', '100[degC]', si(746.3, 'K'), si(373.15, 'K'),
                      parameters) == 'property'
    # only one side evaluates
    assert same_value('T', '1', None, si(1, '1')) == 'property'
    # lists of numbers, also as ranges
    assert same_value('range(0,0.1,0.3)', '0 0.1 0.2 0.3') is None
    assert same_value('range(-0.3,0.1,0)', '-0.3 -0.2 -0.1 0') is None
    assert same_value('range(0,0.1,1)', 'range(0,0.1,2)') == 'property'
    # isotropic tensors: one value or its diagonal
    assert same_value(['7'], ['7', '7', '7.0'], [si(7, '1')],
                      [si(7, '1')] * 3) is None
    assert same_value(['7'], ['7', '8', '7'], [si(7, '1')],
                      [si(7, '1'), si(8, '1'), si(7, '1')]) == 'property'
    assert same_value(['1', '1', '1'], '1') == 'property'
    assert same_value(0.15, 0.15 + 1e-12) is None
    assert same_value(True, 1) == 'property'
    # complex values
    assert same_value('1+2*i', '(1+2*i)', si([1, 2], '1'),
                      si([1, 2], '1')) is None


def test_places_text():
    assert _compare.place_text(box((100, 100), (0, 50), (0, 10), 500)) == \
        'x=100, y 0..50, z 0..10'
    assert _compare.places_text([box((0, 0), (0, 0), (0, 0))] * 5) == \
        'x=0, y=0, z=0; x=0, y=0, z=0; x=0, y=0, z=0; +2 more'


def table(places, scale=1.0, dim=2):
    return Table(places, scale, dim)


def test_tables():
    tol = Tolerance(1e-6, 2.0)
    one = table(cube_faces())
    # the same faces in another order
    cells, left_a, left_b = match_tables(one, table(cube_faces()[::-1]),
                                         tol)
    assert len(cells) == 6 and not left_a and not left_b
    # the face at z = 1 split into four pieces
    split = cube_faces()[:5] + [
        box(x, y, (1, 1), 0.25) for x in ((0, 0.5), (0.5, 1))
        for y in ((0, 0.5), (0.5, 1))]
    cells, left_a, left_b = match_tables(one, table(split), tol)
    assert sorted(len(b) for _, b in cells) == [1, 1, 1, 1, 1, 4]
    assert not left_a and not left_b
    # a face of a millimetre geometry against metres
    millimetres = table([box((0, 1000), (0, 1000), (0, 0), 1e6)], 1e-3)
    cells, _, _ = match_tables(table([box((0, 1), (0, 1), (0, 0), 1)]),
                               millimetres, tol)
    assert len(cells) == 1


def test_tables_tell_faces_apart():
    tol = Tolerance(1e-6, 2.0)
    # two x faces against two z faces of the same cube: same total area
    x_faces = table(cube_faces()[:2])
    z_faces = table(cube_faces()[4:])
    cells, left_a, left_b = match_tables(x_faces, z_faces, tol)
    assert not cells and len(left_a) == 2 and len(left_b) == 2
    assert not _compare.same_region(x_faces, left_a, z_faces, left_b, tol)


def test_same_place_is_one_row():
    # the two faces of a pair in an assembly
    face = box((1, 1), (0, 1), (0, 1), 1.0)
    rows = table([face, face, box((0, 0), (0, 1), (0, 1), 1.0)])
    assert len(rows.rows) == 2
    assert rows.row_of == {1: 0, 2: 0, 3: 1}
    assert rows.size([0]) == 1.0


def test_translator():
    translator = _compare.Translator({'ht', 'comp1', 'intop1'},
                                     {'ht', 'ht2', 'comp2', 'intop2', 'x'})
    translator.add('component', 'comp2', 'comp1')
    translator.add('identifier', 'ht2', 'ht')
    translator.add('physics', 'ht2', 'ht')
    translator.features['ht2'] = {'pc6': 'pc1'}
    translator.names[None] = {'intop2': 'intop1'}
    assert translator.value('ht2.q0+comp2.ht2.T') == 'ht.q0+comp1.ht.T'
    assert translator.value('ht2.pc6.T') == 'ht.pc1.T'
    assert translator.value('intop2(T)') == 'intop1(T)'
    assert translator.value('comp2.intop2(T)') == 'comp1.intop1(T)'
    assert translator.value('h0*comp2.intop2(1)/2') == 'h0*comp1.intop1(1)/2'
    # a parameter named like a tag stays: only prefixes and calls change
    assert translator.value('ht2*2') == 'ht2*2'
    # whole values, paths and step maps
    assert translator.value('ht2') == 'ht'
    assert translator.value('ht2/pc6') == 'ht/pc1'
    assert translator.value({'ht2': 'off', 'frame:spatial2': 'x'}) == \
        {'ht': 'off', 'frame:spatial1': 'x'}
    # a b tag without a partner that a has too is marked
    assert translator.value('ht.T') == '<b only:ht>.T'
    # materials and coordinate systems before a dot
    translator.b_words |= {'mat1', 'mat5', 'sys2'}
    translator.a_words |= {'mat1', 'sys1'}
    translator.add('material', 'mat1', 'mat1')
    translator.add('material', 'mat5', 'mat2')
    translator.add('coordinate', 'sys2', 'sys1')
    assert translator.value('mat1.def.rho*mat5.def.k') == \
        'mat1.def.rho*mat2.def.k'
    assert translator.value('sys2.T11') == 'sys1.T11'


#####################
# Made-up compares  #
#####################

def test_same():
    model = described([node('temp1', 'TemperatureBoundary', faces(1),
                            {'T0': '300[K]'}, {'T0': '293.15[K]'})])
    assert kinds(mk.compare(model, model)) == []
    assert kinds(mk.compare(model, json.loads(json.dumps(model)))) == []


def test_tags_and_order_do_not_matter():
    a = described([node('temp1', 'TemperatureBoundary', faces(1)),
                   node('temp2', 'TemperatureBoundary', faces(2),
                        {'T0': '400[K]'}, {'T0': '293.15[K]'})])
    b = described([node('temp7', 'TemperatureBoundary', faces(2),
                        {'T0': '400[K]'}, {'T0': '293.15[K]'}),
                   node('temp3', 'TemperatureBoundary', faces(1))])
    assert kinds(mk.compare(a, b)) == []
    labels = mk.compare(a, b, show={'label'})
    assert kinds(labels) == []


def test_property_and_default():
    a = described([node('temp1', 'TemperatureBoundary', faces(1),
                        {'T0': '400[K]'}, {'T0': '293.15[K]'})])
    b = described([node('temp1', 'TemperatureBoundary', faces(1))])
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert item['from_default'] is True
    assert item['a'] == '400[K]' and item['b'] == '293.15[K]'
    assert item['path'] == {'a': 'comp1/ht/temp1', 'b': 'comp1/ht/temp1'}
    assert 'T0 is 400[K] in a, 293.15[K] in b' in item['message']


def test_selection_and_only():
    a = described([node('temp1', 'TemperatureBoundary', faces(1, 2))])
    b = described([node('temp1', 'TemperatureBoundary', faces(1)),
                   node('hf1', 'HeatFluxBoundary',
                        {'level': 'boundary', 'geometry': 'geom1',
                         'entities': []})])
    found = mk.compare(a, b)
    assert kinds(found) == ['selection', 'only_in_b']
    selection = found[0]
    assert selection['numbers'] == {'a': [2], 'b': []}
    assert 'x=1' in selection['message']
    assert found[1]['empty'] is True
    assert kinds(mk.compare(a, b, ignore={'empty'})) == ['selection']


def test_unused_and_unknown():
    a = described([node('sz', 'Size', None, {'hmax': 1.0}, {'hmax': 2.0},
                        unused=['hmax'])])
    b = described([node('sz', 'Size', None, {}, {})])
    assert kinds(mk.compare(a, b)) == []
    a = described([node('x1', 'Thing', None, {'p': '1'}, {},
                        unknown_defaults=['p'])])
    b = described([node('x1', 'Thing', None, {}, {})])
    assert kinds(mk.compare(a, b)) == ['unchecked']


def test_geometry_differs_folds():
    thick = geometry([box((0, 1), (0, 1), (0, 2), 2.0)],
                     cube_faces()[:4] + [box((0, 1), (0, 1), (0, 0), 1),
                                         box((0, 1), (0, 1), (2, 2), 1)])
    a = described([node('temp1', 'TemperatureBoundary', faces(6),
                        {'T0': '400[K]'}, {'T0': '293.15[K]'})])
    top = box((0, 1), (0, 1), (2, 2), 1)
    b = described([node('temp1', 'TemperatureBoundary',
                        {'level': 'boundary', 'geometry': 'geom1',
                         'entities': [top]},
                        {'T0': '500[K]'}, {'T0': '293.15[K]'})],
                  geometries=[thick])
    found = mk.compare(a, b)
    # the one temperature of each is paired as a guess: the face it
    # selects moved with the geometry, its value stays a difference
    assert kinds(found) == ['geometry', 'property']
    geometry_item, value = found[:2]
    assert [c['kind'] for c in geometry_item['consequences']] == \
        ['selection']
    assert 'fix the geometry first' in geometry_item['message']
    assert value['matched_by_order'] is True
    assert value['message'].endswith('(paired by order)')


def test_parameters():
    a = described(parameters={
        'Th': {'expression': '100[degC]', 'value': 373.15, 'unit': 'K'},
        'L': {'expression': '1', 'value': 1, 'unit': '1'}})
    b = described(parameters={
        'Th': {'expression': '373.15[K]', 'value': 373.15, 'unit': 'K'},
        'w': {'expression': '2', 'value': 2, 'unit': '1'}})
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_a', 'only_in_b']
    assert found[0]['path'] == {'a': 'parameters/L', 'b': None}


def test_variables_by_name():
    def variables(*nodes):
        return [{'tag': tag, 'path': f'comp1/{tag}', 'label': tag,
                 'active': True, 'component': 'comp1', 'variables': names,
                 'selection': {'level': 'global'}}
                for tag, names in nodes]

    a = described(variables=variables(
        ('var1', {'p': '1', 'q': '2', 'r': '3', 's': '4'})))
    b = described(variables=variables(('var2', {'r': '3', 's': '4'}),
                                      ('var1', {'p': '1', 'q': '2'})))
    assert kinds(mk.compare(a, b)) == []
    b['variables'][0]['variables']['r'] = '5'
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'variable']
    assert item['name'] == 'r'
    b['variables'][0]['active'] = False
    assert kinds(mk.compare(a, b)).count('variable') == 3


def test_named_operators():
    def coupling(tag, names):
        return {'tag': tag, 'path': f'comp1/{tag}', 'type': 'Integration',
                'label': tag, 'active': True, 'names': names,
                'properties': {}, 'defaults': {}, 'selection': None,
                'selections': {}, 'features': [], 'component': 'comp1'}

    # paired by name, the tag and default name do not matter
    a = described([node('hf1', 'HeatFluxBoundary', faces(1),
                        {'q0': 'total(T)'}, {'q0': '0'})],
                  couplings=[coupling('intop1', ['total'])])
    b = described([node('hf1', 'HeatFluxBoundary', faces(1),
                        {'q0': 'total(T)'}, {'q0': '0'})],
                  couplings=[coupling('intop2', ['total'])])
    assert kinds(mk.compare(a, b)) == []
    # names that are the tags pair as the one left of a type
    a['couplings'] = [coupling('intop1', ['intop1'])]
    a['components'][0]['physics'][0]['features'][0]['properties'] = \
        {'q0': 'intop1(T)'}
    b['couplings'] = [coupling('intop2', ['intop2'])]
    b['components'][0]['physics'][0]['features'][0]['properties'] = \
        {'q0': 'intop2(T)'}
    assert kinds(mk.compare(a, b)) == []


def test_global_equations_by_row():
    def equations(tag, names, eqs):
        return node(tag, 'GlobalEquations', {'level': 'global'},
                    rows={'name': names, 'equation': eqs},
                    row_defaults={'name': '', 'equation': '',
                                  'initialValueU': '0'})

    a = described([equations('ge1', ['u1', 'u2'], ['u1t+u1', 'u2-1'])])
    b = described([equations('ge2', ['u2'], ['u2-1']),
                   equations('ge1', ['u1'], ['u1t+u1'])])
    assert kinds(mk.compare(a, b)) == []
    b['components'][0]['physics'][0]['features'][0]['rows']['equation'] = \
        ['u2-2']
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert item['name'] == 'u2'


def equations(tag, names, **extra):
    return node(tag, 'GlobalEquations', {'level': 'global'},
                rows={'name': names, 'equation': [f'{n}t' for n in names]},
                row_defaults={'name': '', 'equation': '',
                              'initialValueU': '0'}, **extra)


def mirrored(a, b):
    """Tells whether compare(b, a) is compare(a, b) with a and b
    swapped, by kind, paths and count."""
    swap = {'only_in_a': 'only_in_b', 'only_in_b': 'only_in_a'}

    def key(items, flip):
        return sorted((swap.get(i['kind'], i['kind']) if flip else i['kind'],
                       i['path']['b' if flip else 'a'],
                       i['path']['a' if flip else 'b'])
                      for i in items if i['kind'] != 'unchecked')
    return key(mk.compare(a, b), False) == key(mk.compare(b, a), True)


def test_global_equation_features():
    # a disabled feature is found where the equations are split otherwise
    a = described([equations('ge1', ['u']),
                   equations('ge2', ['v'], active=False)])
    b = described([equations('ge1', ['u', 'v'])])
    [item] = mk.compare(a, b)
    assert item['kind'] == 'active'
    assert item['path'] == {'a': 'comp1/ht/ge2', 'b': 'comp1/ht/ge1'}
    assert mirrored(a, b)
    # one disabled feature linked to two: one item, with its own path
    a = described([equations('ge1', ['u', 'v'], active=False)])
    b = described([equations('ge1', ['u']), equations('ge2', ['v'])])
    found = [i for i in mk.compare(a, b) if i['kind'] == 'active']
    assert [i['path']['a'] for i in found] == ['comp1/ht/ge1']
    assert mirrored(a, b)
    # labels only where one is paired with one
    b['components'][0]['physics'][0]['features'][1]['label'] = 'Other'
    a['components'][0]['physics'][0]['features'][0]['active'] = True
    assert mk.compare(a, b, show='label') == []
    # a subfeature only one side has, once
    weak = node('wk1', 'WeakContribution', {'level': 'global'})
    a['components'][0]['physics'][0]['features'][0]['features'] = [weak]
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_a']
    assert found[0]['path']['a'] == 'comp1/ht/wk1'
    # an equation only one side has, once
    b = described([equations('ge1', ['u'])])
    a = described([equations('ge1', ['u', 'v'])])
    assert [i['name'] for i in mk.compare(a, b)] == ['v']


def test_global_equation_groups():
    # features linked through shared equations form one group: two in a
    # with two in b, so no label is compared and steps that name them
    # are unchecked
    def step(disabled):
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'solver': {'status': 'automatic',
                                            'sequence': None},
                 'steps': [{'tag': 'stat', 'path': 'std1/stat',
                            'type': 'Stationary', 'label': 'Stationary',
                            'active': True,
                            'properties': {'disabledphysics': disabled},
                            'defaults': {'disabledphysics': []},
                            'selection': None, 'selections': {},
                            'features': []}]}]

    a = described([equations('ge1', ['u'], label='X'),
                   equations('ge2', ['v', 'w'])],
                  studies=step(['ht/ge1']))
    b = described([equations('ge1', ['w'], label='Y'),
                   equations('ge2', ['u', 'v'])],
                  studies=step(['ht/ge2']))
    found = mk.compare(a, b, show='label')
    assert kinds(found) == ['unchecked']
    assert 'put into features otherwise' in found[0]['message']
    assert mirrored(a, b)


def test_global_equation_tags():
    # the tags of global equations translate in study steps
    def model(features, disabled):
        step = {'tag': 'stat', 'path': 'std1/stat', 'type': 'Stationary',
                'label': 'Stationary', 'active': True,
                'properties': {'disabledphysics': disabled},
                'defaults': {'disabledphysics': []}, 'selection': None,
                'selections': {}, 'features': []}
        return described(features, studies=[{
            'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
            'active': True, 'steps': [step],
            'solver': {'status': 'automatic', 'sequence': None}}])

    a = model([equations('ge1', ['u'])], ['ht/ge1'])
    b = model([equations('ge2', ['u'])], ['ht/ge2'])
    assert mk.compare(a, b) == []


def test_global_equation_settings():
    # a setting one side leaves at its default
    a = described([equations('ge1', ['u'],
                             properties={'quantity': 'length'},
                             defaults={'quantity': 'none'})])
    b = described([equations('ge1', ['u'])])
    [item] = mk.compare(a, b)
    assert (item['a'], item['b']) == ('length', 'none')
    # unknown defaults are unchecked
    del a['components'][0]['physics'][0]['features'][0]['defaults'][
        'quantity']
    assert kinds(mk.compare(a, b)) == ['unchecked']
    # b's default in a's tags
    a = described([equations('ge1', ['u'], defaults={'src': 'ht.T'})])
    b = described()
    b['components'][0]['physics'] = [interface('ht2', [equations(
        'ge1', ['u'], properties={'src': 'ht2.x'},
        defaults={'src': 'ht2.T'})])]
    [item] = mk.compare(a, b)
    assert (item['a'], item['b']) == ('ht.T', 'ht2.x')
    assert item['from_default'] is True
    a = described([equations('ge1', ['u'],
                             properties={'quantity': 'length'},
                             defaults={'quantity': 'none'})])
    b = described([equations('ge1', ['u'])])
    del a['components'][0]['physics'][0]['features'][0]['defaults'][
        'quantity']
    # features of another type keep their own values
    b['components'][0]['physics'][0]['features'][0]['type'] = 'Other'
    [item] = mk.compare(a, b)
    assert (item['a'], item['b']) == ('length', None)


def test_step_maps():
    def step(activate, defaults):
        return {'tag': 'stat', 'path': 'std1/stat', 'type': 'Stationary',
                'label': 'Stationary', 'active': True,
                'properties': {'activate': activate} if activate else {},
                'defaults': {'activate': defaults} if defaults else {},
                'selection': None, 'selections': {}, 'features': []}

    def study(*steps):
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'steps': list(steps),
                 'solver': {'status': 'automatic', 'sequence': None}}]

    a = described(studies=study(step({'ec': 'off'}, {'ec': 'on'})))
    b = described(studies=study(step({'ec': 'off', 'ht': 'off'},
                                     {'ec': 'on', 'ht': 'on'})))
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert item['name'] == 'activate[ht]'
    assert (item['a'], item['b']) == ('on', 'off')


def test_solver_not_asked():
    def study(status):
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'steps': [],
                 'solver': {'status': status, 'sequence': 'sol1'}}]

    a = described(studies=study('not_asked'))
    b = described(studies=study('compared'))
    found = mk.compare(a, b)
    assert kinds(found) == []
    assert any('solver=True' in i['message'] for i in found
               if i['kind'] == 'note')


def test_solution_tags():
    # a store-solution node names a new solution each time
    def study(solution, path='su1'):
        change = {'path': path, 'labels': path, 'type': 'StoreSolution',
                  'change': 'property',
                  'properties': {'sol': [[solution], ['sol2']]}}
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'steps': [],
                 'solver': {'status': 'compared', 'sequence': 'sol1',
                            'changes': [change]}}]

    a = described(studies=study('sol2'), solutions=['sol1', 'sol2'])
    b = described(studies=study('sol3'), solutions=['sol1', 'sol3'])
    assert kinds(mk.compare(a, b)) == []
    b = described(studies=study('sol3', 'su2'), solutions=['sol1', 'sol3'])
    found = mk.compare(a, b)
    assert kinds(found) == ['solver', 'solver']
    assert found[0]['message'].endswith(
        'solver su1 (StoreSolution) property sol ["sol2"] (COMSOL: '
        '["sol2"]) only in a')


def test_exterior():
    # domain 0 of boundary elements: a flag next to the real domains
    def domains(exterior, entities='all', applied=None):
        found = {'level': 'domain', 'geometry': 'geom1',
                 'entities': entities}
        if exterior:
            found['exterior'] = True
        if applied is not None:
            found['exterior_applied'] = applied
        return found

    a = described([node('bpam1', 'BoundaryElements', domains(True))])
    b = described([node('bpam1', 'BoundaryElements', domains(True))])
    assert kinds(mk.compare(a, b)) == []
    b = described([node('bpam1', 'BoundaryElements', domains(False))])
    [item] = [i for i in mk.compare(a, b) if i['kind'] != 'note']
    assert item['kind'] == 'selection'
    assert item['message'].endswith('the exterior (domain 0) only in a')
    # only the exterior: not an empty selection
    only = described([node('mat1', 'Common', domains(True, []))])
    found = mk.compare(only, described())
    assert kinds(found) == ['only_in_a']
    assert 'empty' not in found[0]
    # applied to the exterior only: it does not apply nowhere
    a = described([node('bpam1', 'BoundaryElements',
                        domains(True, 'all', applied=True))])
    a['components'][0]['physics'][0]['features'][0]['selection'][
        'applied'] = []
    b = copy.deepcopy(a)
    b['components'][0]['physics'][0]['features'][0]['selection'][
        'entities'] = []
    found = [i for i in mk.compare(a, b) if i['kind'] == 'selection']
    assert found and not any('nowhere' in i['message'] for i in found)


def test_materials():
    def material(tag, groups, component='comp1', kind='Common',
                 properties=None):
        return {'tag': tag, 'path': tag, 'type': kind, 'label': tag,
                'active': True, 'properties': properties or {},
                'defaults': {}, 'component': component,
                'selection': {'level': 'domain', 'geometry': 'geom1',
                              'entities': 'all'} if component else None,
                'selections': {}, 'features': [], 'groups': groups}

    library = {'def': {'properties': {
        'thermalconductivity': ['400[W/(m*K)]'], 'density': '8960[kg/m^3]',
        'sys': 'x', 'relpermittivity': ['1']},
        'si': {'thermalconductivity': [{'value': 400, 'unit': 'W/(m*K)'}],
               'density': {'value': 8960, 'unit': 'kg/m^3'}}}}
    own = {'def': {'properties': {
        'thermalconductivity': ['400[W/(m*K)]'],
        'density': '8960[kg/m^3]'},
        'si': {'thermalconductivity': [{'value': 400, 'unit': 'W/(m*K)'}],
               'density': {'value': 8960, 'unit': 'kg/m^3'}}}}
    a = described(materials=[material('mat1', library)])
    b = described(materials=[material('mat5', own)])
    # the library's own entries are hidden unless shown
    assert kinds(mk.compare(a, b)) == ['property']
    found = mk.compare(a, b, show={'material_info'})
    assert kinds(found) == ['property', 'property']
    info = [i for i in found if i.get('material_info')]
    assert info and info[0]['a'] == {'sys': 'x'}
    # an old ignore= of it is accepted and changes nothing
    assert kinds(mk.compare(a, b, ignore={'material_info'})) == ['property']
    # a link to a global material against a material of the component
    linked = described(materials=[
        material('lnk1', {}, kind='Link', properties={'link': 'gm1'}),
        material('gm1', own, component=None)])
    assert kinds(mk.compare(linked, b)) == []


def thick_plate():
    """The cube made 2 high: the bottom face is the same, the rest moved."""
    return geometry([box((0, 1), (0, 1), (0, 2), 2.0)],
                    [box((0, 0), (0, 1), (0, 2), 2.0),
                     box((1, 1), (0, 1), (0, 2), 2.0),
                     box((0, 1), (0, 0), (0, 2), 2.0),
                     box((0, 1), (1, 1), (0, 2), 2.0),
                     box((0, 1), (0, 1), (0, 0), 1.0),
                     box((0, 1), (0, 1), (2, 2), 1.0)])


def test_geometry_keeps_real_selection_differences():
    # a selects the bottom (the same in both) and a moved side face; b
    # only the moved side face: the bottom is a real difference
    a = described([node('hf1', 'HeatFluxBoundary', faces(2, 5))])
    side = thick_plate()['entities']['boundary'][1]
    b = described([node('hf1', 'HeatFluxBoundary',
                        {'level': 'boundary', 'geometry': 'geom1',
                         'entities': [side]})], geometries=[thick_plate()])
    found = mk.compare(a, b)
    assert kinds(found) == ['geometry', 'selection']
    selection = found[1]
    assert selection['numbers'] == {'a': [5], 'b': []}
    assert selection['elsewhere'] == 2
    assert '+2 entities without a counterpart' in selection['message']
    # the geometry item has the geometry's path
    assert found[0]['path'] == {'a': 'comp1/geom1', 'b': 'comp1/geom1'}


def test_used_only():
    # custom on in a (its sizes count), off in b (its hauto counts)
    a = described([node('sz', 'Size', None,
                        {'custom': 'on', 'hmax': 15.0, 'hmin': 4.86},
                        {'custom': 'off', 'hmax': 6.0, 'hmin': 1.0},
                        unused=['hauto'])])
    b = described([node('sz', 'Size', None, {'hauto': 3.0},
                        {'hauto': 5.0},
                        unused=['hmax', 'hmaxactive', 'hmin'])])
    found = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert [i.get('name') for i in found] == ['custom', None]
    used = found[1]
    assert used['used_only'] is True
    assert used['a'] == {'hmax': 15.0, 'hmin': 4.86}
    assert used['b'] == {'hauto': 3.0}
    assert 'used only in a: hmax 15.0, hmin 4.86; used only in b: ' \
        'hauto 3.0' in used['message']
    # not where defaults are unknown
    b['components'][0]['physics'][0]['features'][0]['all_properties'] = True
    assert not [i for i in mk.compare(a, b) if i.get('used_only')]
    # b takes the value from another node: its own one is unused
    a = described([node('pc1', 'PeriodicHeat', None,
                        {'k': ['kx', '0', '0']}, {'k': ['0', '0', '0']})])
    b = described([node('pc1', 'PeriodicHeat', None,
                        {'k_src': 'root.comp1.ht.pp1.k'},
                        {'k_src': 'userdef'}, unused=['k'])])
    found = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert [i.get('name') for i in found] == ['k_src', None]
    assert found[0]['from_default'] is True
    assert found[1]['used_only'] is True
    assert found[1]['a'] == {'k': ['kx', '0', '0']}


def test_empty_selection_pairs_by_tag():
    empty = {'level': 'boundary', 'geometry': 'geom1', 'entities': []}
    a = described([node('pc1', 'PeriodicHeat', faces(1, 2)),
                   node('pc2', 'PeriodicHeat', empty)])
    b = described([node('pc1', 'PeriodicHeat', empty),
                   node('pc2', 'PeriodicHeat', empty)])
    found = mk.compare(a, b)
    assert kinds(found) == ['selection']
    assert found[0]['path'] == {'a': 'comp1/ht/pc1', 'b': 'comp1/ht/pc1'}


def test_mass_properties_by_name():
    def mass(tag, name):
        return {'tag': tag, 'path': f'comp1/{tag}', 'type': 'MassProperties',
                'label': 'Mass Properties 1', 'active': True, 'names': [name],
                'properties': {'name': name} if name != tag else {},
                'defaults': {'name': tag} if name != tag else {},
                'selection': {'level': 'domain', 'geometry': 'geom1',
                              'entities': 'all'},
                'selections': {}, 'features': []}

    def variables(expression):
        return [{'tag': 'var1', 'path': 'comp1/var1', 'label': 'Variables 1',
                 'active': True, 'component': 'comp1',
                 'variables': {'m': expression},
                 'selection': {'level': 'global'}}]

    a = described(variables=variables('mp.mass*2+comp1.mp.I11'))
    a['components'][0]['mass_properties'] = [mass('mass1', 'mp')]
    b = described(variables=variables('mass1.mass*2+comp1.mass1.I11'))
    b['components'][0]['mass_properties'] = [mass('mass1', 'mass1')]
    assert kinds(mk.compare(a, b)) == []
    b['components'][0]['mass_properties'] = []
    assert kinds(mk.compare(a, b)) == ['only_in_a', 'variable']


def test_messages():
    a = described([node('temp1', 'TemperatureBoundary', faces(1),
                        {'T0': '400[K]'}, {'T0': '293.15[K]'})])
    b = described([node('temp3', 'TemperatureBoundary', faces(1)),
                   node('hf1', 'HeatFluxBoundary', faces(3))])
    found = mk.compare(a, b)
    # paths without the component, each side named by its own letter
    assert found[0]['message'].startswith(
        'TemperatureBoundary (a ht/temp1, b ht/temp3): T0')
    assert found[1]['message'].startswith('HeatFluxBoundary (b ht/hf1)')


def test_material_coordinate_system():
    def material(sys):
        return {'tag': 'mat1', 'path': 'comp1/mat1', 'type': 'Common',
                'label': 'Air', 'active': True,
                'properties': {'sys': sys} if sys != 'sys1' else {},
                'defaults': {'sys': 'sys1'} if sys != 'sys1' else {},
                'component': 'comp1',
                'selection': {'level': 'domain', 'geometry': 'geom1',
                              'entities': 'all'},
                'selections': {}, 'features': [], 'groups': {}}

    # the library's 'none': hidden unless shown
    a, b = described(materials=[material('sys1')]), \
        described(materials=[material('none')])
    assert kinds(mk.compare(a, b)) == []
    assert kinds(mk.compare(a, b, show={'material_info'})) == ['property']
    # two real coordinate systems: a difference
    b = described(materials=[material('sys2')])
    assert kinds(mk.compare(a, b)) == ['property']


def test_material_function_keys():
    def material(function):
        return {'tag': 'mat1', 'path': 'comp1/mat1', 'type': 'Common',
                'label': 'Air', 'active': True, 'properties': {},
                'defaults': {}, 'component': 'comp1',
                'selection': {'level': 'domain', 'geometry': 'geom1',
                              'entities': 'all'},
                'selections': {}, 'features': [],
                'groups': {'def': {'properties': {},
                                   'functions': {'rho': function}}}}

    plain = {'type': 'Analytic', 'expr': 'pA*0.02897/R_const/T'}
    a = described(materials=[material({**plain, 'argders': [['pA', 'd']]})])
    b = described(materials=[material(plain)])
    # only the derivatives differ: a library entry
    assert kinds(mk.compare(a, b)) == []
    [item] = [i for i in mk.compare(a, b, show={'material_info'})
              if i['kind'] == 'property']
    assert 'functions/rho: argders is' in item['message']
    b = described(materials=[material({**plain, 'expr': 'pA/R_const/T'})])
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert 'expr is' in item['message'] and 'argders' in item['message']


#####################################
# Where nodes apply, and the order  #
#####################################

def applying(numbers, applied=None):
    """Cube faces (1 to 6) that a node selects and where it applies."""
    found = faces(*numbers)
    if applied is not None:
        found['applied'] = faces(*applied)['entities']
    return found


def test_applied_linked_to_its_cause():
    # b adds temp2 on face 2, which takes it from temp1
    a = described([node('temp1', 'TemperatureBoundary', applying((1, 2)))])
    b = described([node('temp1', 'TemperatureBoundary',
                        applying((1, 2), (1,))),
                   node('temp2', 'TemperatureBoundary', faces(2))])
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_b']
    [linked] = found[0]['consequences']
    assert linked['kind'] == 'applied'
    assert linked['causes'] == [found[0]['path']]
    assert linked['numbers'] == {'a': [2], 'b': []}
    assert found[0]['message'].endswith('(+1 consequence)')
    # the other way round
    [back] = mk.compare(b, a)[0]['consequences']
    assert back['causes'] == [{'a': 'comp1/ht/temp2', 'b': None}]
    # a hidden cause leaves what it explains
    assert kinds(mk.compare(a, b, ignore={'only_in_b'})) == ['applied']
    shown = mk.compare(a, b, ignore={'applied'})
    assert 'consequences' not in shown[0]
    assert shown[0]['message'].endswith('only in b')


def test_applied_not_for_own_selection():
    # temp1 also selects face 3 in b: a selection difference only
    a = described([node('temp1', 'TemperatureBoundary',
                        applying((1, 2), (1,))),
                   node('temp2', 'TemperatureBoundary', faces(2))])
    b = described([node('temp1', 'TemperatureBoundary',
                        applying((1, 2, 3), (1, 3))),
                   node('temp2', 'TemperatureBoundary', faces(2))])
    assert kinds(mk.compare(a, b)) == ['selection']


def test_applied_without_cause():
    # a disabled node overrides nothing: no cause, at the top
    a = described([node('temp1', 'TemperatureBoundary', applying((1, 2)))])
    b = described([node('temp1', 'TemperatureBoundary',
                        applying((1, 2), (1,))),
                   node('temp2', 'TemperatureBoundary', faces(2),
                        active=False)])
    found = mk.compare(a, b)
    assert kinds(found) == ['applied', 'only_in_b']
    assert 'causes' not in found[0]


def test_applied_under_interface():
    # an interface only in b explains a difference only with couplings
    def interface_b(model, couplings):
        component = model['components'][0]
        component['physics'].append({
            'tag': 'ht2', 'path': 'comp1/ht2', 'identifier': 'ht2',
            'type': 'HeatTransfer', 'label': 'Heat 2', 'active': True,
            'settings': {}, 'defaults': {}, 'selection': {
                'level': 'domain', 'geometry': 'geom1', 'entities': 'all'},
            'features': [{**node('temp1', 'TemperatureBoundary', faces(2)),
                          'path': 'comp1/ht2/temp1'}]})
        component['multiphysics'] = couplings
        return model

    def model_b(couplings):
        return interface_b(described([node(
            'temp1', 'TemperatureBoundary', applying((1, 2), (1,)))]),
            couplings)

    a = described([node('temp1', 'TemperatureBoundary', applying((1, 2)))])
    assert kinds(mk.compare(a, model_b([]))) == ['applied', 'only_in_b']
    coupling = {**node('te1', 'Coupling', {
        'level': 'domain', 'geometry': 'geom1', 'entities': 'all'}),
        'path': 'comp1/te1'}
    b = model_b([coupling])
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_b', 'only_in_b']
    assert found[0]['path']['b'] == 'comp1/ht2'
    assert [c['kind'] for c in found[0]['consequences']] == ['applied']


def test_disabled_interface():
    # a disabled interface: its features read as disabled, and no longer
    # override each other
    def model(active):
        made = described([
            node('temp1', 'TemperatureBoundary',
                 applying((1, 2), (1, 2) if not active else (1,)),
                 active=active),
            node('temp2', 'TemperatureBoundary', faces(2), active=active)])
        made['components'][0]['physics'][0]['active'] = active
        return made

    found = mk.compare(model(True), model(False))
    assert kinds(found) == ['active']
    # a disabled node's own entities read as if enabled: no 'applied'
    assert sorted(c['kind'] for c in found[0]['consequences']) == \
        ['active', 'active']
    assert kinds(mk.compare(model(False), model(False))) == []


def test_material_overridden():
    def material(tag, entities, applied=None):
        selection = {'level': 'domain', 'geometry': 'geom1',
                     'entities': entities}
        if applied is not None:
            selection['applied'] = applied
        return {'tag': tag, 'path': f'comp1/{tag}', 'type': 'Common',
                'label': tag, 'active': True, 'properties': {},
                'defaults': {}, 'component': 'comp1',
                'selection': selection, 'selections': {}, 'features': [],
                'groups': {}}

    first = box((0, 1), (0, 1), (0, 1), 1.0)
    second = box((1, 2), (0, 1), (0, 1), 1.0)
    shapes = [geometry([first, second], cube_faces())]
    a = described(geometries=shapes, materials=[material('mat1', 'all')])
    b = described(geometries=copy.deepcopy(shapes), materials=[
        material('mat1', 'all', [first]), material('mat2', [second])])
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_b']
    assert found[0]['consequences'][0]['path']['a'] == 'comp1/mat1'


def test_order():
    def mesh(*tags):
        made = {'tag': 'mesh1', 'path': 'comp1/mesh1', 'label': 'Mesh 1',
                'geometry': 'geom1', 'automatic': False, 'features': []}
        for tag in tags:
            made['features'].append({
                'tag': tag, 'path': f'comp1/mesh1/{tag}', 'type': 'FreeTet',
                'label': tag, 'active': True, 'properties': {},
                'defaults': {}, 'selection': faces(int(tag[-1])),
                'selections': {}, 'features': []})
        return [made]

    a = described()
    a['components'][0]['meshes'] = mesh('ftet1', 'ftet2', 'ftet3')
    b = described()
    b['components'][0]['meshes'] = mesh('ftet2', 'ftet1', 'ftet3')
    [item] = mk.compare(a, b)[:1]
    assert item['kind'] == 'order'
    assert item['a'] == ['ftet1', 'ftet2', 'ftet3']
    assert item['b'] == ['ftet2', 'ftet1', 'ftet3']
    assert 'ftet2 (a mesh1/ftet2, b mesh1/ftet2) moved' in item['message']
    assert item['message'].endswith('; b: ftet2, ftet1, ftet3')
    assert kinds(mk.compare(a, b, ignore={'mesh'})) == []
    assert kinds(mk.compare(b, a)) == ['order']


def test_order_of_steps():
    def study(*kinds):
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'solver': {'status': 'automatic',
                                            'sequence': None},
                 'steps': [{'tag': kind.lower(), 'path': f'std1/{kind}',
                            'type': kind, 'label': kind, 'active': True,
                            'properties': {}, 'defaults': {},
                            'selection': None, 'selections': {},
                            'features': []} for kind in kinds]}]

    a = described(studies=study('Stationary', 'Frequency'))
    b = described(studies=study('Frequency', 'Stationary'))
    assert kinds(mk.compare(a, b)) == ['order']
    assert kinds(mk.compare(a, a)) == []


def test_file_names_and_words():
    # tags of steps and meshes are not tags before a dot; file names keep
    # theirs
    step = {'tag': 'param', 'path': 'std1/param', 'type': 'Parametric',
            'label': 'Parametric Sweep', 'active': True,
            'properties': {'filename': 'C:\\Users\\x\\param.mph'},
            'defaults': {'filename': ''}, 'selection': None,
            'selections': {}, 'features': []}
    study = [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
              'active': True, 'steps': [step],
              'solver': {'status': 'automatic', 'sequence': None}}]
    model = described(studies=study)
    assert 'param' not in _compare.words(model)
    assert mk.compare(model, copy.deepcopy(model)) == []
    translator = _compare.Translator({'comp2'}, {'comp1'})
    translator.add('component', 'comp1', 'comp2')
    assert translator.value('1/comp1.k') == '1/comp2.k'
    assert translator.value('q/comp1.A') == 'q/comp2.A'
    assert translator.value('/home/u/comp1.mph') == '/home/u/comp1.mph'
    assert translator.value('C:\\data\\comp1.mph') == \
        'C:\\data\\comp1.mph'


def test_component_only_in_b():
    # b's comp1 has no partner; its comp2 is a's comp1: a variable of b's
    # comp1 is not paired with one of a's comp1
    def variables(component, value):
        return {'tag': 'var1', 'path': f'{component}/var1', 'label': 'var1',
                'active': True, 'component': component,
                'variables': {'T0': value}, 'selection': {'level': 'global'}}

    cube = geometry([box((0, 1), (0, 1), (0, 1), 1.0)], cube_faces())
    flat = {**geometry([box((0, 1), (0, 1), (0, 0), 1.0)], []),
            'dimension': 2, 'bounding_box': {'x': [0, 1], 'y': [0, 1]}}
    a = described(variables=[variables('comp1', '300[K]')])
    b = described(variables=[variables('comp2', '300[K]'),
                             variables('comp1', '999[K]')])
    component = b['components'][0]
    b['components'] = [
        {**component, 'tag': 'comp1', 'geometries': [flat], 'physics': []},
        {**component, 'tag': 'comp2', 'geometries': [cube]}]
    found = mk.compare(a, b)
    assert 'variable' not in kinds(found)
    # its variable comes with the component
    [component_only] = [i for i in found
                        if 'component only in b' in i['message']]
    assert component_only['path'] == {'a': None, 'b': 'comp1'}
    assert component_only['message'].endswith('(+1 consequence)')
    [only] = component_only['consequences']
    assert (only['name'], only['path']['b']) == ('T0', 'comp1/var1')
    back = mk.compare(b, a)
    assert sorted(kinds(back)) == sorted(
        k.replace('only_in_b', 'only_in_a') for k in kinds(found))
    [component_only] = [i for i in back
                        if 'component only in a' in i['message']]
    assert [i['name'] for i in component_only['consequences']] == ['T0']
    assert 'comp1/var1' not in str(mk.compare(a, b, ignore='only_in_b'))


def test_component_only_with_its_nodes():
    # the operators and variables of a component only b has come with it
    a, b = described(), described()
    second = copy.deepcopy(b['components'][0])
    second['tag'] = 'comp2'
    second['geometries'] = [geometry([box((5, 6), (0, 1), (0, 1), 1.0)],
                                     cube_faces(5, 6), tag='geom2')]
    second['physics'][0].update(tag='ht2', identifier='ht2',
                                path='comp2/ht2')
    b['components'].append(second)
    b['couplings'] = [{
        'tag': 'intop1', 'path': 'comp2/intop1', 'type': 'Integration',
        'label': 'I', 'active': True, 'names': ['intop1'], 'properties': {},
        'defaults': {}, 'selection': None, 'selections': {}, 'features': [],
        'component': 'comp2'}]
    b['variables'] = [{
        'tag': 'var1', 'path': 'comp2/var1', 'label': 'v', 'active': True,
        'component': 'comp2', 'variables': {'p': '1', 'q': '2'},
        'selection': {'level': 'global'}}]
    [item] = mk.compare(a, b)
    assert item['path']['b'] == 'comp2'
    assert [i['path']['b'] for i in item['consequences']] == \
        ['comp2/intop1', 'comp2/var1', 'comp2/var1']
    assert item['message'].endswith('(+3 consequences)')
    assert mirrored(a, b)


@pytest.mark.parametrize('one, other, expected', [
    ('not_asked', 'not_asked', []), ('not_asked', 'automatic', []),
    ('not_asked', 'compared', ['note']),
    ('not_asked', 'not_compared', ['note', 'unchecked']),
    ('compared', 'automatic', []),
    ('not_compared', 'compared', ['unchecked']),
    ('not_compared', 'not_compared', ['unchecked', 'unchecked'])])
def test_solver_statuses(one, other, expected):
    def study(status):
        return [{'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
                 'active': True, 'steps': [],
                 'solver': {'status': status, 'sequence': 'sol1',
                            'reason': 'mesh not built', 'changes': []}}]

    found = mk.compare(described(studies=study(one)),
                       described(studies=study(other)))
    assert sorted(i['kind'] for i in found) == sorted(expected)


def test_arguments_and_order():
    a = described(parameters={'L': {'expression': '1', 'value': 1,
                                    'unit': '1'}})
    b = described([node('hf1', 'HeatFluxBoundary', faces(1))])
    assert mk.compare(a, b, ignore='only_in_b') == \
        mk.compare(a, b, ignore={'only_in_b'})
    with pytest.raises(ValueError, match='does not know'):
        mk.compare(a, b, show={'parameter'})
    with pytest.raises(ValueError, match='tolerance'):
        mk.compare(a, b, tolerance=-1)
    # a parameter only in a comes before the nodes
    assert kinds(mk.compare(b, a)) == ['only_in_b', 'only_in_a']
    assert kinds(mk.compare(a, b)) == ['only_in_a', 'only_in_b']
    assert mk.compare(b, a)[0]['path']['b'] == 'parameters/L'
    assert mk.compare(a, b, ignore=None) == mk.compare(a, b)
    # names given once, as an iterator
    assert mk.compare(a, b, ignore=iter(['only_in_b'])) == \
        mk.compare(a, b, ignore='only_in_b')
    for wrong in ({'only_in_b': 1}, ['only_in_b', 1], 3):
        with pytest.raises(ValueError, match='a name or a list'):
            mk.compare(a, b, ignore=wrong)
    with pytest.raises(ValueError, match="material_info"):
        mk.compare(a, b, ignore='nothing')


def test_incomplete_results():
    good = described([node('hf1', 'HeatFluxBoundary', faces(1))])
    for broken, problem in (
            ({**good, 'components': None}, "'components' is no list"),
            ({**good, 'parameters': []}, "'parameters' is no dict"),
            ({**good, 'components': [{'geometries': []}]}, 'has no tag'),
            ({**good, 'components': [{'tag': 'comp1'}]}, "'geometries'"),
            ({**good, 'probes': None}, "'probes' is no list")):
        with pytest.raises(ValueError, match='a is not a complete') as found:
            mk.compare(broken, good)
        assert problem in str(found.value)
    # deeper down: the reading error, without blaming the data alone
    deep = copy.deepcopy(good)
    del deep['components'][0]['physics'][0]['features'][0]['tag']
    with pytest.raises(ValueError, match='could not read a or b') as found:
        mk.compare(deep, good)
    assert isinstance(found.value.__cause__, KeyError)
    assert 'bug in mphkit' in str(found.value)
    # an empty model (no components, a 0D geometry list) is complete
    empty = {**good, 'components': []}
    assert mk.compare(empty, empty) == []


def pair_node(tag, source, destination):
    return {'tag': tag, 'path': f'comp1/{tag}', 'type': 'IdentityBoundaryPair',
            'label': 'Pair', 'active': True, 'source': source,
            'destination': destination, 'properties': {}, 'defaults': {},
            'features': []}


def test_pair_unmeasured():
    unknown = {'level': 'boundary', 'geometry': 'geom1',
               'entities': 'unknown'}
    a, b = described(), described()
    a['components'][0]['pairs'] = [pair_node('p1', unknown, faces(2))]
    b['components'][0]['pairs'] = [pair_node('p1', faces(3), faces(2))]
    found = mk.compare(a, b)
    assert kinds(found) == ['unchecked']
    assert 'source not compared' in found[0]['message']
    # what cannot be compared is not the same: no swapped source and
    # destination, the destination differs
    b['components'][0]['pairs'] = [pair_node('p1', faces(2), faces(3))]
    found = mk.compare(a, b)
    assert sorted(kinds(found)) == ['selection', 'unchecked']
    [item] = [i for i in found if i['kind'] == 'selection']
    assert 'destination differs' in item['message']
    a['components'][0]['pairs'] = [pair_node('p1', faces(1), unknown)]
    b['components'][0]['pairs'] = [pair_node('p1', faces(1), faces(2))]
    found = mk.compare(a, b)
    assert kinds(found) == ['unchecked']
    assert 'destination not compared' in found[0]['message']


def test_mesh_units():
    # sizes in the length unit scale; levels and counts do not
    def model(unit, scale, hmax, hauto, layers):
        made = described(geometries=[geometry(
            [box((0, 1 / scale), (0, 1 / scale), (0, 1 / scale),
                 1 / scale**3)], cube_faces(0, 1 / scale), unit=unit,
            scale=scale)])
        made['components'][0]['meshes'] = [{
            'tag': 'mesh1', 'path': 'comp1/mesh1', 'label': 'Mesh 1',
            'geometry': 'geom1', 'automatic': False, 'features': [
                {'tag': 'size', 'path': 'comp1/mesh1/size', 'type': 'Size',
                 'label': 'Size', 'active': True,
                 'properties': {'hmax': hmax, 'hauto': hauto,
                                'blnlayers': layers},
                 'defaults': {}, 'selection': None, 'selections': {},
                 'features': []}]}]
        return made

    a = model('m', 1.0, 0.1, 5, 4)
    b = model('mm', 1e-3, 100.0, 5, 4)
    assert kinds(mk.compare(a, b)) == []
    b = model('mm', 1e-3, 100.0, 4, 8)
    assert sorted(i['name'] for i in mk.compare(a, b)) == \
        ['blnlayers', 'hauto']


def interface(tag, features, kind='HeatTransfer'):
    return {'tag': tag, 'path': f'comp1/{tag}', 'identifier': tag,
            'type': kind, 'label': 'Heat', 'active': True, 'settings': {},
            'defaults': {}, 'selection': {'level': 'domain',
                                          'geometry': 'geom1',
                                          'entities': 'all'},
            'features': features}


def test_same_text_other_nodes():
    # b's 'ht' is another interface than a's: the message says so
    flux = node('hf1', 'HeatFluxBoundary', faces(1), {'q0': 'ht.T'},
                {'q0': '0'})
    a = described([flux])
    b = described()
    moved = {**copy.deepcopy(flux), 'path': 'comp1/ht2/hf1'}
    b['components'][0]['physics'] = [interface('ht2', [moved]),
                                     interface('ht', [], 'Other')]
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert item['message'].endswith("ht.T in a, ht.T in b (in a's tags: "
                                    "<b only:ht>.T)")
    # long values are compared whole, shown cut
    long = 'ht.T + ' + ' + '.join(['1'] * 60)
    flux['properties']['q0'] = moved['properties']['q0'] = long
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert "(in a's tags: <b only:ht>.T + 1" in item['message']
    assert item['message'].endswith("...)")
    # values that read differently need no note
    moved['properties']['q0'] = 'ht.T*2'
    [item] = [i for i in mk.compare(a, b) if i['kind'] == 'property']
    assert "a's tags" not in item['message']


def test_variables_with_swapped_tags():
    # 'comp2.T' in both, but b's comp2 is a's comp1 (by place)
    def model(names):
        found = described(geometries=[geometry(
            [box((0, 1), (0, 1), (0, 1), 1.0)], cube_faces())])
        second = copy.deepcopy(found['components'][0])
        second['geometries'][0] = geometry(
            [box((5, 6), (0, 1), (0, 1), 1.0)], cube_faces(5, 6))
        first, other = names
        found['components'][0]['tag'] = first
        second['tag'] = other
        found['components'].append(second)
        found['variables'] = [{
            'tag': 'var1', 'path': 'var1', 'type': 'Variables',
            'label': 'Variables 1', 'active': True,
            'variables': {'x': 'comp2.T'}, 'selection': None}]
        return found

    a, b = model(('comp1', 'comp2')), model(('comp2', 'comp1'))
    found = [i for i in mk.compare(a, b) if i['kind'] == 'variable']
    assert len(found) == 1
    assert "(in a's tags: comp1.T)" in found[0]['message']


def test_mesh_default_in_its_unit():
    # a in m leaves hmax at its default, b in mm sets it: a's value from
    # b's default is shown in a's unit
    def model(unit, scale, size, hmax=None, default=None):
        made = described(geometries=[geometry(
            [box((0, size), (0, size), (0, size), size**3)],
            cube_faces(0, size), unit=unit, scale=scale)])
        made['components'][0]['meshes'] = [{
            'tag': 'mesh1', 'path': 'comp1/mesh1', 'label': 'Mesh 1',
            'geometry': 'geom1', 'automatic': False, 'features': [
                {'tag': 'size', 'path': 'comp1/mesh1/size', 'type': 'Size',
                 'label': 'Size', 'active': True,
                 'properties': {} if hmax is None else {'hmax': hmax},
                 'defaults': {} if hmax is None else {'hmax': default},
                 'selection': None, 'selections': {}, 'features': []}]}]
        return made

    a = model('m', 1.0, 1.0)
    b = model('mm', 1e-3, 1000.0, hmax=200.0, default=150.0)
    [item] = mk.compare(a, b)
    assert (item['a'], item['b']) == (0.15, 200.0)
    [item] = mk.compare(b, a)
    assert (item['a'], item['b']) == (200.0, 0.15)
    # the same length in both units: no difference
    b = model('mm', 1e-3, 1000.0, hmax=150.0, default=150.0)
    assert mk.compare(a, b) == []
    # an expression stays as it is
    b = model('mm', 1e-3, 1000.0, hmax=200.0, default='L/10')
    [item] = mk.compare(a, b)
    assert item['a'] == 'L/10'


def test_meshes_of_steps():
    def model(meshes, used):
        made = described(studies=[{
            'tag': 'std1', 'path': 'std1', 'label': 'Study 1',
            'active': True, 'solver': {'status': 'automatic',
                                       'sequence': None},
            'steps': [{'tag': 'stat', 'path': 'std1/stat',
                       'type': 'Stationary', 'label': 'Stationary',
                       'active': True,
                       'properties': {'mesh': {'geom1': used}},
                       'defaults': {'mesh': {'geom1': 'mesh1'}},
                       'selection': None, 'selections': {},
                       'features': []}]}])
        made['components'][0]['meshes'] = [{
            'tag': tag, 'path': f'comp1/{tag}', 'label': tag,
            'geometry': 'geom1', 'automatic': False, 'features': [
                {'tag': 'size', 'path': f'comp1/{tag}/size', 'type': 'Size',
                 'label': 'Size', 'active': True,
                 'properties': {'hmax': hmax}, 'defaults': {'hmax': 0.1},
                 'selection': None, 'selections': {}, 'features': []}]}
            for tag, hmax in meshes]
        return made

    # the step's mesh in a's tags
    a = model([('mesh1', 0.2), ('mesh2', 0.05)], 'mesh2')
    b = model([('mesh1', 0.2), ('mesh3', 0.05)], 'mesh3')
    assert mk.compare(a, b) == []
    assert mk.compare(b, a) == []
    b = model([('mesh1', 0.2), ('mesh3', 0.05)], 'mesh3')
    a = model([('mesh1', 0.2), ('mesh2', 0.05)], 'mesh1')
    [item] = mk.compare(a, b)
    assert (item['a'], item['b']) == ('mesh1', 'mesh3')
    assert item['message'].endswith("(in a's tags: mesh2)")
    assert mirrored(a, b)
    # meshes of one geometry pair by tag first, then in order
    a = model([('mesh1', 0.2), ('mesh2', 0.05)], 'mesh2')
    b = model([('mesh2', 0.05), ('mesh1', 0.2)], 'mesh2')
    assert mk.compare(a, b) == []


def kernels(sizes, kernel, finalize='union', end=3.0):
    """Unit cubes in a row with the given sizes (the last one ending at
    `end`), on a geometry kernel."""
    domains = [box((2 * i, 2 * i + 1), (0, 1), (0, 1), size)
               for i, size in enumerate(sizes)]
    domains[-1]['x'][1] = end + 2 * (len(sizes) - 2)
    return described(geometries=[geometry(
        domains, [], finalize=finalize, representation=kernel)])


def test_geometry_kernels():
    # the same entities measured otherwise by another kernel
    a = kernels([1.0, 1.0], 'comsol')
    b = kernels([1.0, 1.0013], 'cadps')
    [item] = mk.compare(a, b)
    assert item['kind'] == 'geometry'
    assert "geometry differs: a uses the COMSOL kernel, b the CAD kernel" \
        in item['message']
    assert "geomRep('comsol') in b before building)" in item['message']
    assert '1 in the same place as one in b, sizes up to 0.13 % apart' \
        in item['message']
    assert (item['a']['representation'], item['b']['representation']) \
        == ('comsol', 'cadps')
    assert json.loads(json.dumps(b)) == b
    [back] = mk.compare(b, a)
    assert 'a uses the CAD kernel, b the COMSOL kernel' in back['message']
    assert "geomRep('cadps') in b before building where a CAD kernel is " \
        'installed' in back['message']
    # the same kernel: only the sizes
    [item] = mk.compare(kernels([1.0, 1.0], 'comsol'),
                        kernels([1.0, 1.0013], 'comsol'))
    assert 'kernel' not in item['message']
    assert '0.13 % apart' in item['message']
    # the largest of several
    [item] = mk.compare(kernels([1.0, 1.0, 1.0], 'comsol'),
                        kernels([1.0011, 1.0, 1.0013], 'cadps'))
    assert '2 in the same place as one in b, sizes up to 0.13 %' in \
        item['message']
    # a kernel COMSOL may add later is named as it is
    [item] = mk.compare(a, kernels([1.0, 1.0013], 'other'))
    assert "geometry representation 'other'" in item['message']
    # union against assembly: the sizes, not the kernel
    found = mk.compare(a, kernels([1.0, 1.0013], 'cadps', 'assembly'))
    [unchecked] = [i for i in found if i['kind'] == 'unchecked']
    assert '0.13 % apart' in unchecked['message']
    assert 'kernel' not in ' '.join(i['message'] for i in found)


def test_geometry_kernels_alone():
    # kernels that differ without entities in the same place: no note
    a = kernels([1.0, 1.0], 'comsol')
    assert mk.compare(a, kernels([1.0, 1.0], 'cadps')) == []
    for end in (3.1, 3.0005):
        found = mk.compare(a, kernels([1.0, 1.0], 'cadps', end=end))
        assert found and 'kernel' not in found[0]['message']
        assert 'same place' not in found[0]['message']


def test_applies_nowhere():
    overridden = {'level': 'boundary', 'geometry': 'geom1',
                  'entities': faces(1)['entities'], 'applied': []}
    a = described()
    b = described([node('hf1', 'HeatFluxBoundary', overridden)])
    [item] = mk.compare(a, b)
    assert item['message'].endswith('only in b (applies nowhere)')
    assert item['empty'] is True
    b = described([node('hf1', 'HeatFluxBoundary', {
        'level': 'boundary', 'geometry': 'geom1', 'entities': []})])
    [item] = mk.compare(a, b)
    assert item['message'].endswith('only in b (selects nothing)')


def test_unchecked_kept_apart():
    # two reasons for one node: both kept
    selection = {'level': 'several', 'levels': ['boundary', 'domain'],
                 'entities': 'unknown'}
    a = described([node('hf1', 'HeatFluxBoundary', selection,
                        {'q0': '1'}, {})])
    b = described([node('hf1', 'HeatFluxBoundary', selection, {}, {})])
    found = [i['message'] for i in mk.compare(a, b)
             if i['kind'] == 'unchecked']
    assert len(found) == 2
    assert any('defaults unknown' in m for m in found)
    assert any('selection not compared' in m for m in found)


def test_messages_read_plainly():
    place = {'x': [-3.9e-19, 0.04], 'y': [0.0, 0.04], 'z': [-0.06, -0.06]}
    assert _compare.place_text(place) == 'x 0..0.04, y 0..0.04, z=-0.06'
    assert _compare.shown(None) == '(none)'
    assert _compare.numbers('range(0,1e-7,1)') is None
    assert _compare.numbers('range(0,0.5,1)') == [0, 0.5, 1]


def test_ignore_and_format():
    model = described()
    with pytest.raises(ValueError, match='does not know'):
        mk.compare(model, model, ignore={'labels'})
    with pytest.raises(ValueError, match='describe the model again'):
        mk.compare({**model, 'format': 1}, model)
    with pytest.raises(TypeError, match='models or results'):
        mk.compare(model, 'old.json')
    old = copy.deepcopy(model)
    assert kinds(mk.compare(old, model, ignore={'unchecked'})) == []


##########
# Models #
##########

@pytest.fixture
def two(client):
    """Two fresh models, removed after the test."""
    models = [client.create('cmpa'), client.create('cmpb')]
    yield models
    for model in models:
        client.remove(model)


def heat_plate(model, *, unit='mm', thickness=10, halves=False,
               reverse=False, T0='Th', h='h0', extra=None):
    """
    A 100 × 50 × thickness plate (in `unit`, as 0.1 × 0.05 m), hot at
    x = 0 and cooled on top, made of one block or two halves, its
    features made in either order.
    """
    model.parameter('Th', '100[degC]')
    model.parameter('h0', '10[W/(m^2*K)]')
    size = 100 if unit == 'mm' else 0.1
    geom = mk.geometry(model, 3, length_unit=unit)
    depth = thickness if unit == 'mm' else thickness / 1000
    if halves:
        parts = [mk.block(geom, (size / 2, size / 2, depth)),
                 mk.block(geom, (size / 2, size / 2, depth),
                          (size / 2, 0, 0))]
        mk.union(geom, parts, intbnd=False)
    else:
        mk.block(geom, (size, size / 2, depth))
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)

    def hot():
        node = heat.create('TemperatureBoundary', 2)
        node.java.selection().set(mk.sel.entities(
            geom, mk.sel.box(geom, 'boundary', x=0)))
        node.property('T0', T0)

    def cool():
        node = heat.create('HeatFluxBoundary', 2)
        node.java.selection().set(mk.sel.entities(
            geom, mk.sel.box(geom, 'boundary', z=depth)))
        node.property('HeatFluxType', 'ConvectiveHeatFlux')
        node.property('h', h)

    for make in ((cool, hot) if reverse else (hot, cool)):
        make()
    if extra:
        extra(model, geom, heat)
    return geom


def test_same_plate_built_otherwise(two):
    a, b = two
    heat_plate(a)
    heat_plate(b, halves=True, reverse=True)
    old, new = mk.describe(a), mk.describe(b)
    found = mk.compare(old, new)
    assert kinds(found) == [], [i['message'] for i in found]
    assert not [i for i in found if i['kind'] == 'unchecked']
    # read back from JSON: the same result
    again = mk.compare(json.loads(json.dumps(old)),
                       json.loads(json.dumps(new)))
    assert again == found


def test_one_property(two):
    a, b = two
    heat_plate(a)
    heat_plate(b, h='20[W/(m^2*K)]')
    [item] = [i for i in mk.compare(a, b) if i['kind'] != 'note']
    assert item['kind'] == 'property'
    assert item['name'] == 'h'
    assert (item['a'], item['b']) == ('h0', '20[W/(m^2*K)]')


def test_empty_feature(two):
    a, b = two
    heat_plate(a)

    def empty(model, geom, heat):
        heat.create('HeatFluxBoundary', 2)

    heat_plate(b, extra=empty)
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_b']
    assert found[0]['empty'] is True
    assert kinds(mk.compare(a, b, ignore={'empty'})) == []
    # the other way round
    assert kinds(mk.compare(b, a)) == ['only_in_a']


def test_units_of_length(two):
    a, b = two
    heat_plate(a, unit='mm')
    heat_plate(b, unit='m')
    found = mk.compare(a, b)
    assert kinds(found) == [], [i['message'] for i in found]


@pytest.mark.parametrize('one, other, expected', [
    ('100[degC]', '373.15[K]', []), ('Th', '100[degC]', ['expression']),
    ('100[degC]', '373.15', ['expression']), ('Th', '2*Th', ['property'])])
def test_values_in_si(two, one, other, expected):
    a, b = two
    heat_plate(a, T0=one)
    heat_plate(b, T0=other)
    assert kinds(mk.compare(a, b)) == expected


def test_thicker_plate(two):
    a, b = two
    heat_plate(a)
    heat_plate(b, thickness=12)
    old, new = mk.describe(a), mk.describe(b)
    found = mk.compare(old, new)
    assert kinds(found) == ['geometry']
    back = mk.compare(new, old)
    assert len(back[0]['consequences']) == len(found[0]['consequences'])
    [item] = found[:1]
    assert item['consequences']
    assert all(c['kind'] in ('selection', 'property', 'only_in_a',
                             'only_in_b', 'applied')
               for c in item['consequences'])


def test_moved_condition(two):
    # the hot face moves to x = 100: same area, other face
    a, b = two
    heat_plate(a)

    def moved(model, geom, heat):
        hot = heat/'Temperature 1'
        hot.java.selection().set(mk.sel.entities(
            geom, mk.sel.box(geom, 'boundary', x=100)))

    heat_plate(b, extra=moved)
    found = mk.compare(a, b)
    assert kinds(found) == ['selection']
    assert found[0]['numbers']['a'] and found[0]['numbers']['b']
    assert 'x=0' in found[0]['message'] and 'x=100' in found[0]['message']
    # the insulation now applies on the other face: a consequence
    [linked] = found[0]['consequences']
    assert linked['kind'] == 'applied'
    assert linked['path']['a'].endswith('/ins1')
    assert linked['causes'] == [found[0]['path']]
    assert found[0]['message'].endswith('(+1 consequence)')


def test_periodic_face_removed(two):
    # b leaves out one face of the periodic condition: the insulation
    # applies there instead, a consequence
    for model, both in zip(two, (True, False)):
        def periodic(model, geom, heat, both=both):
            node = heat.create('PeriodicHeat', 2)
            chosen = mk.sel.entities(geom, mk.sel.box(geom, 'boundary', y=0))
            if both:
                chosen += mk.sel.entities(
                    geom, mk.sel.box(geom, 'boundary', y=50))
            node.java.selection().set(chosen)
        heat_plate(model, extra=periodic)
    found = mk.compare(*two)
    assert 'applied' not in kinds(found), [i['message'] for i in found]
    [item] = [i for i in found if i.get('consequences')]
    assert item['kind'] == 'selection'
    [linked] = item['consequences']
    assert linked['kind'] == 'applied'
    assert linked['path']['a'].endswith('/ins1')
    assert linked['causes'] == [item['path']]


def test_overlap_order(two):
    # two conditions that share a face, made in the other order: where
    # they apply differs, and nothing else explains it
    for model, reverse in zip(two, (False, True)):
        def overlapping(model, geom, heat, reverse=reverse):
            def make(planes):
                node = heat.create('TemperatureBoundary', 2)
                node.java.selection().set([
                    n for plane in planes for n in mk.sel.entities(
                        geom, mk.sel.box(geom, 'boundary', **plane))])
            first, second = [{'y': 0}, {'z': 0}], [{'z': 0}, {'x': 100}]
            for planes in ((second, first) if reverse else (first, second)):
                make(planes)
        heat_plate(model, extra=overlapping)
    found = mk.compare(*two)
    assert kinds(found) == ['applied', 'applied'], \
        [i['message'] for i in found]
    assert not any('causes' in i for i in found)


def test_later_feature_disabled(two):
    # temp2 takes the hot face from temp1 in a only
    for model, active in zip(two, (True, False)):
        def later(model, geom, heat, active=active):
            node = heat.create('TemperatureBoundary', 2)
            node.java.selection().set(mk.sel.entities(
                geom, mk.sel.box(geom, 'boundary', x=0)))
            node.java.active(active)
        heat_plate(model, extra=later)
    found = mk.compare(*two)
    assert kinds(found) == ['active']
    assert [c['path']['a'].rsplit('/', 1)[1]
            for c in found[0]['consequences']] == ['temp1']


def test_interface_disabled_in_one(two):
    a, b = two
    heat_plate(a)
    heat_plate(b, extra=lambda model, geom, heat: heat.java.active(False))
    found = mk.compare(a, b)
    assert kinds(found) == ['active']
    assert found[0]['path']['a'] == 'comp1/ht'
    # the features read as disabled too; where disabled nodes apply is
    # not compared
    assert {c['kind'] for c in found[0]['consequences']} == {'active'}


def test_overridden_by_sibling(two):
    # b's second solid takes domain 2 from solid1 and its subnodes
    a, b = two
    for model in two:
        geom = two_blocks(model)
        heat = (model/'physics').create('HeatTransfer', geom)
    solid = heat.create('SolidHeatTransferModel', 3)
    solid.java.selection().set([2])
    found = mk.compare(a, b)
    assert kinds(found) == ['only_in_b']
    linked = [c['path']['a'] for c in found[0]['consequences']]
    assert 'comp1/ht/solid1' in linked
    assert all(c['kind'] == 'applied' for c in found[0]['consequences'])


def test_operators_by_name(two):
    # operators with other tags (and names) called in expressions
    for model, tag in zip(two, ('intop1', 'intop2')):
        def operator(model, geom, heat, tag=tag):
            component = mk.component_of(geom).java
            made = component.cpl().create(tag, 'Integration')
            made.selection().all()
            (heat/'Heat Flux 1').property('h', f'h0*comp1.{tag}(1)/5e-4')
        heat_plate(model, extra=operator)
    found = mk.compare(*two)
    assert kinds(found) == [], [i['message'] for i in found]


def cylinder_model(model, rotated):
    geom = mk.geometry(model, 3)
    made = mk.cylinder(geom, 1, 2)
    if rotated:
        mk.rotate(geom, [made], 37)
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    side = heat.create('HeatFluxBoundary', 2)
    side.java.selection().set([
        n for n in mk.sel.entities(geom, mk.sel.box(geom, 'boundary'))
        if n not in mk.sel.entities(geom, mk.sel.box(geom, 'boundary', z=0))
        and n not in mk.sel.entities(geom, mk.sel.box(geom, 'boundary',
                                                      z=2))])
    return geom


def test_curved_faces(two):
    # the side of a cylinder has its seams elsewhere when rotated
    a, b = two
    cylinder_model(a, False)
    cylinder_model(b, True)
    found = mk.compare(a, b)
    assert kinds(found) == [], [i['message'] for i in found]


def test_hole(two):
    a, b = two
    heat_plate(a)
    geom = mk.geometry(b, 3, length_unit='mm')
    plate = mk.block(geom, (100, 50, 10))
    mk.difference(geom, plate, [mk.cylinder(geom, 5, 10, (30, 25, 0))])
    b.build(geom)
    b.parameter('Th', '100[degC]')
    b.parameter('h0', '10[W/(m^2*K)]')
    found = mk.compare(a, b)
    assert kinds(found)[0] == 'geometry'


def two_blocks(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    model.build(geom)
    return geom


def test_material_level_and_variable(two):
    for model, on_boundary in zip(two, (False, True)):
        geom = two_blocks(model)
        component = mk.component_of(geom).java
        made = component.material().create('mat1', 'Common')
        made.propertyGroup('def').set('density', '1000[kg/m^3]')
        if on_boundary:
            made.selection().geom(str(geom.java.tag()), 2)
            made.selection().set([1])
        variables = component.variable().create('var1')
        variables.set('q', '2')
        if on_boundary:
            variables.selection().geom(str(geom.java.tag()), 3)
            variables.selection().all()
    found = mk.compare(*two)
    assert sorted(kinds(found)) == ['selection', 'variable']


def test_study_steps(two):
    for model, off in zip(two, ((0,), (0, 1))):
        geom = two_blocks(model)
        for name in ('HeatTransfer', 'HeatTransfer'):
            (model/'physics').create(name, geom)
        study = (model/'studies').create()
        step = study.create('Stationary')
        tags = [str(t) for t in mk.component_of(geom).java.physics().tags()]
        pairs = []
        for i, tag in enumerate(tags):
            pairs += [tag, 'off' if i in off else 'on']
        step.java.set('activate', pairs)
    found = mk.compare(*two)
    assert kinds(found) == ['property']
    assert found[0]['name'] == f'activate[{tags[1]}]'


def test_disabled_physics_by_tag(two):
    # the same study, physics made in another order (other tags)
    for model, order in zip(two, (('ht', 'ec'), ('ec', 'ht'))):
        geom = two_blocks(model)
        made = {}
        for name in order:
            kind = 'HeatTransfer' if name == 'ht' else 'ConductiveMedia'
            node = (model/'physics').create(kind, geom)
            made[name] = node.tag()
        study = (model/'studies').create()
        step = study.create('Stationary')
        step.java.set('disabledphysics', [made['ec']])
    found = mk.compare(*two)
    assert kinds(found) == [], [i['message'] for i in found]


def test_swapped_interfaces(two):
    # two heat interfaces on all domains, tags swapped
    for model, swap in zip(two, (False, True)):
        geom = two_blocks(model)
        first = (model/'physics').create('HeatTransfer', geom)
        second = (model/'physics').create('HeatTransfer', geom)
        hot, cold = (second, first) if swap else (first, second)
        hot.create('TemperatureBoundary', 2).java.selection().set([1])
        cold.create('HeatFluxBoundary', 2).java.selection().set([4])
        cold.java.feature('init1').set('Tinit', '300[K]')
    found = mk.compare(*two)
    assert kinds(found) == [], [i['message'] for i in found]


def test_linked_material(two):
    a, b = two
    geom = two_blocks(a)
    made = a.java.material().create('gm1', 'Common', '')
    made.propertyGroup('def').set('density', '1000[kg/m^3]')
    link = mk.component_of(geom).java.material().create('lnk1', 'Link')
    link.set('link', 'gm1')
    geom = two_blocks(b)
    own = mk.component_of(geom).java.material().create('mat1', 'Common')
    own.propertyGroup('def').set('density', '1000[kg/m^3]')
    found = mk.compare(a, b)
    assert kinds(found) == [], [i['message'] for i in found]


def test_same_applied(two):
    # a material on everything overridden on block 2, against two
    # materials on one block each
    for model, everything in zip(two, (True, False)):
        geom = two_blocks(model)
        component = mk.component_of(geom).java
        first = component.material().create('mat1', 'Common')
        first.propertyGroup('def').set('density', '1')
        if not everything:
            first.selection().set([1])
        second = component.material().create('mat2', 'Common')
        second.propertyGroup('def').set('density', '2')
        second.selection().set([2])
    found = mk.compare(*two)
    assert kinds(found) == ['selection']
    assert found[0]['same_applied'] is True
    assert kinds(mk.compare(*two, ignore={'same_applied'})) == []


def test_interpolation_from_file(two, tmp_path):
    table = tmp_path/'table.txt'
    table.write_text('0 1\n1 2\n2 5\n')
    for model, source in zip(two, ('table', 'file')):
        geom = two_blocks(model)
        function = model.java.func().create(f'int{len(source)}',
                                            'Interpolation')
        if source == 'table':
            function.set('funcname', 'Tdep')
            function.set('table', [['0', '1'], ['1', '2'], ['2', '5']])
        else:
            function.set('source', 'file')
            function.set('filename', str(table))
            function.set('funcs', [['Tdep', '1']])
        variables = mk.component_of(geom).java.variable().create('var1')
        variables.set('w', 'Tdep(1.5)')
    found = mk.compare(*two)
    # the function itself is set up differently; its calls are the same
    assert 'variable' not in kinds(found)
    assert all(i['path']['a'] == 'int5' for i in found
               if i['kind'] == 'property')


def test_probes(two):
    for model, tag in zip(two, ('dom1', 'dom7')):
        geom = two_blocks(model)
        component = mk.component_of(geom).java
        probe = component.probe().create(tag, 'Domain')
        probe.set('probename', 'average')
    assert kinds(mk.compare(*two)) == []
    # in 3D a probe's surface integral changes nothing
    probe.set('intsurface', 'on')
    assert kinds(mk.compare(*two)) == []
    assert 'intsurface' in mk.describe(two[1])['probes'][0]['unused']
    points = []
    for model in two:
        component = mk.component_of(model/'geometries'/'Geometry 1').java
        points.append(component.probe().create('pt1', 'DomainPoint'))
    assert kinds(mk.compare(*two)) == []
    points[1].set('coords3', ['0.5', '0.5', '0.5'])
    found = [i for i in mk.compare(*two) if i['kind'] == 'property']
    # COMSOL keeps the point in `coords` too
    assert sorted(i['name'] for i in found) == ['coords', 'coords3']
    points[1].set('coords3', points[0].getStringArray('coords3'))
    expression = points[1].feature(points[1].feature().tags()[0])
    expression.set('expr', 'T^2')
    found = [i for i in mk.compare(*two) if i['kind'] == 'property']
    # (COMSOL follows the expression with the unit)
    assert 'expr' in [i['name'] for i in found]
    assert all(i['path']['b'].endswith('/ppb1') for i in found)
    mk.component_of(two[1]/'geometries'/'Geometry 1').java.probe().create(
        'bnd1', 'Boundary')
    assert 'only_in_b' in kinds(mk.compare(*two))


def test_step_mesh(two):
    # a step on a second mesh, tagged otherwise in b
    for model, tag in zip(two, ('mesh2', 'mesh7')):
        geom = two_blocks(model)
        (model/'physics').create('HeatTransfer', geom)
        meshes = model.java.component(
            str(mk.component_of(geom).java.tag())).mesh()
        second = meshes.create(tag)
        second.create('ftet1', 'FreeTet')
        second.feature('size').set('hauto', '7')
        # a new step takes the last mesh made: this one is not the default
        meshes.create('mesh1').create('ftet1', 'FreeTet')
        study = (model/'studies').create(name='study')
        step = study.create('Stationary')
        step.java.set('mesh', [str(geom.java.tag()), tag])
    [step] = mk.describe(two[1])['studies'][0]['steps']
    assert step['properties']['mesh'] == {'geom1': 'mesh7'}
    assert kinds(mk.compare(*two)) == []


def test_geometry_kernel(two):
    # a flat block measures the same on both kernels: no difference
    for model, kernel in zip(two, ('cadps', 'comsol')):
        geom = mk.geometry(model, 3)
        geom.java.geomRep(kernel)
        mk.block(geom, (1, 2, 3))
        model.build(geom)
    found = [mk.describe(m)['components'][0]['geometries'][0]
             for m in two]
    assert [g['representation'] for g in found] == ['cadps', 'comsol']
    assert mk.compare(*two) == []
    # 2D and 1D geometries have one too
    for dim in (2, 1):
        geom = mk.geometry(two[0], dim)
        mk.feature(geom, 'Square' if dim == 2 else 'Interval')
        two[0].build(geom)
    kernels = [g['representation'] for c in mk.describe(two[0])[
        'components'] for g in c['geometries']]
    assert len(kernels) == 3 and all(k in ('cadps', 'comsol')
                                     for k in kernels)


def test_time_list(two):
    for model, times in zip(two, ('range(0,0.1,1)',
                                  ' '.join(f'{i / 10:g}'
                                           for i in range(11)))):
        two_blocks(model)
        study = (model/'studies').create()
        study.create('Transient').property('tlist', times)
    assert kinds(mk.compare(*two)) == []
    (two[1]/'studies'/'Study 1'/'Time Dependent').property(
        'tlist', 'range(0,0.1,2)')
    assert kinds(mk.compare(*two)) == ['property']


def test_global_equations(two):
    for model, split in zip(two, (False, True)):
        geom = two_blocks(model)
        physics = mk.component_of(geom).java.physics().create(
            'ge', 'GlobalEquations', str(geom.java.tag()))
        first = physics.feature('ge1')
        rows = [('u1', 'u1t+u1'), ('u2', 'u2-1')]
        if split:
            second = physics.create('ge2', 'GlobalEquations', -1)
            for feature, (name, equation) in zip((second, first), rows):
                feature.setIndex('name', name, 0, 0)
                feature.setIndex('equation', equation, 0, 0)
        else:
            for i, (name, equation) in enumerate(rows):
                first.setIndex('name', name, i, 0)
                first.setIndex('equation', equation, i, 0)
    found = mk.compare(*two)
    assert kinds(found) == [], [i['message'] for i in found]


def test_two_geometries(two):
    # a selection on the second geometry of a component
    for model, face in zip(two, (1, 1)):
        geom = two_blocks(model)
        component = mk.component_of(geom).java
        other = component.geom().create('geom2', 3)
        other.create('blk1', 'Block').set('size', ['2', '2', '2'])
        other.run()
        heat = component.physics().create('ht', 'HeatTransfer', 'geom2')
        heat.create('temp1', 'TemperatureBoundary', 2).selection().set(
            [face])
    assert kinds(mk.compare(*two)) == []
    heat = mk.component_of(two[1]/'geometries'/'Geometry 1').java \
        .physics('ht')
    heat.feature('temp1').selection().set([2])
    found = mk.compare(*two)
    assert kinds(found) == ['selection']
    assert 'x=2' in found[0]['message'] or 'x 0..2' in found[0]['message']


@pytest.mark.parametrize('partial', [False, True])
def test_assembly(two, partial):
    # a chip on a plate: an assembly with pairs against a union
    for model, action in zip(two, ('assembly', 'union')):
        geom = mk.geometry(model, 3)
        mk.block(geom, (10, 10, 1))
        size = 2 if partial else 10
        offset = 4 if partial else 0
        mk.block(geom, (size, size, 1), (offset, offset, 1))
        geom.java.feature('fin').set('action', action)
        model.build(geom)
        heat = (model/'physics').create('HeatTransfer', geom)
        heat.create('TemperatureBoundary', 2).java.selection().set(
            mk.sel.entities(geom, mk.sel.box(geom, 'boundary', z=0)))
    found = mk.compare(*two)
    first = found[0]
    assert first['kind'] == 'geometry'
    assert first['a'] == 'assembly' and first['b'] == 'union'
    assert 'consequences' not in first
    rest = kinds(found)[1:]
    if partial:
        assert 'unchecked' in rest
    assert 'selection' not in rest
    # union against assembly changes where the insulation and the
    # continuity apply; both are explained
    assert 'applied' not in rest
    [cont] = [i for i in found if i['kind'] == 'active']
    assert cont['path']['a'].endswith('/dcont1')
    assert [c['path']['a'].rsplit('/', 1)[1]
            for c in cont['consequences']] == ['ins1']
    # the continuity is off in the union: where it applies is not
    # compared, its 'active' item says it
    [pair] = [i for i in found if i['kind'] == 'only_in_a']
    assert 'consequences' not in pair


def test_library_material(two):
    a, b = two
    geom = two_blocks(a)
    mk.material(geom, 'Copper')
    geom = two_blocks(b)
    own = mk.component_of(geom).java.material().create('mat1', 'Common')
    group = own.propertyGroup('def')
    group.set('thermalconductivity', ['400[W/(m*K)]'])
    group.set('density', '8960[kg/m^3]')
    group.set('heatcapacity', '385[J/(kg*K)]')
    found = mk.compare(a, b, show={'material_info'})
    names = [i.get('name') for i in found if i['kind'] == 'property'
             and not i.get('material_info')]
    assert not {'thermalconductivity', 'density', 'heatcapacity'} & \
        set(names)
    for item in found:
        if item['kind'] == 'property' and 'name' not in item:
            # only the library's own entries, one item per group
            assert 'thermalconductivity' not in (item['a'] or {})
    # the library's coordinate system entry is marked as such
    assert all(i.get('material_info') for i in found
               if i.get('name') == 'sys')
    assert not [i for i in mk.compare(a, b, ignore={'material_info'})
                if i.get('name') == 'sys']


def test_physics_controlled_mesh(two):
    a, b = two
    geom = two_blocks(a)
    mesh = (a/'meshes').create(geom)
    mesh.java.automatic(True)
    mesh.java.autoMeshSize(4)
    geom = two_blocks(b)
    mesh = (b/'meshes').create(geom)
    size = mesh.create('Size')
    mk.set(size, custom='on', hmax=0.3)
    mesh.create('FreeTet')
    layers = mesh.create('BndLayer')
    layers.java.selection().geom(str(geom.java.tag()), 3)
    layers.java.selection().all()
    found = mk.compare(a, b)
    # the physics-controlled size node against the script's own: hauto 4
    # against 5; the script's Size is another type, so it is extra
    assert kinds(found) == ['property', 'only_in_b', 'only_in_b',
                            'only_in_b'], [i['message'] for i in found]
    assert (found[0]['name'], found[0]['a'], found[0]['b']) == \
        ('hauto', 4.0, 5.0)
    assert not [i for i in found if i.get('name') == 'hmax']


def test_material_in_expression(two):
    # 'mat1.def.rho' in one model is 'mat7.def.rho' in the other
    for model, tag in zip(two, ('mat1', 'mat7')):
        geom = two_blocks(model)
        component = mk.component_of(geom).java
        made = component.material().create(tag, 'Common')
        made.propertyGroup('def').set('density', '1000[kg/m^3]')
        component.variable().create('var1').set('m', f'{tag}.def.rho*2')
    found = mk.compare(*two)
    assert kinds(found) == [], [i['message'] for i in found]
