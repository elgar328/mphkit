"""
Checks the lookups of COMSOL's names: physics and feature types,
properties and variables. The catalogue tests run without COMSOL.
"""
import importlib.util
from pathlib import Path

import pytest

import mphkit as mk
from conftest import java_export
from mphkit import _catalog

example = Path(__file__).parents[1]/'examples'/'plate_with_holes.py'
spec = importlib.util.spec_from_file_location('plate_with_holes', example)
plate_with_holes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plate_with_holes)

KINDS = ('physics', 'geom', 'mesh', 'study', 'material')
# Interfaces that COMSOL creates only without a geometry
GLOBAL_ONLY = ('GlobalAlphaBetaPhaseTransformation',
               'GlobalAusteniteDecomposition', 'GlobalMetalPhaseTransformation',
               'BoltzmannEquation')


#############
# Catalogue #
#############

@pytest.fixture
def offline(monkeypatch):
    """Reads the catalogue of the installed COMSOL without starting it."""
    discovery = pytest.importorskip('mph.discovery')
    try:
        root = str(discovery.backend()['root'])
    except Exception as error:
        pytest.skip(f'COMSOL not found: {error}')
    monkeypatch.setattr(_catalog, '_root', lambda: root)
    return root


def interface(type='HeatTransfer'):
    """The catalogue entry of a 3D physics interface."""
    top, _ = _catalog.catalogue('physics')
    element = top.find(f'Physics/S3D/{type}')
    return _catalog.Entry('physics', type, None, None, {}, element)


def heat_transfer():
    return interface('HeatTransfer')


def search(entry, words):
    """Types and how they match, as feature_types orders them."""
    return [(e.type, how) for how, e in
            _catalog.search_types(_catalog.children(entry), words.split())]


@pytest.mark.parametrize('descr, expected', [
    ('Heat_transfer_coefficient#', 'Heat transfer coefficient'),
    ('Frequency_x #1', 'Frequency x (1)'),
    (' Error_estimates_and_residuals', 'Error estimates and residuals'),
    ('#PNG', 'PNG'),
    ('Difference[set]', 'Difference'),
    ('Export[verb]...', 'Export...'),
    ('Rectangle[Beam_cross_section_type]', 'Rectangle'),
    ('Local_z-coordinate_[-1,1]_for_thickness-dependent_results',
     'Local z-coordinate [-1,1] for thickness-dependent results'),
    ('Phase_transition_between_phase_X_and_phase_Y#1#2',
     'Phase transition between phase X and phase Y (1, 2)'),
    ('', None),
    (None, None),
])
def test_text(descr, expected):
    assert _catalog.text(descr) == expected


def test_catalogue_walk(offline):
    features = _catalog.children(heat_transfer())
    types = [f.type for f in features]
    assert len(types) == 73
    assert {'TemperatureBoundary', 'HeatFluxBoundary',
            'GlobalEquations'} <= set(types)
    flux = features[types.index('HeatFluxBoundary')]
    assert (flux.description, flux.tag) == ('Heat flux', 'hf')
    assert flux.props['HeatFluxType'].choices['ConvectiveHeatFlux'] \
        == 'Convective heat flux'
    assert flux.props['h'].description == 'Heat transfer coefficient'
    # study.xml lists some study steps twice
    steps = [s.type for s in _catalog.children(_catalog._study_steps())]
    assert steps.count('Parametric') == 1 and 'Transient' in steps


@pytest.mark.parametrize('words, expected', [
    ('emissivity', {'SurfaceToAmbientRadiation':
                    (3, False, 'epsilon_rad (Surface emissivity)')}),
    ('heat transfer coefficient',
     {'HeatFluxBoundary':
      (3, False, 'HeatTransferCoefficientType (Heat transfer coefficient)')}),
    ('convective', {'ConvectiveOutflow': (1, False, None),
                    'HeatFluxBoundary':
                    (2, False, 'HeatFluxType = ConvectiveHeatFlux '
                               '(Convective heat flux)')}),
    ('initial temperature', {'init': (3, True, 'Tinit (Temperature)')}),
])
def test_search_stages(offline, words, expected):
    found = {f.type: _catalog.stage(f, words.split())
             for f in _catalog.children(heat_transfer())}
    for type, how in expected.items():
        assert found[type] == how


def test_no_model_junk(offline):
    # the catalogue was made from a sample model; its names are left out
    features = _catalog.children(heat_transfer())
    for feature in features:
        for prop in feature.props.values():
            assert not any(value.startswith('root.')
                           for value in prop.choices or {})
            assert not (prop.default or '').startswith('root.')
            assert not any('(ht)' in (descr or '')
                           for descr in (prop.choices or {}).values())
        system = feature.props.get('coordinateSystem')
        if system is not None:
            assert 'sys1' not in (system.choices or {})
            assert system.default == 'GlobalSystem'
    for words in ('temperature', 'heat flux', 'prescribed temperature'):
        assert not any('root.' in (how[2] or '')
                       for _, how in search(heat_transfer(), words))
    # choices that are all names in the model
    layer = _catalog._child(interface('SolidMechanics'), 'ThinLayer')
    spring = _catalog._child(layer, 'SpringMaterial')
    system = spring.props['coordinateSystem']
    assert system.choices is None and system.default is None


# Choices of the sample model the catalogue was made from, found by the
# node label rule: (value, description)
NODE_LABELS = {
    ('frame:material1', '#Moving Mesh 1'), ('gp1', '#Grain Properties 1'),
    ('phase1', '#Metallurgical Phase 1'), ('phase2', '#Metallurgical Phase 2'),
    ('pp1', '#Particle Properties 1'), ('sa1', '#Spacecraft Axes 1'),
    ('so1', '#Spacecraft Orientation 1'), ('sys1', 'Boundary System 1'),
    ('wp1', '#Wall 1'),
}


def raw_choices(kind):
    """The (property, value, description) triples of a catalogue."""
    _, props = _catalog.catalogue(kind)
    for element in props.values():
        if element.get('values') is None:
            continue
        values = element.get('values').split('|')
        descrs = (element.get('descrs') or '').split('|')
        if len(descrs) != len(values):
            descrs = [''] * len(values)
        for value, descr in zip(values, descrs):
            yield element, value, descr


def test_no_model_junk_anywhere(offline):
    # the node label rule catches exactly the sample model's nodes
    found = [(value, descr) for kind in KINDS
             for _, value, descr in raw_choices(kind)
             if _catalog._node_label(value, descr)]
    assert set(found) == NODE_LABELS and len(found) == 36
    for value, descr in [('matern32', 'Matern32'),
                         ('legacy52', 'Legacy_version_52'),
                         ('shcurl2', 'Curl_type_2'), ('CO2', '#CO2'),
                         ('Face1', '#Face1'),
                         ('intTransmittance10', 'Transmittance#10')]:
        assert not _catalog._node_label(value, descr)
    # known leftovers no rule catches without dropping real choices
    left = {}
    for element, value, _ in raw_choices('study'):
        if element.get('name') in ('disableFrameControl', 'pcontinuation'):
            left.update(_catalog._prop(element).choices or {})
    assert left['frame:spatial1'] is None
    assert left['freq'] == 'freq' and left['lambda0'] == 'lambda0'
    assert 'frame:material1' not in left


def fake_physics(tmp_path, monkeypatch, *, features=True, groups=None,
                 code='pa', descr=True, text=None):
    """
    Writes a small physics catalogue of the expected format, or one
    broken as asked, and points the catalogue at it.
    """
    groups = groups or ('S1D', 'S2D', 'S3D', 'S1DAxi', 'S2DAxi')
    interfaces = ''.join(
        f'<{g}><HeatTransfer descr="Heat" id="ht" features="aa"/></{g}>'
        for g in groups)
    listed = ('<Features><aa name="HeatSource" descr="Heat_source" id="hs" '
              f'props="{code}"/></Features>' if features else '')
    described = ' descr="Heat_source"' if descr else ''
    if text is None:
        text = ('<COMSOLCompletionData><model.physics>'
                f'<Physics><Global/>{interfaces}</Physics>{listed}'
                f'<Properties><pa name="Q0"{described}/></Properties>'
                '</model.physics></COMSOLCompletionData>')
    folder = tmp_path/'data'/'completion'
    folder.mkdir(parents=True)
    (folder/'physics.xml').write_text(text)
    monkeypatch.setattr(_catalog, '_root', lambda: str(tmp_path))


def test_catalogue_format_small(tmp_path, monkeypatch):
    fake_physics(tmp_path, monkeypatch)
    top, props = _catalog.catalogue('physics')
    assert list(props) == ['pa']


@pytest.mark.parametrize('broken', [
    {'features': False}, {'groups': ('S1D', 'S2D', 'S3D', 'S1DAxi')},
    {'code': 'zz'}, {'descr': False}, {'text': '<COMSOLCompletionData>'},
])
def test_catalogue_format_broken(tmp_path, monkeypatch, broken):
    fake_physics(tmp_path, monkeypatch, **broken)
    with pytest.raises(RuntimeError, match="COMSOL's catalogue of names at "
                                           '.* has an unexpected format'):
        _catalog.catalogue('physics')


def test_catalogue_format_installed(offline):
    for kind in KINDS:
        assert _catalog.catalogue(kind)[1]


def test_search_order(offline):
    # matches that need words of the type's own name come last
    found = search(heat_transfer(), 'external temperature')
    assert found[-1][0] == 'ExternalRadiationSource'
    assert found[-1][1][:2] == (3, True)
    assert [how[:2] for _, how in found[:-1]] == [(3, False)] * 5


def test_small_lists(offline):
    # in short lists every property counts
    pair = _catalog._child(interface('SolidMechanics'), 'ThinLayerPair')
    assert [t for t, _ in search(pair, 'density')] == [
        'HyperelasticModel', 'LinearElasticModel', 'NonlinearElasticMaterial']
    # settings most types share count when nothing else matches
    steps = _catalog._study_steps()
    assert len(search(steps, 'geometric nonlinearity')) == 62
    assert len(search(steps, '')) == 81


@pytest.mark.parametrize('where, level', [
    ('Domain 1', 'domain'), ('Boundaries 1–14', 'boundary'),
    ('Edges 1–36', 'edge'), ('Point 3', 'point'), ('Global', 'global'),
    ('Randbereiche 1–14', None),
])
def test_where_level(where, level):
    assert _catalog.where_level(where) == level


def test_catalogue_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_catalog, '_root', lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="catalogue of names was not "
                                           'found'):
        _catalog.catalogue('physics')


###########################
# With a model, in COMSOL #
###########################

@pytest.fixture(scope='module')
def plate(client):
    """The solved example plate: model, geometry, selections."""
    model, geom, selections = plate_with_holes.build_model(client, 2)
    model.solve()
    yield model, geom, selections
    client.remove(model)


def test_physics_types(plate, model):
    _, geom, _ = plate
    found = mk.physics_types(geom)
    types = [item['type'] for item in found]
    assert types == sorted(types, key=_catalog.name_key)
    assert {'type': 'HeatTransfer', 'description': 'Heat transfer in solids',
            'tag': 'ht'} in found
    assert 'GlobalEquations' in types and 'ModelEntityImage' not in types
    assert not set(GLOBAL_ONLY) & set(types)
    heat = mk.physics_types(geom, search='HEAT transfer')
    assert heat and all('heat' in str(i).lower() and 'transfer'
                        in str(i).lower() for i in heat)
    # an axisymmetric geometry has its own list
    axi = mk.geometry(model, 2)
    axi.java.axisymmetric(True)
    top, _ = _catalog.catalogue('physics')
    listed = sorted((e.tag for e in top.find('Physics/S2DAxi')
                     if e.tag != 'ModelEntityImage'), key=_catalog.name_key)
    assert [i['type'] for i in mk.physics_types(axi)] == listed
    solid = mk.geometry(model, 3)
    with pytest.raises(TypeError, match='takes a geometry'):
        mk.physics_types(mk.workplane(solid, quickz=1))


def test_feature_types(plate, tmp_path):
    model, geom, selections = plate
    heat = model/'physics'/'heat'
    before = java_export(model, tmp_path/'model.java')
    flux = mk.integral(geom, 'boundary', 'ht.ntflux', selections['hot end'])
    found = {f['type']: f for f in mk.feature_types(heat)}
    assert len(found) == 73
    assert found['TemperatureBoundary']['levels'] == {'boundary': 2}
    assert found['HeatSource']['levels'] == {'domain': 3}
    assert found['PointHeatSource']['levels'] == {'point': 0}
    assert found['GlobalEquations']['levels'] == {'global': -1}
    assert found['WeakInequalityConstraint']['levels'] == \
        {'domain': 3, 'boundary': 2, 'edge': 1, 'point': 0}
    assert found['LocalThermalNonequilibriumBoundary']['levels'] == {}
    assert all(f['match'] is None for f in found.values())
    # subfeatures go on their parent's level
    assert mk.feature_types(heat/'hot end') == [
        {'type': 'TemperatureHarmonicPerturbation',
         'description': 'Harmonic perturbation', 'tag': 'temphp',
         'levels': {'boundary': 2}, 'match': None}]
    solid = next(c for c in heat.children()
                 if c.java.getType() == 'SolidHeatTransferModel')
    change = mk.feature_types(solid, search='phase change')
    assert change[0]['type'] == 'PhaseChangeMaterial'
    assert change[0]['levels'] == {'domain': 3}
    # the solved model is left as it was
    assert java_export(model, tmp_path/'model.java') == before
    assert mk.integral(geom, 'boundary', 'ht.ntflux',
                       selections['hot end']) == flux


def test_feature_search(plate):
    heat = plate[0]/'physics'/'heat'
    found = mk.feature_types(heat, search='convective')
    assert found[0]['type'] == 'ConvectiveOutflow'
    assert found[0]['match'] is None
    flux = next(f for f in found if f['type'] == 'HeatFluxBoundary')
    assert flux['match'] == ('HeatFluxType = ConvectiveHeatFlux '
                             '(Convective heat flux)')
    assert flux['levels'] == {'boundary': 2}
    # types found by name come first
    steps = mk.feature_types(plate[0]/'studies'/'static', search='time')
    assert 'Transient' in [s['type'] for s in steps]
    named = [s['match'] is None for s in steps]
    assert named[0] and named == sorted(named, reverse=True)


def test_feature_types_lists(plate, model):
    # geometry, mesh and study types, without levels
    plate_model, geom, _ = plate
    found = mk.feature_types(geom)
    assert len(found) == 97
    block = next(f for f in found if f['type'] == 'Block')
    assert block['levels'] is None and block['match'] is None
    mesh = plate_model/'meshes'/'mesh'
    assert len(mk.feature_types(mesh)) == 22
    free = mesh.create('FreeTet')
    try:
        assert [f['type'] for f in mk.feature_types(free)] == [
            'CornerRefinement', 'Distribution', 'Size', 'SizeExpression']
    finally:
        free.remove()
    steps = mk.feature_types(plate_model/'studies'/'static')
    assert len(steps) == 81 and 'Transient' in [s['type'] for s in steps]
    # in a work plane
    solid = mk.geometry(model, 3)
    plane = mk.workplane(solid, quickz=1)
    found = mk.feature_types(plane)
    assert len(found) == 84 and 'Circle' in [f['type'] for f in found]
    assert all(f['levels'] is None for f in found)


@pytest.mark.parametrize('axisymmetric', [False, True])
def test_feature_types_other_dimensions(model, axisymmetric):
    flat = mk.geometry(model, 2)
    if axisymmetric:
        flat.java.axisymmetric(True)
    mk.square(flat, 1)
    model.build(flat)
    assert len(mk.feature_types(flat)) == 84
    mesh = (model/'meshes').create(flat)
    assert len(mk.feature_types(mesh)) == 19


def test_properties_ignore_unmatched_entry(plate, monkeypatch):
    # a catalogue entry that fits none of the node's properties is ignored
    cooling = plate[0]/'physics'/'heat'/'cooling'
    stranger = _catalog.Prop('nothing', 'Nothing', None, None)
    unmatched = _catalog.Entry('physics', 'Stranger', None, None,
                               {'nothing': stranger}, None)
    monkeypatch.setattr(_catalog, '_existing_entry',
                        lambda node, java: unmatched)
    found = mk.properties(cooling)
    assert found['h']['value'] == '200[W/(m^2*K)]'
    assert all(row['description'] is None for row in found.values())


def test_feature_types_not_in_catalogue(plate, monkeypatch):
    heat = plate[0]/'physics'/'heat'
    monkeypatch.setattr(_catalog, '_interface', lambda *args: None)
    with pytest.raises(RuntimeError, match='not in the catalogue of COMSOL'):
        mk.feature_types(heat)


@pytest.mark.parametrize('dim, level', [(2, {'boundary': 1}),
                                        (1, {'point': 0})])
def test_feature_levels_lower_dimensions(model, dim, level):
    geom = mk.geometry(model, dim)
    if dim == 2:
        mk.square(geom, 1)
    else:
        mk.interval(geom, [0, 1])
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    found = mk.feature_types(heat, search='TemperatureBoundary')
    assert found[0]['levels'] == level


def test_physics_without_geometry(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    model.build(geom)
    events = (model/'physics').create('Events')
    with pytest.raises(TypeError, match='has no geometry'):
        mk.feature_types(events)
    with pytest.raises(TypeError, match='has no geometry'):
        mk.variables(events)
    with pytest.raises(TypeError, match='has no geometry'):
        mk.properties(events, 'DiscreteStates')
    settings = mk.properties(events)
    assert all(row['description'] is None for row in settings.values())


def test_wrong_nodes(plate):
    model, geom, _ = plate
    heat = model/'physics'/'heat'
    with pytest.raises(TypeError, match='mk.feature_types takes a physics '
                                        'interface or feature'):
        mk.feature_types(model/'materials'/'steel')
    with pytest.raises(TypeError, match='no properties of its own'):
        mk.properties(geom)
    with pytest.raises(TypeError, match='physics interface or a geometry'):
        mk.variables(heat/'cooling')
    with pytest.raises(TypeError, match='search must be a string'):
        mk.variables(heat, search=['flux'])
    with pytest.raises(TypeError, match='takes a model node'):
        mk.properties('heat')


def test_properties_physics(plate):
    model = plate[0]
    heat = model/'physics'/'heat'
    cooling = mk.properties(heat/'cooling')
    assert cooling['HeatFluxType'] == {
        'value': 'ConvectiveHeatFlux', 'description': 'Flux type',
        'default': 'GeneralInwardHeatFlux',
        'choices': {'GeneralInwardHeatFlux': 'General inward heat flux',
                    'ConvectiveHeatFlux': 'Convective heat flux',
                    'NucleateBoilingHeatFlux': 'Nucleate boiling heat flux',
                    'HeatRate': 'Heat rate'}}
    assert cooling['h'] == {'value': '200[W/(m^2*K)]',
                            'description': 'Heat transfer coefficient',
                            'default': None, 'choices': None}
    assert list(cooling) == sorted(cooling, key=_catalog.name_key)
    # internal properties are left out
    assert 'StudyStep' not in cooling and 'editModelInputs' not in cooling
    # choices that depend on the model, without a description
    materials = cooling['matList']['choices']
    assert materials['dommat'] is not None
    assert list(materials)[1:] == ['mat1'] and materials['mat1'] is None
    # search
    found = mk.properties(heat/'cooling', search='transfer coefficient')
    assert 'h' in found and 'HeatFluxType' not in found
    # before creating one
    before = mk.properties(heat, 'HeatFluxBoundary')
    assert before['HeatFluxType']['value'] is None
    assert before['HeatFluxType']['default'] == 'GeneralInwardHeatFlux'
    assert set(before) >= set(cooling)
    sub = mk.properties(heat/'hot end', 'TemperatureHarmonicPerturbation')
    assert sub
    # the physics interface's own settings
    settings = mk.properties(heat)
    order = settings['ShapeProperty/order_temperature']
    assert order['value'] == '2' and '1' in order['choices']
    assert not any(k.startswith(('ObsoleteProperty/', 'StudyStep/'))
                   for k in settings)
    with pytest.raises(ValueError, match="Did you mean 'HeatFluxBoundary'"):
        mk.properties(heat, 'HeatFlux')


def test_properties_geometry(plate, model):
    geom = plate[1]
    assert mk.properties(geom/'plate')['size']['value'] == [100, 40, 5]
    assert mk.properties(geom/'plate')['size']['description'] == 'Size'
    difference = next(c for c in geom.children()
                      if c.java.getType() == 'Difference')
    assert mk.properties(difference)['input']['value'] == ['blk1']
    # a feature missing from the catalogue: all properties, no descriptions
    final = next(c for c in geom.children()
                 if c.java.getType() == 'Finalize')
    assert all(row['description'] is None
               for row in mk.properties(final).values())
    assert mk.properties(geom, 'Cone')['r']['description'] == 'Bottom radius'
    # entities picked from an object, and features in a work plane
    other = mk.geometry(model, 3)
    box = mk.block(other, (1, 1, 1))
    chamfer = mk.chamfer(other, box, 0.1)
    plane = mk.workplane(other, quickz=2)
    circle = mk.circle(plane, 0.5)
    model.build(other)
    edges = mk.properties(chamfer)['edge']['value']
    assert list(edges) == ['blk1'] and edges['blk1'] == list(range(1, 13))
    assert mk.properties(circle)['r']['value'] == 0.5
    assert mk.properties(circle)['r']['description'] == 'Radius'
    assert 'r' in mk.properties(plane, 'Circle')


def test_properties_mesh_study_material(plate):
    model = plate[0]
    mesh = model/'meshes'/'mesh'
    size = mk.properties(mesh.children()[0])        # MeshSizeDefault
    assert size['hmax']['description'] == 'Maximum element size'
    assert size['table']['choices']['cfd'] == 'Fluid dynamics'
    assert 'hmax' in mk.properties(mesh, 'Size')
    free = mesh.create('FreeTet')
    try:
        assert 'hmax' in mk.properties(free, 'Size')    # a subfeature
        assert mk.properties(free)['method']['description']
    finally:
        free.remove()
    step = mk.properties(model/'studies'/'static'/'stationary')
    stationary = _catalog._child(_catalog._study_steps(), 'Stationary')
    assert set(step) <= set(stationary.props) and 'useparam' in step
    assert 'tlist' in mk.properties(model/'studies'/'static', 'Transient')
    basic = mk.properties(model/'materials'/'steel'/'Basic')
    assert basic['thermalconductivity']['value'] == ['44.5[W/(m*K)]']
    assert basic['thermalconductivity']['description'] is None
    steel = mk.properties(model/'materials'/'steel')
    assert 'thickness' in steel and 'bndType' not in steel


def test_properties_not_in_catalogue(model):
    # the catalogue lists no properties for some features
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    model.build(geom)
    try:
        acoustics = (model/'physics').create('PressureAcoustics', geom)
    except Exception as error:
        pytest.skip(f'No acoustics license: {error}')
    initial = next(c for c in acoustics.children()
                   if c.java.getType() == 'init')
    found = mk.properties(initial)
    assert found and all(r['description'] is None for r in found.values())
    assert mk.properties(acoustics, 'init') == {}


def test_variables(plate, model):
    heat = plate[0]/'physics'/'heat'
    found = mk.variables(heat)
    assert found[0] == {'name': 'T', 'unit': None,
                        'description': 'Dependent variable (temperature)',
                        'where': None, 'levels': None}
    assert 'T' in [v['name'] for v in mk.variables(heat, search='temperature')]
    names = [v['name'] for v in found if v['where'] is not None]
    assert names == sorted(names, key=_catalog.name_key)
    flux = next(v for v in mk.variables(heat, search='normal heat flux')
                if v['name'] == 'ht.ntflux')
    assert flux['unit'] == 'W/m^2' and flux['levels'] == ['boundary']
    temperature = next(v for v in found if v['name'] == 'ht.Tvar')
    assert temperature['levels'] == ['domain', 'boundary', 'edge', 'point']
    assert temperature['where'][0] == 'Domain 1'
    # all physics on a geometry
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    with pytest.raises(RuntimeError, match='not built or empty'):
        mk.variables(geom)
    model.build(geom)
    (model/'physics').create('HeatTransfer', geom)
    (model/'physics').create('SolidMechanics', geom)
    found = mk.variables(geom)
    dependent = [v['name'] for v in found if v['where'] is None]
    assert {'T', 'u', 'v', 'w'} <= set(dependent)
    assert [v['name'] for v in found[:len(dependent)]] == dependent
    names = [v['name'] for v in found]
    assert 'solid.mises' in names and 'ht.ntflux' in names


def test_variables_surface_geometry(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1), type='surface')
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    assert mk.variables(heat)       # built, although it has no domains


def test_lookups_leave_nothing(plate, tmp_path):
    model, geom, _ = plate
    heat = model/'physics'/'heat'
    before = java_export(model, tmp_path/'model.java')
    mk.physics_types(geom, search='heat')
    mk.feature_types(heat, search='flux')
    mk.feature_types(geom)
    mk.feature_types(model/'meshes'/'mesh')
    mk.feature_types(model/'studies'/'static')
    mk.properties(heat/'cooling')
    mk.properties(heat)
    mk.properties(heat, 'HeatFluxBoundary')
    mk.variables(geom)
    assert java_export(model, tmp_path/'model.java') == before


def test_without_catalogue(plate, monkeypatch, tmp_path):
    model, geom, _ = plate
    heat = model/'physics'/'heat'
    monkeypatch.setattr(_catalog, '_root', lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match='catalogue of names'):
        mk.physics_types(geom)
    with pytest.raises(RuntimeError, match='catalogue of names'):
        mk.feature_types(heat)
    cooling = mk.properties(heat/'cooling')
    assert cooling['h']['value'] == '200[W/(m^2*K)]'
    assert cooling['h']['description'] is None
    assert cooling['HeatFluxType']['choices']['HeatRate'] is None
    assert mk.properties(heat)
    assert any(v['name'] == 'ht.ntflux' for v in mk.variables(heat))


def test_readme_names(plate):
    # the lookups in the README, on the example script's model
    import re
    readme = (Path(__file__).parents[1]/'README.md').read_text()
    lines = re.search(r"COMSOL's names, looked up.*?```python\n(.*?)```",
                      readme, re.S).group(1).splitlines()
    assert len(lines) == 7, 'update the checks below with the README'
    model, geom, _ = plate
    namespace = {'mk': mk, 'geom': geom, 'heat': model/'physics'/'heat',
                 'mesh': model/'meshes'/'mesh',
                 'study': model/'studies'/'static'}
    physics, features, meshing, steps, before, cooling, found = [
        eval(line.split('#')[0], namespace) for line in lines]
    assert 'HeatTransfer' in [p['type'] for p in physics]
    assert features[0]['type'] == 'ConvectiveOutflow'
    flux = next(f for f in features if f['type'] == 'HeatFluxBoundary')
    assert flux['levels'] == {'boundary': 2} and flux['match']
    assert {'FreeTet', 'Size'} <= {m['type'] for m in meshing}
    assert 'Transient' in [s['type'] for s in steps]
    assert before['HeatFluxType']['choices']
    assert cooling['HeatFluxType']['value'] == 'ConvectiveHeatFlux'
    flux = next(v for v in found if v['name'] == 'ht.ntflux')
    assert flux['unit'] == 'W/m^2' and flux['levels'] == ['boundary']
