"""
Checks mk.check, the check before solving, on models with deliberate
mistakes and on the example script's model.
"""
import json
import re
from pathlib import Path

import pytest

import mphkit as mk
from conftest import java_export, read
from test_example import plate_with_holes

root = Path(__file__).parents[1]


def blocks(model, count=2, dim=3):
    """Unit blocks (squares, intervals) side by side along x, built."""
    geom = mk.geometry(model, dim)
    for n in range(count):
        if dim == 3:
            mk.block(geom, (1, 1, 1), (n, 0, 0))
        elif dim == 2:
            mk.square(geom, 1, (n, 0))
        else:
            mk.interval(geom, [n, n + 1])
    model.build(geom)
    return geom


def solvable(model, geom):
    """Adds a mesh and a stationary study."""
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')


def kinds(found, severity=None):
    return [(p['kind'], p['node']) for p in found
            if severity is None or p['severity'] == severity]


def item(found, kind, node):
    matches = [p for p in found if p['kind'] == kind and p['node'] == node]
    assert len(matches) == 1, found
    return matches[0]


###########
# Example #
###########

def test_plate(client, tmp_path):
    # the example is correct: only the boundaries left insulated
    model, geom, selections = plate_with_holes.build_model(client, 2)
    try:
        for solved in (False, True):
            if solved:
                model.solve()
                heat = mk.integral(geom, 'boundary', 'ht.ntflux',
                                   selections['hot end'], unit='W')
            mesh = model.java.mesh((model/'meshes'/'mesh').tag())
            before = (java_export(model, tmp_path/'model.java'),
                      model.datasets(), client.models(), mesh.isEmpty(),
                      mesh.getNumElem())
            found = mk.check(model)
            assert kinds(found) == [('default_condition',
                                     'physics/heat/Thermal Insulation 1')]
            assert found[0]['severity'] == 'info'
            assert found[0]['level'] == 'boundary'
            assert (java_export(model, tmp_path/'model.java'),
                    model.datasets(), client.models(), mesh.isEmpty(),
                    mesh.getNumElem()) == before
            assert mk.check(model) == found
            json.dumps(found)
        assert mk.integral(geom, 'boundary', 'ht.ntflux',
                           selections['hot end'], unit='W') == heat
    finally:
        client.remove(model)


#########
# Units #
#########

def test_units(model):
    geom = blocks(model, 2)
    model.parameter('Th', '100[degC]')
    model.parameter('T0', '300')
    model.parameter('p', '1')
    steel = mk.material(geom, 'Structural steel', [1])
    steel.java.propertyGroup('def').set('density', '7850[m]')
    # a global material through a link, and an unused one
    java = model.java
    used = java.material().create('gm1', 'Common', '')
    used.label('global')
    used.propertyGroup('def').set('density', '7850[m]')
    unused = java.material().create('gm2', 'Common', '')
    unused.label('unused')
    unused.propertyGroup('def').set('density', '5[m]')
    component = mk.component_of(geom).java
    component.material().create('lnk1', 'Link')
    component.material('lnk1').set('link', 'gm1')
    component.material('lnk1').selection().set([2])
    component.material().create('lnk2', 'Link')     # the same, once more
    component.material('lnk2').set('link', 'gm1')
    component.material('lnk2').selection().set([2])
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    for name, x, value in (('hot', 0, '5[m]'), ('fine', 2, 'Th')):
        temperature = heat.create('TemperatureBoundary', 2, name=name)
        temperature.select(mk.sel.box(geom, 'boundary', x=x))
        temperature.property('T0', value)
    plain = heat.create('TemperatureBoundary', 2, name='plain')
    plain.select(mk.sel.box(geom, 'boundary', y=0))
    plain.property('T0', 'T0')
    source = heat.create('HeatSource', 3, name='src')
    source.select([1])
    source.property('Q0', '1e3[W/m^2]')
    flux = heat.create('HeatFluxBoundary', 2, name='flux')
    flux.select(mk.sel.box(geom, 'boundary', y=1))
    flux.property('HeatFluxType', 'ConvectiveHeatFlux')
    flux.property('h', '10*p[W/(m^2*K)]')
    flux.property('Text', '0[K]*unknown + 100[degC]')
    solid = (model/'physics').create('SolidMechanics', geom, name='solid')
    load = solid.create('BoundaryLoad', 2, name='load')
    load.select(mk.sel.box(geom, 'boundary', x=2))
    load.property('FperArea', ['1[MPa]', '2[m]', '2[m]'])
    solvable(model, geom)
    found = [p for p in mk.check(model) if p['kind'] == 'unit_mismatch']
    messages = sorted(p['message'].split('.')[0] for p in found)
    assert messages == sorted([
        '"global" (Basic): density = 7850[m] is in m, not kg/m^3',
        '"hot": T0 = 5[m] is in m, not K',
        '"load": forceReferenceArea[1] = 2[m] is in m, not N/m^2',
        '"load": forceReferenceArea[2] = 2[m] is in m, not N/m^2',
        '"src": Q0 = 1e3[W/m^2] is in W/m^2, not W/m^3',
        '"Structural steel" (Basic): density = 7850[m] is in m, not kg/m^3'])
    hot = item(found, 'unit_mismatch', 'physics/heat/hot')
    assert 'write it in K (e.g. 5[K])' in hot['message']
    # the link to the global material counts as a material
    assert 'no_material' not in [p['kind'] for p in mk.check(model)]
    assert hot['severity'] == 'warning'
    item(found, 'unit_mismatch', 'materials/Structural steel/Basic')
    # fixed, it is gone
    (heat/'hot').property('T0', '5[K]')
    assert 'physics/heat/hot' not in [p['node'] for p in mk.check(model)]


#############
# Materials #
#############

def test_no_material(model):
    geom = blocks(model, 3)
    mk.material(geom, 'Structural steel', [1, 2])
    disabled = mk.material(geom, 'Aluminum', [3])
    disabled.java.active(False)
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    heat.create('TemperatureBoundary', 2).select(
        mk.sel.box(geom, 'boundary', x=0))
    solid = (model/'physics').create('SolidMechanics', geom, name='solid')
    solid.create('Fixed', 2).select(mk.sel.box(geom, 'boundary', x=0))
    solvable(model, geom)
    found = mk.check(model)
    for node, properties in (('physics/heat/Solid 1', 'Cp, k, rho'),
                             ('physics/solid/Linear Elastic Material 1',
                              'E, nu, rho')):
        missing = item(found, 'no_material', node)
        assert missing['entities'] == [3] and missing['level'] == 'domain'
        assert f'takes {properties} from a material, but domain 3 has ' \
               'none' in missing['message']
        assert 'mk.materials(search=' in missing['message']
    # properties typed in need no material
    elastic = solid.java.feature('lemm1')
    for name, value in (('E', '200e9'), ('nu', '0.3'), ('rho', '7850')):
        elastic.set(f'{name}_mat', 'userdef')
        elastic.set(name, value)
    found = mk.check(model)
    assert [p['node'] for p in found if p['kind'] == 'no_material'] == \
        ['physics/heat/Solid 1']
    # a material fixes it
    mk.material(geom, 'Copper', [3])
    assert not [p for p in mk.check(model) if p['kind'] == 'no_material']


def test_no_material_boundary_coupling(model):
    # a coupling on boundaries does not exempt domains
    geom = blocks(model, 2)
    acoustics = (model/'physics').create('PressureAcoustics', geom,
                                         name='acoustics')
    acoustics.java.selection().set([1])
    solid = (model/'physics').create('SolidMechanics', geom, name='solid')
    solid.java.selection().set([2])
    component = mk.component_of(geom).java
    component.multiphysics().create('asb1', 'AcousticStructureBoundary', 2)
    component.multiphysics('asb1').selection().all()
    solvable(model, geom)
    assert [(p['node'], p['entities']) for p in mk.check(model)
            if p['kind'] == 'no_material'] == [
        ('physics/acoustics/Pressure Acoustics 1', [1]),
        ('physics/solid/Linear Elastic Material 1', [2])]


def test_no_material_coupling_known_miss(model):
    # domains with a multiphysics coupling are skipped, since couplings
    # may supply properties; thermal expansion does not, a known miss
    geom = blocks(model, 1)
    (model/'physics').create('HeatTransfer', geom, name='heat')
    (model/'physics').create('SolidMechanics', geom, name='solid')
    solvable(model, geom)
    assert [p['kind'] for p in mk.check(model)].count('no_material') == 2
    component = mk.component_of(geom).java
    component.multiphysics().create('te1', 'ThermalExpansion', 3)
    component.multiphysics('te1').selection().all()
    assert 'no_material' not in [p['kind'] for p in mk.check(model)]


##############
# Selections #
##############

def test_selections(model):
    geom = blocks(model, 3)
    mk.material(geom, 'Structural steel')
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    heat.java.selection().set([1, 2])           # not on domain 3
    # first, so that it is no later feature: it applies on interior
    # boundaries too (a thin insulating layer there)
    everywhere = heat.create('ThermalInsulation', 2, name='everywhere')
    everywhere.java.selection().all()
    inner = mk.sel.box(geom, 'boundary', x=1)
    heat.create('HeatFluxBoundary', 2, name='inner').select(inner)
    heat.create('TemperatureBoundary', 2, name='far').select(
        mk.sel.box(geom, 'boundary', x=3))
    heat.create('LineHeatSource', 1, name='line').select(
        mk.sel.box(geom, 'edge', x=3, y=0))
    numbered = heat.create('TemperatureBoundary', 2, name='numbered')
    numbered.select(mk.sel.find(geom, 'boundary', x=3))
    heat.create('HeatFluxBoundary', 2, name='nothing').select(
        mk.sel.box(geom, 'boundary', x=(7, 8)))
    front = mk.sel.box(geom, 'boundary', y=0, x=(0, 1))
    heat.create('HeatFluxBoundary', 2, name='flux').select(front)
    heat.create('TemperatureBoundary', 2, name='temperature').select(front)
    heat.create('HeatFluxBoundary', 2, name='bottom').select(
        mk.sel.box(geom, 'boundary', z=0))
    solvable(model, geom)
    found = mk.check(model)
    warnings = kinds(found, 'warning')
    assert warnings == [('empty_selection', 'physics/heat/nothing'),
                        ('not_applied', 'physics/heat/inner'),
                        ('not_applied', 'physics/heat/far'),
                        ('not_applied', 'physics/heat/line'),
                        ('not_applied', 'physics/heat/numbered')]
    applied = item(found, 'not_applied', 'physics/heat/inner')
    assert applied['level'] == 'boundary'
    assert applied['entities'] == mk.sel.entities(geom, inner)
    call = re.search(r'mk\.sel\.entities\(geom, (.*?)\)\.',
                     applied['message']).group(1)
    assert mk.sel.entities(geom, eval(call)) == applied['entities']
    numbers = mk.sel.find(geom, 'boundary', x=3)
    assert f'boundary {numbers[0]} by number' in item(
        found, 'not_applied', 'physics/heat/numbered')['message']
    assert item(found, 'not_applied', 'physics/heat/line')['level'] == 'edge'
    later = item(found, 'not_applied', 'physics/heat/flux')
    assert later['severity'] == 'info'
    assert 'the later "temperature" applies there' in later['message']
    partial = item(found, 'not_applied', 'physics/heat/bottom')
    assert partial['severity'] == 'info' and 'of its selection' in \
        partial['message']
    assert 'physics/heat/everywhere' not in [p['node'] for p in found]
    assert 'physics/heat/temperature' not in [p['node'] for p in found]
    # warnings first
    severities = [p['severity'] for p in found]
    assert severities == sorted(severities, key=lambda s: s != 'warning')
    # disabled features are not checked
    (heat/'inner').java.active(False)
    assert 'physics/heat/inner' not in [p['node'] for p in mk.check(model)]


def test_default_conditions(model):
    geom = blocks(model, 1)
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    solvable(model, geom)
    default = item(mk.check(model), 'default_condition',
                   'physics/heat/Thermal Insulation 1')
    assert default['entities'] == [1, 2, 3, 4, 5, 6]
    assert default['message'].startswith('Boundaries 1, 2, 3, 4, 5, 6 keep')
    heat.create('TemperatureBoundary', 2).select(
        mk.sel.box(geom, 'boundary', x=(-1, 2), y=(-1, 2), z=(-1, 2)))
    assert 'default_condition' not in [p['kind'] for p in mk.check(model)]


@pytest.mark.parametrize('dim, axisymmetric', [(2, False), (2, True),
                                               (1, False)])
def test_other_dimensions(model, dim, axisymmetric):
    geom = mk.geometry(model, dim)
    if axisymmetric:
        geom.java.axisymmetric(True)
    if dim == 2:
        mk.rectangle(geom, (1, 2))
    else:
        mk.interval(geom, [0, 1])
    model.build(geom)
    mk.material(geom, 'Structural steel')
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    solvable(model, geom)
    found = mk.check(model)
    default = item(found, 'default_condition',
                   'physics/heat/Thermal Insulation 1')
    assert default['level'] == ('boundary' if dim == 2 else 'point')
    assert not kinds(found, 'warning')
    if axisymmetric:
        # the axis is no default condition, and nothing applies on it
        assert 'Axial Symmetry' not in str(found)
        axis = heat.create('TemperatureBoundary', 1, name='axis')
        axis.select(mk.sel.box(geom, 'boundary', x=0))
        assert kinds(mk.check(model), 'warning') == [
            ('not_applied', 'physics/heat/axis')]


def test_assembly_and_shell(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    geom.java.feature('fin').set('action', 'assembly')
    model.build(geom)
    mk.material(geom, 'Structural steel')
    (model/'physics').create('HeatTransfer', geom, name='heat')
    solid = (model/'physics').create('SolidMechanics', geom, name='solid')
    solid.create('Contact', 2, name='contact')      # a pair feature
    shell = (model/'physics').create('Shell', geom, name='shell')
    solvable(model, geom)
    found = mk.check(model)
    nodes = [p['node'] for p in found]
    assert 'physics/solid/contact' not in nodes
    # continuity on the pairs is a pair feature too
    assert 'physics/heat/Continuity 1' not in nodes
    assert not [p for p in found if (p['node'] or '').startswith(
        str(shell) + '/')]          # closed surfaces: no free edges
    assert None not in nodes


def test_shell(model):
    # a shell on a single face: its free edges keep the default "Free"
    geom = mk.geometry(model, 3)
    plane = mk.workplane(geom)
    mk.rectangle(plane, (1, 1))
    model.build(geom)
    shell = (model/'physics').create('Shell', geom, name='shell')
    solvable(model, geom)
    found = mk.check(model)
    free = item(found, 'default_condition', str(shell/'Free 1'))
    assert free['level'] == 'edge' and len(free['entities']) == 4
    assert [p['kind'] for p in found] == ['default_condition']


#####################
# Meshes and studies #
#####################

def test_meshes_and_studies(model):
    geom = blocks(model, 1)
    mk.material(geom, 'Structural steel')
    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    found = mk.check(model)
    assert kinds(found, 'warning') == [('no_mesh', None),
                                       ('not_solved', None)]
    assert "create(model/'geometries/Geometry 1')" in found[0]['message']
    assert 'no study' in found[1]['message']
    (model/'meshes').create(geom)
    # a study with a parametric sweep only solves nothing
    sweep = (model/'studies').create(name='sweep')
    model.java.study(sweep.tag()).create('param', 'Parametric')
    assert kinds(mk.check(model), 'warning') == [('not_solved',
                                                  'physics/heat')]
    sweep.remove()
    study = (model/'studies').create(name='study')
    step = study.create('Stationary', name='step')
    assert not kinds(mk.check(model), 'warning')
    java = model.java.study(study.tag()).feature(step.tag())
    java.setEntry('activate', heat.tag(), False)
    assert kinds(mk.check(model), 'warning') == [('not_solved',
                                                  'physics/heat')]
    java.setEntry('activate', heat.tag(), True)
    java.setEntry('activate', 'comp1', False)
    assert kinds(mk.check(model), 'warning') == [('not_solved',
                                                  'physics/heat')]
    java.setEntry('activate', 'comp1', True)
    # a parametric sweep has no physics list of its own
    model.java.study(study.tag()).create('param', 'Parametric')
    assert not kinds(mk.check(model), 'warning')
    # a disabled physics is not checked
    heat.java.active(False)
    assert mk.check(model) == []


#########
# Rules #
#########

def test_rules(client, model, tmp_path):
    with pytest.raises(TypeError, match='takes a model'):
        mk.check(model/'physics')
    assert mk.check(model) == []                # no components
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    other = mk.geometry(model, 3)               # no physics: not checked
    mk.block(other, (1, 1, 1))
    (model/'physics').create('HeatTransfer', geom, name='heat')
    with pytest.raises(RuntimeError, match=r'run model.build\(geom\) '
                                           'first, also after loading'):
        mk.check(model)
    model.build(geom)
    assert mk.check(model)
    # labels that repeat across components give the right node or None
    second = blocks(model, 1)
    (model/'physics').create('HeatTransfer', second, name='heat2')
    (model/'physics'/'heat2').java.label('heat')
    defaults = [p['node'] for p in mk.check(model)
                if p['kind'] == 'default_condition']
    assert defaults == ['physics/heat/Thermal Insulation 1', None]
    # slashes in labels
    slash = (model/'physics'/'heat').create('HeatFluxBoundary', 2)
    slash.java.label('in/out')
    unselected = item(mk.check(model), 'empty_selection',
                      'physics/heat/in//out')
    assert 'Give it a selection with .select(...)' in unselected['message']
    # a saved and loaded model gives the same
    found = mk.check(model)
    model.save(tmp_path/'check.mph')
    loaded = client.load(tmp_path/'check.mph')
    try:
        assert mk.check(loaded) == found
    finally:
        client.remove(loaded)


########
# Docs #
########

def test_documented(client):
    readme = read(root/'README.md')
    lines = [re.search(r'Before solving, a check.*?```python\n(.*?)```',
                       readme, re.S).group(1)]
    lines.append(re.search(r'\n    (warnings = .*?)\n', mk.__doc__).group(1))
    model, geom, _ = plate_with_holes.build_model(client, 1)
    try:
        for code in lines:
            namespace = {'mk': mk, 'model': model}
            exec(code, namespace)
            assert namespace['warnings'] == []
    finally:
        client.remove(model)
