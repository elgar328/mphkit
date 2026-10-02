"""
Tests of result plots (`mk.plot`) and mesh pictures (`mk.image(mesh=)`).

Most tests draw the solved plate-with-holes example (mm, 2 holes). Tests
that change a model build their own. What the pictures show was checked
by eye while writing these; the tests compare sizes and bytes.
"""
import struct

import numpy
import pytest
from test_example import plate_with_holes
from test_results import plate as heat_plate

import mphkit as mk
from conftest import java_export, read


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', data[16:24])


def leftovers(model):
    """Tags of everything a picture could leave in a model."""
    java = model.java
    results = java.result()
    found = [sorted(str(t) for t in container.tags())
             for container in (results, results.export(), results.dataset(),
                               results.numerical(), java.selection())]
    for component in java.component().tags():
        found.append(sorted(str(t) for t in
                            java.component(component).view().tags()))
    return found


@pytest.fixture(autouse=True)
def leaves_nothing(monkeypatch):
    """Checks after every picture, also failed ones, that nothing is left."""
    for name in ('plot', 'image'):
        def checked(geom, *args, _helper=getattr(mk, name), **kwargs):
            before = leftovers(geom.model)
            try:
                return _helper(geom, *args, **kwargs)
            finally:
                assert leftovers(geom.model) == before
        monkeypatch.setattr(mk, name, checked)


@pytest.fixture(scope='module')
def solved(client):
    model, geom, selections = plate_with_holes.build_model(client, 2)
    model.solve()
    yield model, geom, selections
    client.remove(model)


def solid(client, dim=3, axisymmetric=False):
    """A cantilever loaded at its free end, solved: solid mechanics."""
    model = client.create('solid')
    geom = mk.geometry(model, dim)
    if axisymmetric:
        geom.java.axisymmetric(True)
    if dim == 3:
        mk.block(geom, (1, 0.1, 0.1))
    else:
        mk.rectangle(geom, (1, 0.1))
    model.build(geom)
    mechanics = (model/'physics').create('SolidMechanics', geom)
    fixed, loaded = (('y', 0), ('y', 0.1)) if axisymmetric else \
        (('x', 0), ('x', 1))
    mechanics.create('Fixed', dim - 1).select(
        mk.sel.box(geom, 'boundary', **dict([fixed])))
    load = mechanics.create('BoundaryLoad', dim - 1)
    load.select(mk.sel.box(geom, 'boundary', **dict([loaded])))
    load.property('FperArea', ['0', '0', '-1e6'] if dim == 3 or axisymmetric
                  else ['0', '-1e6', '0'])
    steel = (model/'materials').create('Common')
    for key, value in (('youngsmodulus', '200e9'), ('poissonsratio', '0.3'),
                       ('density', '7850')):
        (steel/'Basic').property(key, [value])
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    return model, geom


############
# Pictures #
############

def test_pictures(solved, tmp_path):
    model, geom, selections = solved
    walls = selections['hole walls']
    plate = mk.sel.all(geom, 'domain', name='plate domain')
    calls = {
        'surface': {},
        'walls': {'selection': walls},
        'domain': {'selection': plate},
        'slice': {'z': 2.5},
        'slices': {'x': [10, 50, 90]},
        'array': {'y': numpy.linspace(10, 30, 3)},
        'top face': {'z': 5},
        'slice in domain': {'selection': plate, 'x': 50},
        'range': {'color_range': (90, 100)},
        'colors': {'colortable': 'HeatCamera'},
        **{f'view {v}': {'view': v} for v in
           ('top', 'bottom', 'front', 'back', 'left', 'right', 'iso')},
    }
    for name, options in calls.items():
        options = dict(options)
        selection = options.pop('selection', None)
        path = mk.plot(geom, 'T', tmp_path/f'{name}.png', selection,
                       unit='degC', **options)
        assert png_size(path) == (800, 600), name
    path = mk.plot(geom, 'T', tmp_path/'new'/'folder'/'small.PNG',
                   size=(320, 240))
    assert path == tmp_path/'new'/'folder'/'small.png'
    assert png_size(path) == (320, 240)
    path = mk.plot(geom, 'T', tmp_path/'photo.jpg')
    assert path.read_bytes()[:3] == b'\xff\xd8\xff'


def test_pictures_change_with_the_options(solved, tmp_path):
    model, geom, selections = solved

    def picture(name, **options):
        return mk.plot(geom, 'T', tmp_path/f'{name}.png', unit='degC',
                       **options).read_bytes()

    plain = picture('plain')
    assert picture('again') == plain
    views = [picture(v, view=v) for v in ('top', 'bottom', 'front', 'iso')]
    assert len(set(views + [plain])) == 5
    assert picture('range', color_range=(90, 100)) != plain
    assert picture('colors', colortable='HeatCamera') != plain


def test_view_camera_and_history_unchanged(solved, tmp_path):
    model, geom, selections = solved
    camera = model.java.component('comp1').view('view1').camera()
    position = str(camera.getString('position'))
    before = java_export(model, tmp_path/'model.java')
    mk.plot(geom, 'T', tmp_path/'top.png', view='top', z=2.5)
    mk.image(geom, tmp_path/'mesh.png', mesh=True)
    with pytest.raises(ValueError):
        mk.plot(geom, 'T', tmp_path/'kg.png', unit='kg')
    assert java_export(model, tmp_path/'model.java') == before
    assert str(camera.getString('position')) == position


def test_history_switched_off_stays_off(solved, tmp_path):
    model, geom, selections = solved
    history = model.java.hist()
    history.disable()
    try:
        mk.plot(geom, 'T', tmp_path/'T.png')
        model.parameter('unrecorded', '1')
        model.java.save(str(tmp_path/'model'), 'java')
    finally:
        history.enable()
        model.java.param().remove('unrecorded')
    assert 'unrecorded' not in read(tmp_path/'model.java')


def test_kinds_of_selections(solved, tmp_path):
    model, geom, selections = solved
    difference = geom/'Difference 1'
    result = mk.sel.result(geom, difference, 'boundary', name='cut faces')
    around = mk.sel.adjacent(geom, mk.sel.all(geom, 'domain', name='all'),
                             name='around')
    for n, selection in enumerate((result, around)):
        assert mk.plot(geom, 'T', tmp_path/f'{n}.png', selection).exists()


@pytest.mark.parametrize('options, error, match', [
    ({'unit': 'kg'}, ValueError, 'COMSOL evaluated "T" in K, not \'kg\''),
    ({'z': 50}, ValueError, r'z=50 is outside the geometry \(z from 0 to 5\)'),
    ({'x': []}, ValueError, 'x must be a number or a list'),
    ({'z': True}, TypeError, 'z must be a number'),
    ({'z': float('nan')}, ValueError, 'z must be a number'),
    ({'z': '2'}, TypeError, 'z must be a number'),
    ({'z': [[1, 2]]}, TypeError, 'z must be a number'),
    ({'x': 1, 'y': 2}, ValueError, 'along one axis'),
    ({'view': 'side'}, ValueError, "view must be None or one of 'top'"),
    ({'color_range': (2, 1)}, ValueError, 'color_range must be'),
    ({'color_range': 'ab'}, ValueError, 'color_range must be'),
    ({'colortable': 'Heatcamera'}, ValueError,
     "No colortable 'Heatcamera'; did you mean 'HeatCamera'"),
    ({'colortable': 3}, TypeError, 'colortable must be a name'),
    ({'deform': 0}, ValueError, 'positive scale factor'),
    ({'deform': -1.0}, ValueError, 'positive scale factor'),
    ({'deform': 'yes'}, TypeError, 'positive scale factor'),
    ({'deform': True}, ValueError,
     r'needs a displacement field \(u, v, w\)'),
    ({'deform': True, 'z': 1}, ValueError, 'leave out x/y/z'),
    ({'step': 'all'}, ValueError, 'plot\\(\\) draws one step'),
    ({'step': [1]}, ValueError, 'plot\\(\\) draws one step'),
    ({'step': 2}, ValueError, 'has 1 step, not 2'),
    ({'size': (0, 10)}, ValueError, 'size must be'),
])
def test_plot_errors(solved, tmp_path, options, error, match):
    model, geom, selections = solved
    with pytest.raises(error, match=match):
        mk.plot(geom, 'T', tmp_path/'x.png', **options)


def test_more_plot_errors(solved, tmp_path):
    model, geom, selections = solved
    walls = selections['hole walls']
    with pytest.raises(RuntimeError, match='Undefined variable'):
        mk.plot(geom, 'nothing', tmp_path/'x.png')
    with pytest.raises(RuntimeError, match='If "ht.ntflux" exists on '
                                           'boundaries only'):
        mk.plot(geom, 'ht.ntflux', tmp_path/'x.png', z=2)
    assert mk.plot(geom, 'ht.ntflux', tmp_path/'flux.png').exists()
    with pytest.raises(ValueError, match=r'in W/\(m\*K\), not \'K\''):
        mk.plot(geom, '45[W/(m*K)]', tmp_path/'unit.png', unit='K')
    # no picture in the wrong unit is left behind
    assert not (tmp_path/'unit.png').exists()
    with pytest.raises(TypeError, match='file name comes third'):
        mk.plot(geom, 'T.png', 'T')
    with pytest.raises(TypeError, match='file name comes third'):
        mk.plot(geom, 'T', walls)
    with pytest.raises(TypeError, match='file name comes third'):
        mk.plot(geom, tmp_path/'x.png', 'T')
    with pytest.raises(ValueError, match='Slices take a domain selection'):
        mk.plot(geom, 'T', tmp_path/'x.png', walls, z=2)
    edges = mk.sel.box(geom, 'edge', x=0, name='hot edges')
    with pytest.raises(ValueError, match='boundary or domain selection; '
                                         '".*" selects edge entities'):
        mk.plot(geom, 'T', tmp_path/'x.png', edges)
    # a folder that cannot be made: its name is taken by a file
    (tmp_path/'file').write_text('', encoding='utf-8')
    with pytest.raises(OSError, match='[Cc]ould not write the picture'):
        mk.plot(geom, 'T', tmp_path/'file'/'x.png')
    nothing = mk.sel.box(geom, 'boundary', z=50, name='nothing')
    with pytest.raises(ValueError, match='is empty; nothing to draw'):
        mk.plot(geom, 'T', tmp_path/'x.png', nothing)
    with pytest.raises(ValueError, match='is empty; nothing to draw'):
        mk.image(geom, tmp_path/'x.png', nothing, mesh=True)
    with pytest.raises(OSError, match='could not write the picture'):
        mk.image(geom, tmp_path/'file'/'x.png', mesh=True)
    plane = mk.workplane(geom)
    try:
        with pytest.raises(TypeError, match='work plane'):
            mk.plot(plane, 'T', tmp_path/'x.png')
    finally:
        plane.remove()      # the shared geometry stays as it was built


##############
# Own models #
##############

def test_slice_between_bodies(model, tmp_path):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (3, 0, 0))
    model.build(geom)
    heat = (model/'physics').create('HeatTransfer', geom)
    material = (model/'materials').create('Common')
    for key in ('thermalconductivity', 'density', 'heatcapacity'):
        (material/'Basic').property(key, ['1'])
    ends = mk.sel.union(geom, 'boundary', [
        mk.sel.box(geom, 'boundary', x=0), mk.sel.box(geom, 'boundary', x=4)])
    boundary = heat.create('TemperatureBoundary', 2)
    boundary.select(ends)
    boundary.property('T0', '300[K]')
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    with pytest.raises(ValueError, match=r'x=2 passes through no domain\.'):
        mk.plot(geom, 'T', tmp_path/'x.png', x=2)
    second = mk.sel.box(geom, 'domain', x=(3, 4), name='second')
    with pytest.raises(ValueError, match='x=0.5 passes through no domain of '
                                         'the selection'):
        mk.plot(geom, 'T', tmp_path/'x.png', second, x=0.5)
    assert mk.plot(geom, 'T', tmp_path/'ok.png', second, x=3.5).exists()
    (geom/'Block 2').property('size', ['2', '1', '1'])
    with pytest.raises(RuntimeError, match='changed or its mesh was cleared'):
        mk.plot(geom, 'T', tmp_path/'x.png')


def test_time_steps(client, tmp_path):
    model, geom, faces = heat_plate(client, 'transient', study=False)
    try:
        study = (model/'studies').create(name='transient')
        study.create('Transient').property('tlist', 'range(0,1,4)')
        model.solve()
        with pytest.raises(ValueError, match=r"has 5 steps; pass step='last' "
                                             r'or a number from 1 to 5'):
            mk.plot(geom, 'T', tmp_path/'x.png')
        first = mk.plot(geom, 'T', tmp_path/'1.png', step=1).read_bytes()
        last = mk.plot(geom, 'T', tmp_path/'5.png', step='last').read_bytes()
        assert first != last
        assert mk.plot(geom, 'T', tmp_path/'5b.png', step=5).read_bytes() \
            == last
        assert mk.plot(geom, 'T', tmp_path/'1b.png', step=1,
                       dataset='dset1').read_bytes() == first
    finally:
        client.remove(model)


def test_second_component(client, tmp_path):
    model, geom, selections = plate_with_holes.build_model(client, 1)
    try:
        other = mk.geometry(model, 3)
        mk.block(other, (10, 10, 10))
        model.build(other)
        heat = (model/'physics').create('HeatTransfer', other)
        for x, temperature in ((0, '50[degC]'), (10, '0[degC]')):
            boundary = heat.create('TemperatureBoundary', 2)
            boundary.select(mk.sel.box(other, 'boundary', x=x))
            boundary.property('T0', temperature)
        material = model.java.component('comp2').material().create(
            'mat9', 'Common')
        for key in ('thermalconductivity', 'density', 'heatcapacity'):
            material.propertyGroup('def').set(key, '1')
        (model/'meshes').create(other)
        model.solve()
        own = mk.plot(other, 'T2', tmp_path/'other.png', unit='degC')
        top = mk.plot(other, 'T2', tmp_path/'top.png', view='top')
        plate = mk.plot(geom, 'T', tmp_path/'plate.png', unit='degC')
        assert len({own.read_bytes(), top.read_bytes(),
                    plate.read_bytes()}) == 3
        assert mk.image(other, tmp_path/'mesh.png', mesh=True).exists()
        with pytest.raises(ValueError, match='does not belong to geometry'):
            mk.image(other, tmp_path/'x.png', mesh='mesh')
    finally:
        client.remove(model)


def test_2d_complex_and_frequencies(model, tmp_path):
    geom = mk.geometry(model, 2)
    mk.rectangle(geom, (1, 0.2))
    model.build(geom)
    acoustics = (model/'physics').create('PressureAcoustics', geom)
    source = acoustics.create('Pressure', 1)
    source.select(mk.sel.box(geom, 'boundary', x=0))
    source.property('p0', '1[Pa]')
    acoustics.create('PlaneWaveRadiation', 1).select(
        mk.sel.box(geom, 'boundary', x=1))
    air = (model/'materials').create('Common')
    (air/'Basic').property('density', ['1.2'])
    (air/'Basic').property('soundspeed', ['343'])
    (model/'meshes').create(geom)
    (model/'studies').create().create('Frequency').property('plist',
                                                            '500 1000')
    model.solve()
    low = mk.plot(geom, 'p', tmp_path/'500.png', unit='Pa', step=1)
    high = mk.plot(geom, 'p', tmp_path/'1000.png', unit='Pa', step=2)
    assert low.read_bytes() != high.read_bytes()
    assert mk.plot(geom, 'abs(p)', tmp_path/'abs.png', step=2).exists()
    left = mk.sel.box(geom, 'domain', x=(0, 1), name='air')
    assert mk.plot(geom, 'p', tmp_path/'sel.png', left, step=1).exists()
    for options in ({'x': 0.5}, {'view': 'top'}):
        with pytest.raises(ValueError, match='Slices and views are for 3D'):
            mk.plot(geom, 'p', tmp_path/'x.png', step=1, **options)
    with pytest.raises(ValueError, match=r'needs a displacement field '
                                         r'\(u, v\)'):
        mk.plot(geom, 'p', tmp_path/'x.png', step=1, deform=True)
    with pytest.raises(ValueError, match='A plot in 2D takes a domain '
                                         'selection'):
        mk.plot(geom, 'p', tmp_path/'x.png', mk.sel.box(
            geom, 'boundary', x=0, name='inlet'), step=1)


def test_deform(client, tmp_path):
    model, geom = solid(client)
    try:
        auto = mk.plot(geom, 'solid.mises', tmp_path/'auto.png', unit='MPa',
                       deform=True)
        true = mk.plot(geom, 'solid.mises', tmp_path/'true.png', unit='MPa',
                       deform=1)
        plain = mk.plot(geom, 'solid.mises', tmp_path/'plain.png',
                        unit='MPa')
        # the true shape differs from the plain one by a pixel or two only
        assert auto.read_bytes() not in (true.read_bytes(),
                                         plain.read_bytes())
        top = mk.sel.box(geom, 'boundary', z=0.1, name='top')
        assert mk.plot(geom, 'solid.disp', tmp_path/'top.png', top,
                       deform=True).exists()
    finally:
        client.remove(model)
    for axisymmetric in (False, True):
        model, geom = solid(client, 2, axisymmetric)
        try:
            assert mk.plot(geom, 'solid.disp', tmp_path/'2d.png',
                           deform=True).exists()
        finally:
            client.remove(model)


def test_deform_without_displacement_axisymmetric(model, tmp_path):
    geom = mk.geometry(model, 2)
    geom.java.axisymmetric(True)
    mk.rectangle(geom, (2, 1))
    model.build(geom)
    material = (model/'materials').create('Common')
    for key in ('thermalconductivity', 'density', 'heatcapacity'):
        (material/'Basic').property(key, ['1'])
    heat = (model/'physics').create('HeatTransfer', geom)
    boundary = heat.create('TemperatureBoundary', 1)
    boundary.select(mk.sel.box(geom, 'boundary', y=0))
    boundary.property('T0', '300[K]')
    (model/'meshes').create(geom)
    (model/'studies').create().create('Stationary')
    model.solve()
    with pytest.raises(ValueError, match=r'displacement field \(u, w\)'):
        mk.plot(geom, 'T', tmp_path/'x.png', deform=True)


#################
# Mesh pictures #
#################

def test_mesh_pictures(model, tmp_path):
    geom = mk.geometry(model, 3)
    with pytest.raises(RuntimeError, match=r"has no mesh; create one with "
                                           r"\(model/'meshes'\)"):
        mk.image(geom, tmp_path/'x.png', mesh=True)
    mk.block(geom, (1, 1, 1))
    mk.block(geom, (1, 1, 1), (1, 0, 0))
    model.build(geom)
    mesh = (model/'meshes').create(geom, name='fine')
    with pytest.raises(RuntimeError, match=r'has no mesh yet; run '
                                           r'model.mesh\(\) first'):
        mk.image(geom, tmp_path/'x.png', mesh=True)
    model.mesh()
    whole = mk.image(geom, tmp_path/'mesh.png', mesh=True)
    assert png_size(whole) == (800, 600)
    faces = mk.sel.box(geom, 'boundary', z=1, name='top')
    first = mk.sel.box(geom, 'domain', x=(0, 1), name='first')
    edges = mk.sel.box(geom, 'edge', z=1, name='top edges')
    pictures = {whole.read_bytes()}
    for name, selection in (('faces', faces), ('first', first)):
        for zoom in ('all', 'selection'):
            path = mk.image(geom, tmp_path/f'{name}{zoom}.png', selection,
                            mesh=True, zoom=zoom)
            pictures.add(path.read_bytes())
    assert len(pictures) == 3          # zoom has no effect
    with pytest.raises(ValueError, match='selects edge entities'):
        mk.image(geom, tmp_path/'x.png', edges, mesh=True)
    with pytest.raises(ValueError, match='leave it out with mesh=True'):
        mk.image(geom, tmp_path/'x.png', faces, mesh=True, labels=True)
    with pytest.raises(ValueError, match="zoom='selection' needs"):
        mk.image(geom, tmp_path/'x.png', mesh=True, zoom='selection')
    with pytest.raises(TypeError, match='mesh must be True, False'):
        mk.image(geom, tmp_path/'x.png', mesh=1)
    with pytest.raises(TypeError, match='mesh must be True, False'):
        mk.image(geom, tmp_path/'x.png', mesh=None, labels=True)
    with pytest.raises(LookupError, match='No mesh "coarse"'):
        mk.image(geom, tmp_path/'x.png', mesh='coarse')
    # a second mesh: pick one by name, tag or node
    coarse = (model/'meshes').create(geom, name='coarse')
    coarse.java.autoMeshSize(9)
    coarse.run()
    with pytest.raises(ValueError, match=r'several meshes: "fine" \(mesh1\), '
                                         r'"coarse" \(mesh2\)'):
        mk.image(geom, tmp_path/'x.png', mesh=True)
    chosen = {mk.image(geom, tmp_path/f'{n}.png', mesh=m).read_bytes()
              for n, m in enumerate(('coarse', 'mesh2', coarse))}
    assert len(chosen) == 1 and chosen != {whole.read_bytes()}
    # a disabled mesh feature stays unbuilt for good; that is not stale
    # (adding a feature replaces COMSOL's automatic ones: add the meshing)
    size = mesh.java.create('local', 'Size')
    size.selection().geom(geom.tag(), 3)
    size.selection().set([1])
    mesh.java.create('tets', 'FreeTet')
    mesh.run()
    assert not mesh.java.isEmpty()
    size.active(False)
    mesh.run()
    assert mk.image(geom, tmp_path/'disabled.png', mesh='fine').exists()
    # changed settings: the old mesh would be drawn
    size.active(True)
    with pytest.raises(RuntimeError, match='changed since it was built'):
        mk.image(geom, tmp_path/'x.png', mesh='fine')
    mesh.run()
    # a changed geometry: not built, then its meshes are empty
    (geom/'Block 1').property('size', ['2', '1', '1'])
    with pytest.raises(RuntimeError, match='not built'):
        mk.image(geom, tmp_path/'x.png', mesh='fine')
    model.build(geom)
    with pytest.raises(RuntimeError, match='has no mesh yet'):
        mk.image(geom, tmp_path/'x.png', mesh=True)
    with pytest.raises(RuntimeError, match='Mesh "fine" is empty'):
        mk.image(geom, tmp_path/'x.png', mesh='fine')


def test_mesh_pictures_2d_and_1d(model, tmp_path):
    flat = mk.geometry(model, 2)
    mk.rectangle(flat, (2, 1))
    mk.circle(flat, 0.3, (1, 0.5))
    model.build(flat)
    (model/'meshes').create(flat)
    model.mesh()
    disk = mk.sel.disk(flat, 'domain', (1, 0.5), 0.3, name='disk')
    assert mk.image(flat, tmp_path/'flat.png', disk, mesh=True).exists()
    rim = mk.sel.disk(flat, 'boundary', (1, 0.5), 0.3, rin=0.29, name='rim')
    with pytest.raises(ValueError, match='a domain selection in 2D'):
        mk.image(flat, tmp_path/'x.png', rim, mesh=True)
    line = mk.geometry(model, 1)
    mk.interval(line, [0, 1])
    model.build(line)
    with pytest.raises(ValueError, match='Mesh pictures are for 2D and 3D'):
        mk.image(line, tmp_path/'x.png', mesh=True)


##########
# Sweeps #
##########

def sweep(study, values='100 200 300', name='Th', unit='degC'):
    """Adds a parametric sweep to a study node."""
    node = study.create('Parametric')
    node.property('pname', [name])
    node.property('plistarr', [values])
    node.property('punit', [unit])
    return node


@pytest.fixture(scope='module')
def swept(client):
    """The heat plate swept over Th = 100, 200, 300 degC, time-dependent."""
    model, geom, faces = heat_plate(client, 'swept plate', study=False)
    study = (model/'studies').create(name='sweep')
    study.create('Transient').property('tlist', 'range(0,1,2)')
    sweep(study)
    model.solve()
    yield model, geom
    client.remove(model)


def test_sweep_pictures(swept, tmp_path):
    model, geom = swept
    one = mk.plot(geom, 'T', tmp_path/'two.png', outer=2, step='last')
    assert one == tmp_path/'two.png'
    every = mk.plot(geom, 'T', str(tmp_path/'T_{outer}.png'), outer='all',
                    step='last', unit='degC')
    assert every == [tmp_path/f'T_{n}.png' for n in (1, 2, 3)]
    pictures = [path.read_bytes() for path in every]
    assert len(set(pictures)) == 3
    again = mk.plot(geom, 'T', tmp_path/'again_{outer}.png', outer=[3, 1],
                    step='last', unit='degC')
    assert again == [tmp_path/'again_3.png', tmp_path/'again_1.png']
    assert [path.read_bytes() for path in again] == [pictures[2], pictures[0]]
    # one value, with or without {outer}
    assert mk.plot(geom, 'T', tmp_path/'v{outer}.png', outer=2, step='last',
                   unit='degC') == tmp_path/'v2.png'
    assert mk.plot(geom, 'T', tmp_path/'list_{outer}.png', outer=[2],
                   step='last', unit='degC') == [tmp_path/'list_2.png']
    assert mk.plot(geom, 'T', tmp_path/'first.png', outer=1,
                   step=1).exists()
    # no temporary file is left
    assert not [path for path in tmp_path.iterdir()
                if path.name.startswith('.')]


def test_sweep_picture_errors(swept, tmp_path):
    model, geom = swept
    with pytest.raises(ValueError, match=r'sweep over 3 values \(1: '
                                         r'Th=373.15 \(100 degC\);'):
        mk.plot(geom, 'T', tmp_path/'x.png', step='last')
    with pytest.raises(ValueError, match=r"put \{outer\} in the file name, "
                       r"e.g. '.*x_\{outer\}.png' \(a plain string"):
        mk.plot(geom, 'T', tmp_path/'x.png', outer='all', step='last')
    with pytest.raises(ValueError, match='goes in the file name, not in the '
                                         'folder'):
        mk.plot(geom, 'T', tmp_path/'{outer}'/'x.png', outer=1, step='last')
    with pytest.raises(ValueError, match='value 2 more than once'):
        mk.plot(geom, 'T', tmp_path/'x{outer}.png', outer=[2, 1, 2],
                step='last')
    with pytest.raises(ValueError, match="plot\\(\\) draws one step"):
        mk.plot(geom, 'T', tmp_path/'x.png', outer=1, step='all')
    with pytest.raises(ValueError, match=r'at outer=2 \(Th=473.15 \(200 '
                                         r'degC\)\) has 3 steps'):
        mk.plot(geom, 'T', tmp_path/'x.png', outer=2)
    # an error leaves existing files as they were
    keep = tmp_path/'keep_2.png'
    keep.write_bytes(b'old')
    with pytest.raises(ValueError, match=r"not 'kg'"):
        mk.plot(geom, 'T', tmp_path/'keep_{outer}.png', outer=[1, 2],
                step='last', unit='kg')
    assert keep.read_bytes() == b'old'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['keep_2.png']


def test_outer_without_sweep(client, tmp_path):
    model, geom, faces = heat_plate(client, 'plain')
    try:
        with pytest.raises(ValueError, match=r'has \{outer\} for the outer '
                           r'value, but dataset .* has no outer sweep'):
            mk.plot(geom, 'T', tmp_path/'T_{outer}.png')
        with pytest.raises(ValueError, match='has no outer sweep; leave out '
                                             'outer='):
            mk.plot(geom, 'T', tmp_path/'T.png', outer=1)
    finally:
        client.remove(model)


def test_mesh_and_geometry_sweeps(client, tmp_path):
    model, geom, faces = heat_plate(client, 'mesh sweep', study=False)
    try:
        model.parameter('hm', '0.02')
        size = (model/'meshes'/'mesh').create('Size')
        size.property('custom', 'on')
        size.property('hmaxactive', True)
        size.property('hmax', 'hm')
        (model/'meshes'/'mesh').create('FreeTet')
        study = (model/'studies').create(name='mesh')
        study.create('Stationary')
        sweep(study, '0.02 0.01', 'hm', 'm')
        model.solve()
        coarse, fine = mk.plot(geom, 'T', tmp_path/'hm_{outer}.png',
                               outer='all')
        assert coarse.read_bytes() != fine.read_bytes()
        # a sweep that changes the geometry
        model.parameter('W', '0.1')
        (geom/'Block 1').property('size', ['W', '0.05', '0.01'])
        model.build(geom)
        width = (model/'studies').create(name='width')
        width.create('Stationary')
        sweep(width, '0.1 0.12', 'W', 'm')
        model.solve('width')
        with pytest.raises(ValueError, match='changes the geometry, and '
                                             'mk.plot draws on the geometry '
                                             'as built'):
            mk.plot(geom, 'T', tmp_path/'W.png', dataset='width', outer=1)
    finally:
        client.remove(model)


def conductivity(model):
    """Makes the heat plate's conductivity the parameter k."""
    model.parameter('k', '45[W/(m*K)]')
    (model/'materials'/'steel'/'Basic').property('thermalconductivity',
                                                ['k'])


def two_parameters(model, geom, study):
    conductivity(model)
    study.create('Transient').property('tlist', '0 1')
    both = study.create('Parametric')
    both.property('pname', ['Th', 'k'])
    both.property('plistarr', ['100 200', '10 90'])
    both.property('punit', ['degC', 'W/(m*K)'])
    both.java.set('sweeptype', 'filled')
    return 4, 'last'


def auxiliary(model, geom, study):
    conductivity(model)
    stationary = study.create('Stationary')
    stationary.property('useparam', True)
    stationary.property('pname', ['k'])
    stationary.property('plistarr', ['10 90'])
    stationary.property('punit', ['W/(m*K)'])
    sweep(study, '100 200')
    return 2, 1


def nested(model, geom, study):
    conductivity(model)
    study.create('Stationary')
    sweep(study, '100 200')
    sweep(study, '10 90', 'k', 'W/(m*K)')
    return 2, 2


def materials(model, geom, study):
    container = mk.component_of(geom).java.material()
    for tag in list(container.tags()):
        container.remove(tag)
    switch = container.create('sw1', 'Switch')
    switch.selection().all()
    for tag, value in (('ma', '10'), ('mb', '90')):
        group = switch.feature().create(tag, 'Common').propertyGroup('def')
        for key, number in (('thermalconductivity', value),
                            ('density', '7850'), ('heatcapacity', '475')):
            group.set(key, number)
    study.create('Transient').property('tlist', '0 1')
    swept = study.java.create('matsw', 'MaterialSweep')
    swept.set('pname', ['matsw.comp1.sw1'])
    swept.set('plistarr', ['1 2'])
    return 2, 'last'


@pytest.mark.parametrize('kind', [two_parameters, auxiliary, nested,
                                  materials])
def test_sweep_kinds(client, tmp_path, kind):
    model, geom, faces = heat_plate(client, kind.__name__, study=False)
    try:
        study = (model/'studies').create(name='sweep')
        count, step = kind(model, geom, study)
        model.solve()
        paths = mk.plot(geom, 'T', tmp_path/'T_{outer}.png', outer='all',
                        step=step)
        assert len(paths) == count
        assert len({path.read_bytes() for path in paths}) == count
    finally:
        client.remove(model)


def test_eigenvalue_sweep(client, tmp_path):
    model = client.create('eigen')
    try:
        model.parameter('a', '1')
        geom = mk.geometry(model, 2)
        mk.feature(geom, 'Rectangle', size=[1, 0.5])
        model.build(geom)
        pde = (model/'physics').create('CoefficientFormPDE', geom)
        pde.java.create('dir1', 'DirichletBoundary', 1).selection().all()
        # a coefficient, not the geometry: the same mesh for every value
        pde.java.feature('cfeq1').set('da', 'a')
        (model/'meshes').create(geom)
        study = (model/'studies').create(name='eigen')
        steps = study.create('Eigenvalue')
        steps.property('neigs', 3)
        steps.property('shift', '0')
        sweep(study, '1 2', 'a', '')
        model.solve()
        first = mk.plot(geom, 'u', tmp_path/'first.png', outer=2, step=1)
        third = mk.plot(geom, 'u', tmp_path/'third.png', outer=2, step=3)
        assert first.read_bytes() != third.read_bytes()
        assert len(mk.plot(geom, 'u', tmp_path/'u_{outer}.png', outer='all',
                           step=2)) == 2
    finally:
        client.remove(model)
