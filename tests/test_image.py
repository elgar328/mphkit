"""Tests for pictures of geometries and selections."""
import struct

import pytest

import mphkit as mk
from conftest import read


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', data[16:24])


@pytest.fixture
def plate(model, geom):
    """A 40 x 30 x 5 mm plate with a hole; its wall is a selection."""
    mk.difference(geom, mk.block(geom, (40, 30, 5)),
                  [mk.cylinder(geom, 5, 5, (20, 15, 0))])
    model.build(geom)
    return geom


@pytest.fixture
def wall(plate):
    return mk.sel.cylinder(plate, 'boundary', (20, 15, 0), 5, rin=4.95,
                           bottom=0, top=5, name='wall')


def test_image_files(plate, wall, tmp_path):
    path = mk.image(plate, tmp_path/'geometry.png')
    assert path == tmp_path/'geometry.png'
    assert png_size(path) == (800, 600)
    small = mk.image(plate, tmp_path/'wall.png', wall, size=(400, 300))
    assert png_size(small) == (400, 300)
    photo = mk.image(plate, str(tmp_path/'wall.JPEG'), wall)
    assert photo == tmp_path/'wall.jpeg'           # as COMSOL writes it
    assert photo.read_bytes()[:3] == b'\xff\xd8\xff'
    upper = mk.image(plate, tmp_path/'Upper.PNG')
    assert upper.name == 'Upper.png' and upper.exists()
    deep = mk.image(plate, tmp_path/'new'/'folder'/'g.png')
    assert deep.exists()                           # COMSOL makes folders
    photo = mk.image(plate, tmp_path/'wall.jpg', wall)
    assert photo.read_bytes()[:3] == b'\xff\xd8\xff'


def test_image_path_from_home(plate, tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))     # Windows
    assert mk.image(plate, '~/g.png') == tmp_path/'g.png'
    assert (tmp_path/'g.png').exists()


def test_image_options_change_the_picture(plate, wall, tmp_path):
    def picture(name, *args, **kwargs):
        return mk.image(plate, tmp_path/name, *args, **kwargs).read_bytes()

    geometry = picture('a.png')
    assert picture('b.png') == geometry            # deterministic
    highlighted = picture('c.png', wall)
    assert highlighted != geometry
    assert picture('d.png', wall, labels=True) != highlighted
    assert picture('e.png', wall, zoom='selection') != highlighted


def test_image_leaves_the_model_as_it_was(model, plate, wall, tmp_path):
    component = mk.component_of(plate).java
    view = component.view(component.view().tags()[0])

    def state():
        return ([str(t) for t in component.selection().tags()],
                [str(t) for t in model.java.result().tags()],
                [str(t) for t in model.java.result().export().tags()],
                str(view.getString('showlabels')))

    before = state()
    mk.image(plate, tmp_path/'g.png')
    mk.image(plate, tmp_path/'w.png', wall, labels=True)
    assert state() == before
    # nothing in the history: a Java export shows no picture
    model.java.save(str(tmp_path/'model'), 'java')
    code = read(tmp_path/'model.java')
    assert 'image()' not in code and 'showlabels' not in code
    # labels shown in the view do not leak into a picture without labels
    plain = mk.image(plate, tmp_path/'plain.png', wall).read_bytes()
    view.set('showlabels', 'on')
    assert mk.image(plate, tmp_path/'p2.png', wall).read_bytes() == plain
    assert str(view.getString('showlabels')) == 'on'


def test_image_keeps_a_history_switched_off(model, plate, tmp_path):
    history = model.java.hist()
    history.disable()
    try:
        mk.image(plate, tmp_path/'g.png')
        model.parameter('switched_off', '1')
        model.java.save(str(tmp_path/'model'), 'java')
    finally:
        history.enable()
    assert 'switched_off' not in read(tmp_path/'model.java')


def test_image_with_two_views(model, plate, wall, tmp_path):
    component = mk.component_of(plate).java
    component.view().create('extra', plate.tag())
    assert mk.image(plate, tmp_path/'w.png', wall, labels=True).exists()


def test_image_kinds_of_selections(model, plate, tmp_path):
    in_sequence = mk.sel.box(plate, 'boundary', z=5, where='geometry')
    model.build(plate)
    result = mk.sel.result(plate, plate/'Difference 1', 'boundary')
    around = mk.sel.adjacent(plate, mk.sel.all(plate, 'domain'))
    for n, selection in enumerate((in_sequence, result, around)):
        assert mk.image(plate, tmp_path/f'{n}.png', selection).exists()


def test_image_2d_1d_and_second_component(model, tmp_path):
    flat = mk.geometry(model, 2)
    mk.difference(flat, mk.rectangle(flat, (4, 3)),
                  [mk.circle(flat, 0.5, (2, 1.5))])
    model.build(flat)
    ring = mk.sel.disk(flat, 'boundary', (2, 1.5), 0.5, rin=0.49)
    assert mk.image(flat, tmp_path/'flat.png', ring, labels=True).exists()
    line = mk.geometry(model, 1)
    mk.interval(line, [0, 1, 3])
    model.build(line)
    assert mk.image(line, tmp_path/'line.png').exists()


def test_image_errors(model, plate, wall, tmp_path):
    with pytest.raises(ValueError, match=r'\.png, \.jpg or \.jpeg, not'):
        mk.image(plate, tmp_path/'g.gif')
    with pytest.raises(ValueError, match=r'\.png, \.jpg or \.jpeg, not'):
        mk.image(plate, tmp_path/'g')
    with pytest.raises(TypeError, match='file name comes second'):
        mk.image(plate, wall, tmp_path/'g.png')
    with pytest.raises(ValueError, match="zoom='selection' needs"):
        mk.image(plate, tmp_path/'g.png', zoom='selection')
    with pytest.raises(ValueError, match='zoom must be'):
        mk.image(plate, tmp_path/'g.png', wall, zoom='near')
    with pytest.raises(ValueError, match='labels=True needs a selection'):
        mk.image(plate, tmp_path/'g.png', labels=True)
    empty = mk.sel.box(plate, 'boundary', z=99)
    with pytest.raises(ValueError, match='is empty'):
        mk.image(plate, tmp_path/'g.png', empty)
    objects = mk.sel.box(plate, 'object', x=(-1, 41), where='geometry')
    model.build(plate)
    with pytest.raises(TypeError, match='selects objects'):
        mk.image(plate, tmp_path/'g.png', objects)
    mk.sel.box(plate, 'boundary', z=5, where='geometry', name='top')
    model.build(plate)
    with pytest.raises(TypeError, match="model/'selections'/'top'"):
        mk.image(plate, tmp_path/'g.png', plate/'top')
    with pytest.raises(TypeError, match='not a selection node'):
        mk.image(plate, tmp_path/'g.png', plate/'Difference 1')
    for size in (800, (1, 2, 3), (100.5, 50), (0, 10)):
        with pytest.raises(ValueError, match='size must be'):
            mk.image(plate, tmp_path/'g.png', size=size)
    # a folder that cannot be made: its name is taken by a file
    (tmp_path/'file').write_text('', encoding='utf-8')
    with pytest.raises(OSError, match='could not write the picture'):
        mk.image(plate, tmp_path/'file'/'g.png')
    with pytest.raises(TypeError, match='work plane'):
        mk.image(mk.workplane(plate), tmp_path/'g.png')
    mk.block(plate, (1, 1, 1), (50, 0, 0))
    with pytest.raises(RuntimeError, match='not built'):
        mk.image(plate, tmp_path/'g.png')
