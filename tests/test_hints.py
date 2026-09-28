"""
Checks the guidance for guessed names and the documentation's references.

Runs without COMSOL, except the tests that execute the documented examples.
"""
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mphkit as mk
from mphkit import _hints

root = Path(__file__).parents[1]
readme = (root/'README.md').read_text()


def message(module, name):
    with pytest.raises(AttributeError) as error:
        getattr(module, name)
    return str(error.value)


@pytest.mark.parametrize('module, name, expected', [
    (mk, 'box', 'Did you mean mphkit.sel.box (select) or mphkit.block (create)?'),
    (mk, 'Block', 'Did you mean mphkit.block (create)'),
    (mk, 'cube', 'Did you mean mphkit.block (create) or mphkit.sel.box'),
    (mk, 'select_box', 'Did you mean mphkit.sel.box (select)'),
    (mk, 'get_entities', 'Did you mean mphkit.sel.entities?'),
    (mk, 'measure_volume', 'Did you mean mphkit.measure?'),
    (mk, 'all', 'Did you mean mphkit.sel.all?'),
    (mk, 'blok', 'Did you mean mphkit.block?'),
    (mk.sel, 'block', 'Did you mean mphkit.sel.box (select) or mphkit.block (create)?'),
    (mk.sel, 'sphere', 'Did you mean mphkit.sel.ball (select)'),
    (mk.sel, 'set', 'Did you mean mphkit.set?'),
    (mk.sel, 'faces', "mphkit.sel.box(geom, 'boundary', ...)"),
    (mk.sel, 'point', "mphkit.sel.all(geom, 'point')"),
    # a prefix says what is meant
    (mk, 'select_cylinder', 'Did you mean mphkit.sel.cylinder?'),
    (mk.sel, 'create_cylinder', 'Did you mean mphkit.cylinder?'),
    (mk, 'select_sphere', 'Did you mean mphkit.sel.ball (select) or'),
    (mk, 'create_box', 'Did you mean mphkit.block (create) or'),
    (mk.sel, 'create_box', 'Did you mean mphkit.block (create) or'),
    (mk, 'select_line', "mphkit.sel.box(geom, 'edge', ...)"),
    (mk, 'select_volume', "mphkit.sel.box(geom, 'domain', ...)"),
    (mk, 'find_edges', "Did you mean mphkit.sel.find(geom, 'edge', ...)?"),
    (mk.sel, 'find_faces', "Did you mean mphkit.sel.find(geom, 'boundary', ...)?"),
    (mk, 'get_boundaries', 'or mphkit.sel.entities(geom, selection)?'),
    (mk, 'get_volume', 'Did you mean mphkit.measure?'),
    # entity words ask for a selection, except where they name a helper
    (mk, 'faces', "mphkit.sel.box(geom, 'boundary', ...)"),
    (mk, 'points', 'mphkit.point creates a point.'),
    (mk.sel, 'points', 'mphkit.point creates a point.'),
    (mk.sel, 'line', "mphkit.sel.box(geom, 'edge', ...)"),
    (mk, 'line', 'Did you mean mphkit.line_segment?'),
    (mk, 'volume', 'Did you mean mphkit.measure?'),
    (mk.sel, 'all_boundaries', "Did you mean mphkit.sel.all(geom, 'boundary', ...)?"),
    # plurals, meanings and features without a helper
    (mk, 'boxes', 'Did you mean mphkit.sel.box (select)'),
    (mk, 'polyline', 'Did you mean mphkit.polygon?'),
    (mk, 'subtract', 'Did you mean mphkit.difference?'),
    (mk, 'merge', 'Did you mean mphkit.union?'),
    (mk, 'fuse', 'Pass intbnd=False to merge touching'),
    (mk, 'unite', 'Did you mean mphkit.union?'),
    (mk, 'create_union', 'Pass intbnd=False'),
    (mk.sel, 'unite', 'Did you mean mphkit.sel.union?'),
    (mk.sel, 'merge', 'Did you mean mphkit.sel.union?'),
    (mk.sel, 'subtract', 'Did you mean mphkit.sel.difference?'),
    (mk, 'translate', 'Did you mean mphkit.move?'),
    (mk, 'cone', "Did you mean mphkit.feature(geom, 'Cone', ...)?"),
    (mk, 'bounding', 'Did you mean mphkit.bounding_box?'),
    (mk, 'fillet_edges', 'Did you mean mphkit.fillet?'),
])
def test_suggestion(module, name, expected):
    text = message(module, name)
    assert text.startswith(f"module '{module.__name__}' has no attribute {name!r}.")
    assert expected in text
    assert text.endswith(f'See help({module.__name__}) for all helpers.')


def test_no_suggestion_still_points_to_help():
    text = message(mk, 'xyzzy')
    assert 'Did you mean' not in text
    assert 'help(mphkit)' in text
    # no far-fetched matches, and no module suggesting itself
    assert 'Did you mean' not in message(mk, 'rect')
    assert 'mphkit.sel?' not in message(mk.sel, 'selection')
    assert message(mk, 'bounding').count('mphkit.') == 1


def test_python_adds_no_second_suggestion():
    # Python only adds its own suggestion when it prints an uncaught error
    for name in ('blok', 'box'):
        run = subprocess.run([sys.executable, '-c', f'import mphkit; mphkit.{name}'],
                             capture_output=True, text=True)
        last = run.stderr.strip().splitlines()[-1]
        assert 'help(mphkit)' in last
        assert 'Did you mean:' not in last


def test_attribute_protocol_unchanged():
    assert not hasattr(mk, 'box')
    assert not hasattr(mk.sel, 'block')
    with pytest.raises(AttributeError) as error:
        mk.__wrapped__
    assert 'help' not in str(error.value)
    namespace = {}
    exec('from mphkit import *', namespace)
    assert 'block' in namespace and 'set' not in namespace
    assert type(mk).__name__ == type(mk.sel).__name__ == 'HintModule'


def test_union_note_only_for_geometry():
    # in mphkit.sel, "merge" means the selection union: no intbnd note
    assert 'intbnd' not in message(mk.sel, 'merge')
    assert 'intbnd' not in message(mk, 'unoin')     # two suggestions


def test_broken_hint_falls_back(monkeypatch):
    def fail(module, name):
        raise RuntimeError('bug in the hints')
    monkeypatch.setattr(_hints, '_message', fail)
    with pytest.raises(AttributeError, match="has no attribute 'box'$"):
        mk.box


def test_help_physics_levels(model):
    # the levels named in the Rules of help(mphkit)
    def level(feature):
        return [int(d) for d in feature.java.selection().dimension()]

    flat = mk.geometry(model, 2)
    mk.square(flat, 1)
    model.build(flat)
    heat = (model/'physics').create('HeatTransfer', flat)
    assert level(heat.create('TemperatureBoundary', 1)) == [1]
    assert level(heat.create('HeatSource', 2)) == [2]
    with pytest.raises(Exception, match='specified element dimension'):
        heat.create('TemperatureBoundary', 2)
    solid = mk.geometry(model, 3)
    mk.block(solid, (1, 1, 1))
    model.build(solid)
    heat = (model/'physics').create('HeatTransfer', solid)
    assert level(heat.create('TemperatureBoundary', 2)) == [2]
    assert level(heat.create('HeatSource', 3)) == [3]


def test_help_lists_no_hook():
    import pydoc
    text = pydoc.render_doc(mk, renderer=pydoc.plaintext)
    assert 'Rules:' in text
    assert '__getattr__' not in text and 'HintModule' not in text


def documented_names(text):
    """Returns the `mk.x`, `mphkit.x.y` and `sel.x` references in a text."""
    names = set(re.findall(r'(?<![\w./])(?:mk|mphkit)((?:\.[A-Za-z_]\w*)+)',
                           text))
    names |= {f'.sel.{name}' for name in
              re.findall(r'(?<![\w./])sel\.([A-Za-z_]\w*)', text)}
    return names


# README, the module docs and the docs of every public helper
docs = {'README.md': readme, 'mphkit': mk.__doc__,
        'mphkit.sel': mk.sel.__doc__}
docs.update({f'mphkit.{name}': inspect.getdoc(getattr(mk, name)) or ''
             for name in mk.__all__ + ['set'] if name != 'sel'})
docs.update({f'mphkit.sel.{name}': inspect.getdoc(getattr(mk.sel, name)) or ''
             for name in mk.sel.__all__})


@pytest.mark.parametrize('source', docs)
def test_documented_names_exist(source):
    names = documented_names(docs[source])
    if source in ('README.md', 'mphkit', 'mphkit.sel'):
        assert names
    for name in names:
        target = mk
        for part in name.strip('.').split('.'):
            target = getattr(target, part)


def run_example(code, client, monkeypatch, tmp_path):
    """Runs documented example code with the test session's client."""
    import mph
    monkeypatch.setattr(mph, 'start', lambda *args, **kwargs: client)
    code = code.replace("'demo.mph'", repr(str(tmp_path/'demo.mph')))
    before = client.models()
    try:
        exec(code, {})
    finally:
        for model in client.models():
            if model not in before:
                client.remove(model)


def test_readme_example(client, monkeypatch, tmp_path):
    code = re.search(r'## Example\n\n```python\n(.*?)```', readme, re.S).group(1)
    run_example(code, client, monkeypatch, tmp_path)
    assert (tmp_path/'demo.mph').exists()


def test_help_example(client, monkeypatch, tmp_path):
    lines = [line[4:] for line in mk.__doc__.splitlines()
             if line.startswith('    ') or not line.strip()]
    code = re.search(r'import mph\n.*?\.select\(bottom\)[^\n]*\n',
                     '\n'.join(lines), re.S).group(0)
    run_example(code, client, monkeypatch, tmp_path)
