"""
Checks the guidance for guessed names and the documentation's references.

Runs without COMSOL, except the tests that execute the documented examples.
"""
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


def test_broken_hint_falls_back(monkeypatch):
    def fail(module, name):
        raise RuntimeError('bug in the hints')
    monkeypatch.setattr(_hints, '_message', fail)
    with pytest.raises(AttributeError, match="has no attribute 'box'$"):
        mk.box


def test_help_lists_no_hook():
    import pydoc
    text = pydoc.render_doc(mk, renderer=pydoc.plaintext)
    assert 'Rules:' in text
    assert '__getattr__' not in text and 'HintModule' not in text


def documented_names(text):
    """Returns the `mk.x` and `mphkit.x.y` references in a text."""
    return set(re.findall(r'(?<![\w./])(?:mk|mphkit)((?:\.[A-Za-z_]\w*)+)',
                          text))


@pytest.mark.parametrize('source', ['README.md', 'mphkit', 'mphkit.sel'])
def test_documented_names_exist(source):
    text = {'README.md': readme, 'mphkit': mk.__doc__,
            'mphkit.sel': mk.sel.__doc__}[source]
    names = documented_names(text)
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
