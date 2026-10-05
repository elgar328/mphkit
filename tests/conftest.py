"""
Shared fixtures. Tests needing COMSOL are skipped when it is unavailable,
and marked `comsol`: `pytest -m "not comsol"` runs the others in seconds.
"""
import os
import sys
from pathlib import Path

import pytest

# Fixtures that need COMSOL: a running client, or the installation's files
COMSOL_FIXTURES = {'client', 'offline'}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Marks the tests that use COMSOL, before `-m` selects tests."""
    for item in items:
        if COMSOL_FIXTURES & set(getattr(item, 'fixturenames', ())):
            item.add_marker(pytest.mark.comsol)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    """
    Gives pytest's exit status to MPh, which ends the process through Java
    with the status it recorded: 0 even after failed tests.
    """
    mph_session = sys.modules.get('mph.session')
    if mph_session is None:
        return
    if not hasattr(mph_session, 'exit_code'):
        raise RuntimeError('mph.session.exit_code is gone (MPh changed): '
                           'pytest may exit with 0 after failed tests.')
    mph_session.exit_code = int(session.exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """
    On macOS, ends the process once pytest is done, before MPh's clean-up:
    disconnecting the client and ending Java there crash more often than
    not, with exit status 139 or 138 after passing tests (seen 2026-10-04
    in 10 of 19 test files). The COMSOL server ends once its client is
    gone. Not used elsewhere: Windows exits cleanly, Linux is untried.
    """
    mph_session = sys.modules.get('mph.session')
    if (sys.platform != 'darwin' or mph_session is None
            or getattr(mph_session, 'client', None) is None):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(mph_session.exit_code)


@pytest.fixture(scope='session')
def client():
    """One COMSOL client per test session (starting it is slow)."""
    mph = pytest.importorskip('mph')
    try:
        return mph.start(cores=1)
    except Exception as error:
        pytest.skip(f'COMSOL not available: {error}')


@pytest.fixture
def model(client):
    """A fresh, empty model that is removed after the test."""
    model = client.create('test')
    yield model
    client.remove(model)


@pytest.fixture
def geom(model):
    """A 3D geometry in millimeters."""
    import mphkit as mk
    return mk.geometry(model, 3, length_unit='mm')


def import_or_skip(call):
    """Runs a CAD import, skipping the test without a CAD import license."""
    import mphkit as mk
    try:
        return call()
    except mk.LicenseError as error:
        pytest.skip(f'No CAD import license: {error}')


def count(geom, what):
    """Returns the number of domains or boundaries of a built geometry."""
    java = geom.java
    return {'domains': java.getNDomains(),
            'boundaries': java.getNBoundaries()}[what]


def java_export(model, path):
    """
    Returns the lines of the model's Java export, without the one with the
    export time, which changes every minute.
    """
    model.save(path)
    return [line for line in read(path).splitlines()
            if not line.startswith('/** Model exported on')]


def read(path):
    """
    Returns a text file as UTF-8 (Windows reads the locale's code page
    otherwise); undecodable bytes, e.g. in a Java export, are replaced.
    """
    return Path(path).read_text(encoding='utf-8', errors='replace')


# Containers whose tags, properties and selections model_state() records
MODEL_LISTS = ('study', 'sol', 'batch', 'func', 'variable', 'cpl',
               'coordSystem', 'material', 'selection', 'physics',
               'multiphysics', 'mesh', 'geom', 'component', 'common')
COMPONENT_LISTS = ('physics', 'multiphysics', 'mesh', 'geom', 'pair',
                   'common')


def model_state(model):
    """
    Returns everything a helper that should leave a model as it was might
    change: the tags, properties and selections of the model's nodes (also
    ones a Java export does not show, as work done with the history off
    is not in it), mesh element counts, which solutions hold data and the
    open models. NaN reads as 'NaN', so equal states compare equal.
    """
    import math

    import jpype
    from mph.node import get
    from mphkit import _catalog
    java = model.java
    found = {}

    def plain(value):
        if isinstance(value, float) and math.isnan(value):
            return 'NaN'
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value

    def node(item):
        values = {}
        if hasattr(item, 'properties'):
            for name in [str(n) for n in item.properties()]:
                try:
                    if str(item.getValueType(name)) == 'Selection':
                        continue
                    values[name] = plain(_catalog.plain(get(item, name)))
                except Exception:
                    values[name] = '<unreadable>'
        try:
            selection = item.selection()
            values['<selection>'] = ([int(d) for d in selection.dimension()],
                                     [int(e) for e in selection.entities()])
        except Exception:
            pass
        return values

    def walk(container, path, depth=0):
        try:
            tags = [str(t) for t in container.tags()]
        except Exception:
            return
        found[path] = tags
        for tag in tags:
            try:
                item = container.get(tag)
            except Exception:
                continue
            found[f'{path}/{tag}'] = node(item)
            if depth > 5:
                continue
            for name in ('feature', 'propertyGroup'):
                if hasattr(item, name):
                    try:
                        walk(getattr(item, name)(), f'{path}/{tag}/{name}',
                             depth + 1)
                    except Exception:
                        pass
            if hasattr(item, 'prop'):
                try:
                    for group in item.prop():
                        found[f'{path}/{tag}/prop/{group.tag()}'] = \
                            node(group)
                except Exception:
                    pass

    for name in MODEL_LISTS:
        walk(getattr(java, name)(), name)
    for ctag in java.component().tags():
        component = java.component(ctag)
        for name in COMPONENT_LISTS:
            walk(getattr(component, name)(), f'{ctag}:{name}')
    results = java.result()
    walk(results, 'plot')
    for name in ('dataset', 'numerical', 'table', 'export'):
        walk(getattr(results, name)(), name)
    found['<parameters>'] = {str(n): str(java.param().get(n))
                             for n in java.param().varnames()}
    found['<elements>'] = {str(t): int(java.mesh(t).getNumElem())
                           for t in java.mesh().tags()}
    found['<solved>'] = {str(t): bool(java.sol(t).isEmpty())
                         for t in java.sol().tags()}
    util = jpype.JClass('com.comsol.model.util.ModelUtil')
    found['<models>'] = sorted(str(t) for t in util.tags())
    return found
