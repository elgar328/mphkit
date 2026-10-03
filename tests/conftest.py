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
