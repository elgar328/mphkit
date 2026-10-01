"""Shared fixtures. Tests needing COMSOL are skipped when it is unavailable."""
from pathlib import Path

import pytest


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
