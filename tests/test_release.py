"""
Checks that the version and the changelog agree, as docs/RELEASING.md
requires.

On `main` the version is the next one with `.devN`; only a release commit
has a plain version. Runs without COMSOL.
"""
import re
from pathlib import Path

import mphkit

root = Path(__file__).parents[1]


def project_version():
    text = (root/'pyproject.toml').read_text()
    return re.search(r'^version = "(.+)"$', text, re.MULTILINE).group(1)


def released():
    """Returns the released versions in the changelog, newest first."""
    changelog = (root/'CHANGELOG.md').read_text()
    return re.findall(r'^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$',
                      changelog, re.MULTILINE)


def numbers(version):
    return tuple(int(n) for n in version.split('.'))


def test_version_matches_package():
    assert mphkit.__version__ == project_version()


def test_version_form():
    assert re.fullmatch(r'\d+\.\d+\.\d+(\.dev\d+)?', project_version())


def test_changelog():
    changelog = (root/'CHANGELOG.md').read_text()
    assert '## [Unreleased]' in changelog
    versions = released()
    assert versions == sorted(versions, key=numbers, reverse=True)
    for version in versions:
        assert f'[{version}]: ' in changelog
    base, dev, _ = project_version().partition('.dev')
    if dev:
        # the next version, above every release
        assert not versions or numbers(base) > numbers(versions[0])
    else:
        # a release commit: its section is the newest
        assert versions and versions[0] == base
