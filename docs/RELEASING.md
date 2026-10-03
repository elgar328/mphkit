# Releasing

How changes are recorded and how a version is released. The changelog
guidelines apply to every change; the release checklist is for the
maintainers who publish releases.

## Branch and versions

Development happens on `main`, which always passes the full test suite.
Work that might be thrown away goes on a branch.

Versions follow [Semantic Versioning](https://semver.org/):
`MAJOR.MINOR.PATCH`, tagged `vMAJOR.MINOR.PATCH`. Until 1.0:

- **Minor** (0.1.0 → 0.2.0): new features, and any change that can break
  existing scripts (renamed or removed functions, changed arguments or
  defaults, different results).
- **Patch** (0.1.0 → 0.1.1): bug fixes only; existing scripts keep working.

Between releases, `main` carries the next minor version with a `.dev0`
suffix, e.g. `0.2.0.dev0` after 0.1.0, so an install from `main` never
reports the version of a release. Only the release commit has a plain
version. Versions are written in this form (`0.2.0.dev0`, not
`0.2.0-dev`): it is how Python normalizes them. The version is set in two
places, `pyproject.toml` and `src/mphkit/__init__.py`.

## CHANGELOG guidelines

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Every change a user could notice gets an entry under `## [Unreleased]`,
in the same commit as the change.

Categories, in this order; leave out empty ones:

- **Added**: new functions, arguments or behavior.
- **Changed**: existing behavior that works differently now.
- **Deprecated**: still works, will be removed.
- **Removed**: no longer available.
- **Fixed**: bugs.
- **Security**: vulnerability fixes.

Rules:

- Write from the user's point of view: what changed and what to do about
  it, not how it is implemented.
- Each entry is a concise, complete sentence.
- Start an entry with **Breaking:** when existing scripts need to change,
  and say how.
- Tests, refactoring and documentation-only changes are not listed.
- Most recent release first; always keep `## [Unreleased]` at the top.

For example (made-up entries):

```markdown
### Changed

- **Breaking:** `mk.array` takes `count=` instead of `size=`; rename the
  argument in existing scripts.

### Fixed

- `mk.revolve` accepts a numpy array as `pos`.
```

## Release checklist

`X.Y.Z` is the new version, `vA.B.C` the previous release tag. You need
`uv`, the GitHub CLI logged in with push rights (`gh auth status`), and
COMSOL for the tests. Steps 1 to 9 stay on your machine and can be
undone (see step 9); steps 10 to 14 publish and cannot.

1. **Pick the version**: the one on `main` without `.dev0`, or the next
   patch version if `[Unreleased]` lists only fixes.

2. **Start from an up-to-date `main`**:
   ```sh
   git switch main && git pull --ff-only
   git status --short                  # prints nothing
   git log --oneline origin/main..     # prints nothing
   ```
   If the last command lists commits, push them first (or leave them
   for later on a branch): the release starts from what is on GitHub.

3. **Run the checks** with COMSOL:
   ```sh
   uv run pytest                                   # includes mypy
   uv run python examples/plate_with_holes.py
   uvx pyright@1.1.414 --pythonpath .venv/bin/python src/mphkit \
     tests/typed_results.py
   ```
   pyright needs `--pythonpath` to find MPh (`.venv/Scripts/python.exe`
   on Windows); it should report no errors.
   If `.github/workflows/publish.yml` changed since the last release
   (`git diff vA.B.C -- .github/workflows/publish.yml`), also do a dry
   run of it on `main` (see [Publishing to PyPI](#publishing-to-pypi)):
   ```sh
   gh workflow run publish.yml --ref main   # prints the run's URL
   gh run watch <run id from the URL> --exit-status
   ```

4. **Finalize the version**: set `X.Y.Z` in `pyproject.toml` and
   `src/mphkit/__init__.py`, then run `uv lock`.

5. **Update CHANGELOG.md**
   - Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (today) and
     add a new, empty `## [Unreleased]` above it.
   - Update the links at the bottom, newest first:
     ```markdown
     [Unreleased]: https://github.com/elgar328/mphkit/compare/vX.Y.Z...HEAD
     [X.Y.Z]: https://github.com/elgar328/mphkit/compare/vA.B.C...vX.Y.Z
     ```

6. **Update README.md** if the tested COMSOL or MPh version changed: the
   sentence under Requirements and the COMSOL badge.

7. **Check consistency and the package**:
   ```sh
   uv run --locked pytest tests/test_release.py   # version and changelog
   uv build
   uv run python -m zipfile -l dist/mphkit-X.Y.Z-py3-none-any.whl |
     grep py.typed
   rm -rf dist
   ```
   `--locked` fails if `uv.lock` was not updated in step 4, as the
   workflow would.

8. **Commit and tag**, then check that the tag names the version:
   ```sh
   git commit -am "Release X.Y.Z"
   git tag -a vX.Y.Z -m "mphkit X.Y.Z"
   git describe --exact-match      # vX.Y.Z
   uv version --short              # X.Y.Z
   git show --stat HEAD            # version files, uv.lock, CHANGELOG.md
   ```
   The tag becomes permanent in step 10, and nothing before the real
   upload compares it with the version.

9. **Start the next development cycle**: set the next minor version with
   `.dev0` in both places (after 0.1.0: `0.2.0.dev0`), run `uv lock`, and
   commit:
   ```sh
   uv run --locked pytest tests/test_release.py
   git commit -am "Start next development cycle"
   ```
   To start over before step 10 (this drops every local commit since
   step 2):
   ```sh
   git tag -d vX.Y.Z 2>/dev/null; git reset --hard origin/main
   ```

10. **Push** both commits and the tag, all or nothing:
    ```sh
    git push --atomic origin main vX.Y.Z
    ```
    If `main` moved on GitHub, the push is refused and nothing is sent.
    From here on the tag is permanent: the repository's rules forbid
    moving or deleting `v*` tags.

11. **Create the GitHub Release as a draft** from the changelog section.
    A draft does not start the upload. GitHub shows every line break in
    release notes, so the second `awk` joins each wrapped entry into one
    line:
    ```sh
    awk -v v="X.Y.Z" '/^## \[/ { p = index($0, "## [" v "]") == 1; next }
      /^\[[^]]+\]: / { p = 0 } p' CHANGELOG.md |
      awk '/^  +[^ ]/ { sub(/^ +/, ""); line = line " " $0; next }
        NR > 1 { print line } { line = $0 } END { print line }' |
      { cat; echo "**Full Changelog**: https://github.com/elgar328/mphkit/compare/vA.B.C...vX.Y.Z"; } |
      gh release create vX.Y.Z --title "vX.Y.Z" --verify-tag --draft \
        --notes-file -
    ```
    Read the notes as GitHub shows them (the command prints the URL).

12. **Dry-run the workflow on the tag**: it checks and builds on GitHub
    with the tagged files, and skips the upload.
    ```sh
    gh workflow run publish.yml --ref vX.Y.Z   # prints the run's URL
    gh run watch <run id from the URL> --exit-status
    ```
    If it prints no URL, find the run with
    `gh run list --workflow publish.yml -L3`. If the dry run fails, do
    not publish: delete the draft (`gh release delete vX.Y.Z --yes`),
    fix the cause on `main` and release the next patch version; the tag
    stays.

13. **Publish the release**. This starts the upload to PyPI:
    ```sh
    gh release edit vX.Y.Z --draft=false
    ```

14. **Check the upload**:
    ```sh
    until id=$(gh run list --workflow publish.yml --event release \
        --branch vX.Y.Z -L1 --json databaseId -q '.[0].databaseId') &&
        [ -n "$id" ]; do sleep 5; done
    gh run watch "$id" --exit-status
    uv run --isolated --no-project --refresh-package mphkit \
      --with "mphkit==X.Y.Z" \
      python -c "import mphkit; print(mphkit.__version__)"
    ```
    The last command prints `X.Y.Z` (no COMSOL needed). PyPI may take a
    minute to serve a new version: retry if the install does not find
    it. Also look at https://pypi.org/project/mphkit/X.Y.Z/ for the
    README.

A pushed tag is not moved or reused. If a release turns out broken, fix
it in a patch release:

- If `main` has nothing but fixes since the release, release from `main`
  as above with the patch version (e.g. `0.1.1`), then go back to the dev
  version (`0.2.0.dev0`).
- Otherwise, so that the patch carries no unreleased features, branch
  from the tag (`git switch -c fix-0.1 v0.1.0`), fix there, and release
  `0.1.1` from that branch (steps 3 to 8, then `git push origin v0.1.1`
  and steps 11 to 14, with `--latest=false` in step 11 so that GitHub
  keeps the newer release as the latest). Then bring the fix and its
  changelog entry to `main`.

## Publishing to PyPI

Publishing a GitHub Release runs `.github/workflows/publish.yml`. It
checks that the tag matches the version and that the version and the
changelog agree, builds the package, and uploads it through
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): the PyPI
project trusts this workflow, running in the GitHub environment `pypi`,
so no token is stored anywhere. The environment only accepts `v*` tags.

Running the workflow by hand (`gh workflow run publish.yml`) is a dry
run: it checks and builds the package on GitHub, but skips the tag
comparison and the upload, so it does not test the environment or the
PyPI permission. `--ref vX.Y.Z` runs it with the tagged files (step 12).

A version uploaded to PyPI can never be uploaded again, even after
deleting it. So:

- If the workflow fails before the upload step, fix the cause and rerun
  it (`gh run rerun <id>`). A successful rerun uploads, so decide as in
  step 13. A rerun uses the workflow file of the tagged commit: if the
  workflow itself is broken, or the fix needs any other change to the
  code, release a new patch version instead of moving the tag.
- A broken release on PyPI is **yanked**, not deleted (in the release's
  options on PyPI): pip then skips it unless it is pinned exactly. Then
  release a fixed patch version.
