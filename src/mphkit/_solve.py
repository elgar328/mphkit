"""
The size of a problem before solving and the progress of a solve. Public
as `mk.problem_size`, `mk.log_progress` and `mk.progress`.

`problem_size()` compiles the equations of a study in a temporary solver
sequence, as `model.solve()` would, and reads the number of degrees of
freedom and the solver COMSOL would use; it leaves nothing in the model.
`log_progress()` makes COMSOL write its progress log (the one the COMSOL
Desktop shows) to a file, and `progress()` reads that file and the
operating system's facts about the solving processes, from another
Python process: an agent runs a long solve in the background and checks
on it. Nothing here judges whether a solve goes well; the numbers are
COMSOL's and the operating system's.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, overload

from mph.model import Model

from . import _check, _comsol

# Tag prefix of the temporary solver sequence; its blocks in a progress
# log are skipped
ESTIMATE = 'mkest'
# Linear solver nodes and what problem_size calls them
SOLVERS = {'Direct': 'direct', 'Iterative': 'iterative'}
UNKNOWN = {'solver': None, 'linear_solver': None}
# Child types that make a solver sequence feature a solver (Stationary,
# Time, Eigenvalue, ...)
SOLVER_CHILDREN = ('Direct', 'Iterative', 'FullyCoupled', 'Segregated')
# Lines of the last block that progress() returns
LAST_LINES = 10
# Slack between a process's start and the `started` time in the info file
SLACK = 2.0

HEADER = re.compile(r'^<---- (.*) \((\w+)\)\s*-*$')
FOOTER = re.compile(r'^----- (.*) \((\w+)\)\s*-*>$')
PROGRESS = re.compile(r'Current Progress:\s+(\d+) % - ?(.*)$')
MEMORY = re.compile(r'^Memory: (\d+)/(\d+) ')
DOFS = re.compile(r'Number of degrees of freedom solved for: (\d+)'
                  r'(?: \(plus (\d+) internal DOFs\))?\.')
CORES = re.compile(r'Using .* with (\d+) cores?')
PARAMETER = re.compile(r'^(?:Continuation parameter|Parameter) '
                       r'(?!stepsize )(.+?)\s*\.$')
# An output time of a time-dependent solver: a step row or a '-' row
# marked 'out'
OUTPUT = re.compile(r'^\s*(?:\d+|-)\s+(\S+)\s+\S+\s+out\b')
# The time-stepping table and its step rows
TABLE = re.compile(r'^\s*Step\s+Time\s+Stepsize\b')
STEP = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s')
# Values that a new solver block starts afresh
PER_BLOCK = ('percent', 'task', 'dofs', 'solved_dofs', 'time', 'solver_time',
             'step_size')
LOG_KEYS = ('percent', 'task', 'parameter', 'block', 'block_open', 'dofs',
            'solved_dofs', 'memory_mb', 'peak_memory_mb', 'comsol_cores',
            'time', 'solver_time', 'step_size', 'last_lines')


################
# Problem size #
################

def problem_size(model: Model, /, *, study=None) -> dict:
    """
    Returns the size of the problem a study solves, before solving it:

    ```python
    mk.problem_size(model)
    # {'study': 'static',
    #  'steps': [{'step': 'stationary', 'type': 'Stationary',
    #             'dofs': 961286, 'fields': {'comp1.T': 961286},
    #             'solver': 'iterative', 'linear_solver': 'gmres'}],
    #  'mesh_elements': {'mesh': 637509},
    #  'machine': {'memory_mb': 36864, 'free_memory_mb': 19300,
    #              'cores': 14, 'comsol_cores': 4}}
    ```

    `study` is a study node or name; it may be left out when the model
    has one study. With several, call it for each one you will solve
    (`model.solve()` without a name solves them all).

    For each study step: `dofs`, the number of degrees of freedom
    including internal ones, and `fields`, the same per dependent
    variable (they add up to `dofs`); `None` for steps COMSOL cannot
    compile before solving: those that need the solution of the one
    before (e.g. a time-dependent step after a stationary one), or that
    cannot be compiled at all (solving shows why). COMSOL's log says
    "solved for N (plus M internal DOFs)" with `dofs` = N + M;
    `mk.progress` gives N as `solved_dofs`. `solver` is `'direct'`,
    `'iterative'`, `'segregated'` or `None` (not known), as in the solver
    sequence the study has (also one you changed) or else the one COMSOL
    would create, which picks iterative solvers for large 3D problems.
    A step added after the study's solver sequence was made gets the
    solver COMSOL would create. `linear_solver` is the iterative method, e.g. `'gmres'`, and `None`
    for direct solvers: the direct solver set (e.g. PARDISO) may be
    replaced when solving (by MUMPS on macOS); the progress log names the
    one used. With a parametric sweep, the sizes are those at the current
    parameter values; a sweep that changes the mesh changes them too.

    `mesh_elements` counts the elements of each mesh the study uses (by
    name; a name used in several components gets the tag added).
    `machine` is about the computer Python runs on (a remote COMSOL
    server may differ): its memory and free memory in MB (macOS and
    Windows, else `None`), its cores, and the cores COMSOL uses (set with
    `mph.start(cores=...)`).

    It does not estimate memory or time. As a guide, a direct solver
    needs far more memory than an iterative one: 17.5 GB against 2.8 GB
    for the whole COMSOL process at 961 thousand degrees of freedom. Use
    `mk.progress` to watch the real values while solving.

    The geometries must be built and the meshes the study uses built
    after their last change (`model.build(geom)`, `model.mesh()`), since
    compiling the equations would rebuild them silently; a component
    with physics needs a mesh. It compiles the equations once, which
    takes seconds for large models. Imported or copied meshes were not
    tried. Returns plain values and leaves nothing in the model.
    """
    if not isinstance(model, Model):
        raise TypeError(f'mk.problem_size takes a model, not {model!r}.')
    java = model.java
    study_java = _study(model, study)
    std = str(study_java.tag())
    interfaces = _check.active_physics(model)
    meshes = _check_meshes(model, study_java, interfaces)
    with _comsol.history_off(java):
        attached = _attached(java, std)
        steps, solvers = _sizes(java, study_java)
        if attached is not None:
            # a step added after the sequence was made is not in it
            solvers = {**solvers, **_sequence_solvers(attached)}
    for step in steps:
        step.update(solvers.get(step.pop('tag'), UNKNOWN))
    return {'study': _comsol.name_of(study_java), 'steps': steps,
            'mesh_elements': meshes, 'machine': _machine()}


def _study(model: Model, study) -> Any:
    """Returns the Java study to size."""
    studies = model.java.study()
    active = []
    for tag in studies.tags():
        try:
            if studies.get(tag).isActive():
                active.append(studies.get(tag))
        except Exception:
            continue
    if study is None:
        if not active:
            raise ValueError("The model has no study; add one, e.g. "
                             "(model/'studies').create().create("
                             "'Stationary').")
        if len(active) > 1:
            names = ', '.join(f'"{_comsol.name_of(s)}"' for s in active)
            raise ValueError(
                f'The model has several studies: {names}; pass study= the '
                'one to size, e.g. mk.problem_size(model, study='
                f'{_comsol.name_of(active[0])!r}), once for each '
                '(model.solve() without a name solves them all).')
        return active[0]
    wrong = f'study must be a study node or name, not {study!r}.'
    found = _comsol.find_node(studies, study, 'study', 'studies', wrong)
    if not any(str(s.tag()) == str(found.tag()) for s in active):
        raise ValueError(f'Study "{_comsol.name_of(found)}" is disabled.')
    return found


def _check_meshes(model: Model, study, interfaces) -> dict[str, int]:
    """
    Checks the geometries and meshes the study's steps use, and returns
    the element counts of the meshes. Compiling the equations would build
    an empty or changed mesh, or one of a changed geometry, silently.
    """
    java = model.java
    pairs: list[tuple[str, str]] = []
    found = False
    for tag in study.feature().tags():
        step = study.feature(tag)
        try:
            if not step.isActive():
                continue
            values = [str(v) for v in step.getStringArray('mesh')]
        except Exception:
            continue            # e.g. a parametric sweep
        found = True
        for pair in zip(values[::2], values[1::2]):
            if pair not in pairs:
                pairs.append(pair)
    if not found:
        for physics in interfaces:
            gtag = str(physics.geometry.tag())
            for mtag in physics.component.mesh().tags():
                if (gtag, str(mtag)) not in pairs:
                    pairs.append((gtag, str(mtag)))
    with_physics = {str(p.geometry.tag()): p.geometry for p in interfaces}
    for gtag, mtag in pairs:
        if mtag == 'nomesh' and gtag in with_physics:
            name = _comsol.name_of(with_physics[gtag])
            raise RuntimeError(
                f'Geometry "{name}" has physics but no mesh; the solve would '
                "give an empty solution. Create one, e.g. (model/'meshes')"
                '.create(geom), and run model.mesh().')
    used = [(g, m) for g, m in pairs if m != 'nomesh']
    for gtag in dict.fromkeys(g for g, _ in used):
        geometry = java.geom(gtag)
        _comsol.check_geometry_built(
            geometry, 'geometries/' + _comsol.name_of(geometry))
    sequences = [java.mesh(m) for _, m in used]
    for sequence in sequences:
        _comsol.check_mesh_built(sequence)
    names = [_comsol.name_of(s) for s in sequences]
    return {(n if names.count(n) == 1 else f'{n} ({s.tag()})'):
            int(s.getNumElem()) for n, s in zip(names, sequences)}


def _sizes(java, study) -> tuple[list[dict], dict[str, dict]]:
    """
    Compiles the equations of each step in a temporary solver sequence,
    the one COMSOL would create, and returns their sizes and the
    sequence's solvers. Call it with the history off.
    """
    sequences = java.sol()
    tag = str(sequences.uniquetag(ESTIMATE))
    sequence = sequences.create(tag)
    try:
        std = str(study.tag())
        try:
            sequence.study(std)
            sequence.createAutoSequence(std)
        except Exception as error:
            raise RuntimeError(f'COMSOL could not compile the equations: '
                               f'{_comsol.reason(error)}') from error
        solvers = _sequence_solvers(sequence)
        steps = []
        first = True
        for ftag in sequence.feature().tags():
            feature = sequence.feature(ftag)
            if str(feature.getType()) != 'StudyStep' or not feature.isActive():
                continue
            step_tag = str(feature.getString('studystep'))
            step = study.feature(step_tag)
            size: dict[str, Any] = {
                'step': _comsol.name_of(step), 'type': str(step.getType()),
                'tag': step_tag, 'dofs': None, 'fields': None}
            try:
                sequence.runFromTo(str(ftag), str(ftag))
                info = sequence.xmeshInfo()
                size['dofs'] = int(info.nDofs())
                size['fields'] = dict(zip(
                    [str(n) for n in info.fieldNames()],
                    [int(n) for n in info.fieldNDofs()]))
            except Exception as error:
                # later steps need the solution of the one before
                if first:
                    raise RuntimeError(f'COMSOL could not compile the '
                                       f'equations: {_comsol.reason(error)}'
                                       ) from error
            first = False
            steps.append(size)
        return steps, solvers
    finally:
        sequences.remove(tag)


def _attached(java, std: str) -> Any:
    """
    Returns the first solver sequence of the study (the one model.solve()
    runs, also if the user changed it), or None.
    """
    sequences = java.sol()
    for tag in sequences.tags():
        sequence = sequences.get(tag)
        for ftag in sequence.feature().tags():
            feature = sequence.feature(ftag)
            if str(feature.getType()) == 'StudyStep':
                try:
                    if str(feature.getString('study')) == std:
                        return sequence
                except Exception:
                    pass
                break
    return None


def _sequence_solvers(sequence) -> dict[str, dict]:
    """
    Pairs the steps of a solver sequence with their solvers by position:
    the first solver after a StudyStep, before the next one, is its own.
    """
    found: dict[str, dict] = {}
    step = None
    for ftag in sequence.feature().tags():
        feature = sequence.feature(ftag)
        kind = str(feature.getType())
        if kind == 'StudyStep':
            try:
                step = str(feature.getString('studystep'))
            except Exception:
                step = None
        elif (step is not None and step not in found
              and feature.isActive()):
            children = feature.feature()
            types = {str(t): str(children.get(t).getType())
                     for t in children.tags()}
            if any(t in SOLVER_CHILDREN for t in types.values()):
                found[step] = _solver_kind(children, types)
    return found


def _solver_kind(children, types: dict[str, str]) -> dict:
    """
    Returns the solver of a solver feature from its own children (not the
    coarse solver inside a multigrid): a segregated solver if one is on,
    else the linear solver the active FullyCoupled node names, else the
    only active linear solver (an eigenvalue solver has no FullyCoupled).
    """
    def active(tag):
        try:
            return bool(children.get(tag).isActive())
        except Exception:
            return False

    if any(t == 'Segregated' and active(tag) for tag, t in types.items()):
        return {'solver': 'segregated', 'linear_solver': None}
    coupled = [tag for tag, t in types.items()
               if t == 'FullyCoupled' and active(tag)]
    if coupled:
        try:
            chosen = str(children.get(coupled[0]).getString('linsolver'))
        except Exception:
            return UNKNOWN
    else:
        linear = [tag for tag, t in types.items()
                  if t in SOLVERS and active(tag)]
        if len(linear) != 1:
            return UNKNOWN
        chosen = linear[0]
    solver = SOLVERS.get(types.get(chosen, ''))
    if solver != 'iterative':
        return {'solver': solver, 'linear_solver': None}
    try:
        method = str(children.get(chosen).getString('linsolver'))
    except Exception:
        method = None
    return {'solver': solver, 'linear_solver': method}


def _machine() -> dict:
    """
    Returns the memory (macOS and Windows) and cores of this computer and
    of COMSOL.
    """
    total = None
    if sys.platform == 'win32':
        from . import _winproc
        memory = _winproc.memory()
        total = memory['total_mb'] if memory is not None else None
    elif sys.platform == 'darwin':
        out = _run(['sysctl', '-n', 'hw.memsize'])
        if out.strip().isdigit():
            total = int(out) // 2**20
    return {'memory_mb': total, 'free_memory_mb': _free_memory_mb(),
            'cores': os.cpu_count(), 'comsol_cores': _comsol_cores()}


def _comsol_cores() -> int | None:
    """Returns the cores COMSOL uses, as MPh's `client.cores` does."""
    try:
        import jpype  # type: ignore[import-untyped]
        util = jpype.JClass('com.comsol.model.util.ModelUtil')
        return int(str(util.getPreference(
            'cluster.processor.numberofprocessors')))
    except Exception:
        return None


################
# Progress log #
################

@overload
def log_progress(path: None, /) -> None: ...
@overload
def log_progress(path: str | os.PathLike, /) -> Path: ...
def log_progress(path, /):
    """
    Makes COMSOL write its progress log to a file, for `mk.progress`:

    ```python
    client = mph.start(cores=4)
    mk.log_progress('solve.log')     # before loading or solving
    model = client.load('model.mph')
    model.solve()
    ```

    Returns the absolute path of the log; `None` switches the log off.
    The log is the one the COMSOL Desktop shows: progress in percent,
    the task, memory, degrees of freedom, solver iterations and time
    steps; mesh builds go in too. The call empties the file. Call it
    again before each solve: otherwise the next solve is appended with
    no mark where it starts. The setting holds for the whole COMSOL
    session (all models). A relative path is relative to Python's working
    directory.

    It also writes `<path>.json` with the ID of this Python process and
    the time of the call, from which `mk.progress` finds the processes
    that solve; a file a launcher wrote for this process (on Windows also
    for the virtual environment's python.exe that started it) keeps the
    launcher's time.
    See `help(mk.progress)` for running a long solve in the background.
    """
    if path is not None and not isinstance(path, (str, os.PathLike)):
        raise TypeError(f'path must be a file name or None, not {path!r}.')
    log = None if path is None else Path(path).resolve()
    if log is not None and not log.parent.is_dir():
        raise FileNotFoundError(f'No folder {str(log.parent)!r} for the '
                                'log.')
    import jpype  # type: ignore[import-untyped]
    if not jpype.isJVMStarted():
        raise RuntimeError('COMSOL is not running; start it first, e.g. '
                           'client = mph.start().')
    util = jpype.JClass('com.comsol.model.util.ModelUtil')
    if log is None:
        util.showProgress(False)
        return None
    util.showProgress(str(log))
    info = _read_info(log)
    pid = os.getpid()
    if (info is None or _started(info.get('started')) is None
            or (info.get('pid') != pid and not _launched_me(info))):
        info = {'pid': pid, 'started': _now()}
    _write_info(log, {'pid': pid, 'started': info['started']})
    return log


def progress(path, /, *, pid: int | None = None) -> dict:
    """
    Returns the progress of a solve from COMSOL's progress log and the
    operating system, from any Python process (COMSOL is not needed):

    ```python
    mk.progress('/abs/path/solve.log')
    # {'exists': True, 'started': '2026-10-01T11:45:13+09:00',
    #  'elapsed_s': 41.2, 'percent': 55, 'task': 'Matrix factorization',
    #  'parameter': None,
    #  'block': 'Stationary Solver 1 in static/Solution 1',
    #  'block_open': True, 'dofs': 961286, 'solved_dofs': 886292,
    #  'memory_mb': 9182, 'peak_memory_mb': 9182, 'comsol_cores': 4,
    #  'time': None, 'solver_time': None, 'step_size': None,
    #  'last_lines': ['Started at Oct 1, 2026, 11:47:00 AM.', ...],
    #  'updated_s_ago': 0.4, 'pid': 74165, 'alive': True,
    #  'cpu_percent': 395.0, 'rss_mb': 9708, 'processes': [...],
    #  'free_memory_mb': 13200, 'swap_used_mb': 0}
    ```

    From the log (written by `mk.log_progress` or `comsol batch
    -batchlog`; `None` where the log has no such line yet):

    - `percent`, `task`: COMSOL's progress and what it does. The percent is
      of all that COMSOL is running at the time (a whole study, say), not
      just of `task`, and may go back; over a parametric sweep it covers the
      whole sweep, and in a single time-dependent solve it roughly follows
      the time solved, barely moving during the first, small time steps. It
      is `None` in a `block` until that block logs it; for a short solve
      that may be only at 100.
    - `parameter`: the parameter value of a sweep being solved, e.g.
      `'hh = 300'` (the log does not say how many follow).
    - `block`, `block_open`: the solver step being run, e.g.
      `'Stationary Solver 1 in static/Solution 1'`, and whether it is
      still open (also after a crash).
    - `dofs`, `solved_dofs`: degrees of freedom, with and without internal
      ones (`mk.problem_size` gives `dofs`).
    - `memory_mb`, `peak_memory_mb`: memory of the COMSOL process now and
      at most since it started.
    - `comsol_cores`: the cores COMSOL uses.
    - `time`, `solver_time`, `step_size`: in a Time-Dependent Solver `block`
      (`None` in other blocks), the last output time logged in it, and the
      time and size of its last logged time step. All in seconds, whatever
      the study's time unit. `time` may be behind `solver_time` (between
      output times, with output times COMSOL does not log, or past the end
      time) or ahead of it: output times that fall within a time step are
      logged before that step's line.
    - `last_lines`: the last lines of the current step, without the
      progress and memory lines, e.g. nonlinear iterations or time steps.
    - `updated_s_ago`: seconds since the log last changed.

    From the info file `<path>.json` and the operating system: `started`
    and `elapsed_s` of the solving process (counted from `started`, also
    after the process ended), its `pid`, whether it is
    `alive`, and `processes`, its own and its children's and those of its
    process group (Windows has none: there also the children of the
    process after it ended, if they started before the info file was
    written), with `cpu_percent` and `rss_mb` (memory, also of the
    Python running MPh) and their sums; `free_memory_mb` and
    `swap_used_mb` of the computer (on Windows the page files in use). A
    process that started after `started` (an ID used again) counts as
    not alive. `pid=` overrides the info file's process, e.g. for a job
    started without one (`started` and `elapsed_s` stay the info file's;
    on Windows an ended `pid=` lists no children). Free memory and swap
    come from macOS and Windows; on Linux the summed `cpu_percent` is
    `None` (ps gives averages there). On Windows `cpu_percent` is
    measured over half a second, which the call takes longer, the list
    is in the order the processes started, without the console host
    (`conhost.exe`), and `command` is the program's name without `.exe`.
    Linux was not tried.

    It does not judge whether the solve goes well, and errors do not
    show in the log: read the output of the solving script. It reads the
    whole log each time, in about 0.2 s per 10 MB (some 150,000 lines of
    time steps).

    Long solves in the background (each shell command of an agent runs in
    a new process). `solve.py`, next to the input `m.mph`:

    ```python
    from pathlib import Path
    import mph
    import mphkit as mk

    here = Path(__file__).resolve().parent
    client = mph.start(cores=4)
    mk.log_progress(here/'solve.log')        # before loading
    model = client.load(here/'m.mph')
    model.solve()
    model.save(here/'m_solved.mph')          # not over the input
    print('saved', flush=True)
    ```

    `launch.py` starts it in a new session (on Windows without a
    console), so that it outlives the shell command and neither Ctrl+C
    nor closing the terminal reaches it, and prints its ID and the log's
    path:

    ```python
    import datetime, json, os, subprocess, sys
    from pathlib import Path
    import mphkit as mk

    here = Path(__file__).resolve().parent
    log = here/'solve.log'
    old = mk.progress(log)
    if old['alive'] or old['processes']:
        sys.exit(f'Still running: process {old["pid"]}.')
    for name in ('solve.log', 'solve.log.json', 'solve.out',
                 'm_solved.mph'):
        (here/name).unlink(missing_ok=True)
    options = ({'creationflags': subprocess.CREATE_NO_WINDOW}
               if sys.platform == 'win32' else {'start_new_session': True})
    with open(here/'solve.out', 'w') as out:
        job = subprocess.Popen(
            [sys.executable, '-u', 'solve.py'], cwd=here,
            stdin=subprocess.DEVNULL, stdout=out,
            stderr=subprocess.STDOUT, **options)
    info = {'pid': job.pid,
            'started': datetime.datetime.now().astimezone().isoformat()}
    (here/'info.tmp').write_text(json.dumps(info))
    os.replace(here/'info.tmp', here/'solve.log.json')
    print(job.pid, log)
    ```

    Run `launch.py` once with the Python that has mphkit (e.g. `uv run
    python launch.py`), then check with `mk.progress` and the printed
    path; to wait a minute first:

    ```
    uv run python -c "import time, mphkit as mk; time.sleep(60); print(mk.progress('/abs/path/solve.log'))"
    ```

    On Windows, in a virtual environment made by venv or uv, the ID
    `launch.py` prints is that of the environment's `python.exe`, a launcher
    that starts the real Python running `solve.py` and ends with it.
    `mk.progress` gives that ID as `pid` until `solve.py` calls
    `mk.log_progress`, and the real Python's ID after that, leaving the
    launcher out of `processes`; `pid=` with the printed ID lists the
    launcher too.

    While COMSOL starts, `exists` is `False` and `alive` `True`. The solve
    is over when `alive` is `False` and `processes` is empty; it succeeded
    if `solve.out` has the line `saved`, else `solve.out` says why.
    `percent` 100 alone does not mean it is over: the last time steps may
    still be logged, and `solve.py` may be saving, quitting or doing more
    work, which may log progress again.

    To stop it on macOS (Linux not tried), end the Python process only:
    `os.kill(pid, signal.SIGTERM)`; COMSOL ends with it, and `m.mph`
    stays as it was (Java may leave an `hs_err_pid*.log` file next to it).
    Do not signal the whole process group: the COMSOL server then may
    hang without its client. If `processes` is not empty some ten seconds
    later (e.g. when stopped while COMSOL was starting),
    `os.killpg(pid, signal.SIGKILL)` ends them.

    On Windows end every process `mk.progress` lists, the newest first
    (ending Python only leaves the COMSOL server running when stopped
    while COMSOL starts):

    ```python
    for p in reversed(mk.progress(log)['processes'] or []):
        try:
            os.kill(p['pid'], signal.SIGTERM)
        except OSError:
            pass                                 # ended meanwhile
    ```

    The list is checked against start times, so an ID used again is not
    ended. If `launch.py` cannot delete one of its files and `processes`
    is empty, a COMSOL server was left behind: end `comsolmphserver.exe`
    in the Task Manager, or all of yours with
    `taskkill /im comsolmphserver.exe /f`.

    Batch mode works the same with, in `launch.py`, the command
    `[comsol, 'batch', '-np', '4', '-inputfile', str(here/'m.mph'),
    '-outputfile', str(here/'m_solved.mph'), '-batchlog', str(log)]`,
    where `comsol = Path(mph.discovery.backend()['root'])/'bin'/'comsol'`
    (it may not be on the PATH), and `m_solved.mph.status` added to the
    files to delete. It succeeded if that file's last line is `Done` and
    `m_solved.mph` exists. Stop it with `os.killpg(pid, signal.SIGTERM)`:
    ending only the script leaves COMSOL computing. Batch mode may
    rewrite COMSOL's preferences file. On Windows the command is
    `[comsolbatch, '-np', '4', ...]` with `comsolbatch =
    Path(mph.discovery.backend()['root'])/'bin'/'win64'/'comsolbatch.exe'`
    (`comsol.exe batch` opens the COMSOL Desktop there),
    `m_solved.mph.recovery` is deleted too, and it stops as above.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f'path must be a file name, not {path!r}.')
    if pid is not None and not _is_pid(pid):
        if isinstance(pid, int) and not isinstance(pid, bool):
            raise ValueError(f'pid must be positive, not {pid}.')
        raise TypeError(f'pid must be a process ID, not {pid!r}.')
    log = Path(path)
    info = _read_info(log)
    started = _started(info.get('started')) if info else None
    text: str | None = None
    changed = 0.0
    try:
        text = log.read_bytes().decode('utf-8', errors='replace')
        changed = log.stat().st_mtime
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        text = None             # also deleted meanwhile, by a launcher
    result: dict[str, Any] = {'exists': text is not None}
    result['started'] = (started.isoformat() if started is not None
                         else None)
    result['elapsed_s'] = (round(time.time() - started.timestamp(), 1)
                           if started is not None else None)
    if text is not None:
        result.update(_parse_log(text))
        result['updated_s_ago'] = round(time.time() - changed, 1)
    else:
        result.update(dict.fromkeys(LOG_KEYS))
        result['updated_s_ago'] = None
    from_info = pid is None
    if from_info and info and _is_pid(info.get('pid')):
        pid = info['pid']
    result['pid'] = pid
    result.update(_processes(pid, started if from_info else None,
                             _info_time(log) if from_info else None))
    result['free_memory_mb'] = _free_memory_mb()
    result['swap_used_mb'] = _swap_used_mb()
    return result


def _parse_log(text: str) -> dict:
    """
    Reads a COMSOL progress log. The blocks of the temporary sequences of
    `problem_size` are skipped, and so are the lines just before them;
    other lines outside blocks (progress, memory) count once the next
    block starts or the log ends.
    """
    lines = text.split('\n')
    lines.pop()                   # empty, or a line still being written
    values: dict[str, Any] = dict.fromkeys(LOG_KEYS)
    pending: dict[str, Any] = {}
    block_lines: list[str] = []
    shown: list[str] = []         # lines outside problem_size's blocks
    inside = False                # in a block
    skip = False                  # in a block of problem_size
    table = False                 # in a time-stepping table
    for line in lines:
        line = line.rstrip('\r')
        header = HEADER.match(line)
        if header:
            if header.group(2).startswith(ESTIMATE):
                pending.clear()
                inside = skip = True
                continue
            values.update(pending)
            pending.clear()
            for key in PER_BLOCK:
                values[key] = None
            if header.group(1).startswith('Compile Equations'):
                values['parameter'] = None
            values['block'] = header.group(1)
            values['block_open'] = True
            block_lines = []
            inside, skip, table = True, False, False
            continue
        footer = FOOTER.match(line)
        if footer and inside:
            if not skip:
                values['block_open'] = False
            inside = skip = table = False
            continue
        if skip:
            continue
        if not PROGRESS.search(line) and not MEMORY.match(line):
            shown.append(line)
        target = values if inside else pending
        _read_line(line, target)
        if not inside:
            continue
        if TABLE.match(line):
            table = True
        elif table and (step := STEP.match(line)):
            values['solver_time'] = _number(step.group(2))
            values['step_size'] = _number(step.group(3))
        if not PROGRESS.search(line) and not MEMORY.match(line):
            block_lines.append(line)
    values.update(pending)
    if values['block'] is None:
        block_lines = shown
    values['last_lines'] = block_lines[-LAST_LINES:] if lines else None
    return values


def _read_line(line: str, values: dict):
    """Stores what one log line says in `values`."""
    if found := PROGRESS.search(line):
        values['percent'] = int(found.group(1))
        values['task'] = found.group(2).strip() or None
    elif found := MEMORY.match(line):
        values['memory_mb'] = int(found.group(1))
        values['peak_memory_mb'] = int(found.group(2))
    elif found := DOFS.search(line):
        values['solved_dofs'] = int(found.group(1))
        values['dofs'] = int(found.group(1)) + int(found.group(2) or 0)
    elif found := CORES.search(line):
        values['comsol_cores'] = int(found.group(1))
    elif found := PARAMETER.match(line):
        values['parameter'] = found.group(1)
    elif found := OUTPUT.match(line):
        values['time'] = _number(found.group(1))


def _is_pid(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value > 0)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


#############
# Processes #
#############

def _info_path(log: Path) -> Path:
    return log.with_name(log.name + '.json')


def _read_info(log: Path) -> dict | None:
    """Returns the info file of a log, or None if missing or unreadable."""
    try:
        info = json.loads(_info_path(log).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _info_time(log: Path) -> float | None:
    """Returns when the info file of a log was written, or None."""
    try:
        return _info_path(log).stat().st_mtime
    except OSError:
        return None


def _launched_me(info: dict) -> bool:
    """
    Tells whether an info file was written for the launcher that started
    this process on Windows: a virtual environment's python.exe starts
    the real Python as its child, so a launcher's `job.pid` is the
    parent's. Its start time must match, against a process ID used again.
    """
    if sys.platform == 'win32':
        from . import _winproc
        started = _started(info.get('started'))
        parent = os.getppid()
        if info.get('pid') != parent or started is None:
            return False
        facts = _winproc.facts(parent)
        created = facts['created'] if facts is not None else None
        return (created is not None
                and abs(created - started.timestamp()) <= SLACK)
    return False


def _write_info(log: Path, info: dict):
    """Writes the info file of a log at once, never half."""
    target = _info_path(log)
    temporary = target.with_name(f'{target.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(info), encoding='utf-8')
    os.replace(temporary, target)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _started(value) -> datetime.datetime | None:
    """Reads an ISO time; one without a time zone is local time."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment.astimezone() if moment.tzinfo is None else moment


def _run(command: list[str]) -> str:
    """
    Returns the output of a command, run with the C locale (others write
    decimal commas or localized dates), or '' if it fails.
    """
    try:
        run = subprocess.run(command, capture_output=True, text=True,
                             env={**os.environ, 'LC_ALL': 'C'}, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ''
    return run.stdout if run.returncode == 0 else ''


def _processes(pid: int | None, started: datetime.datetime | None,
               written: float | None = None) -> dict:
    """
    Returns whether a process is alive and its processes: itself, its
    descendants and the members of its process group (they outlive it:
    a batch job, or a COMSOL server left behind); on Windows, which has
    no process groups, its descendants also after it ended. `started` is
    the time the info file gives, against a process ID used again, and
    `written` (Windows only) the time the file was written.
    """
    gone: dict[str, Any] = {'alive': False, 'cpu_percent': None,
                            'rss_mb': None, 'processes': []}
    if pid is None:
        return {**gone, 'alive': None}
    if sys.platform == 'win32':
        from . import _winproc
        return _winproc.processes(
            pid, started.timestamp() if started is not None else None,
            written, SLACK)
    rows = []
    for line in _run(['ps', '-A', '-o',
                      'pid=,ppid=,pgid=,stat=,pcpu=,rss=,comm=']).splitlines():
        parts = line.split(None, 6)
        if len(parts) == 7 and parts[0].isdigit():
            rows.append(parts)
    own = next((r for r in rows if int(r[0]) == pid), None)
    try:
        os.kill(pid, 0)
        alive = True
    except PermissionError:
        alive = True
    except (ProcessLookupError, OSError, OverflowError):
        alive = False
    if own is not None and own[3].startswith('Z'):
        alive = False                     # ended, not yet reaped
    if own is not None and started is not None:
        elapsed = _elapsed(_run(['ps', '-o', 'etime=', '-p', str(pid)]))
        if (elapsed is not None
                and time.time() - elapsed > started.timestamp() + SLACK):
            return gone                   # another process with that ID
    members = {pid}
    changed = True
    while changed:
        changed = False
        for row in rows:
            child = int(row[0])
            if child not in members and (int(row[1]) in members
                                         or int(row[2]) == pid):
                members.add(child)
                changed = True
    kept = [r for r in rows
            if int(r[0]) in members and not r[3].startswith('Z')]
    cpu = [_number(r[4]) for r in kept]
    rss = [int(r[5]) // 1024 if r[5].isdigit() else None for r in kept]
    processes = [{'pid': int(r[0]), 'command': r[6].rsplit('/', 1)[-1],
                  'cpu_percent': c, 'rss_mb': m}
                 for r, c, m in zip(kept, cpu, rss)]
    linux = sys.platform.startswith('linux')
    return {'alive': alive,
            'cpu_percent': (None if linux or None in cpu   # Linux: averages
                            else round(sum(c or 0 for c in cpu), 1)),
            'rss_mb': None if None in rss else sum(r or 0 for r in rss),
            'processes': processes}


def _elapsed(text: str) -> float | None:
    """Reads the `etime` of ps: [[dd-]hh:]mm:ss."""
    text = text.strip()
    found = re.fullmatch(r'(?:(?:(\d+)-)?(\d+):)?(\d+):(\d+)', text)
    if not found:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in found.groups())
    return ((days*24 + hours)*60 + minutes)*60 + seconds


def _free_memory_mb() -> int | None:
    """
    Free memory of macOS (free, inactive and speculative pages) and of
    Windows (available memory).
    """
    if sys.platform == 'win32':
        from . import _winproc
        memory = _winproc.memory()
        return memory['free_mb'] if memory is not None else None
    if sys.platform != 'darwin':
        return None
    text = _run(['vm_stat'])
    size = re.search(r'page size of (\d+) bytes', text)
    pages = [re.search(rf'^Pages {kind}:\s+(\d+)\.', text, re.M)
             for kind in ('free', 'inactive', 'speculative')]
    if not size or not all(pages):
        return None
    total = sum(int(p.group(1)) for p in pages if p)
    return total*int(size.group(1)) // 2**20


def _swap_used_mb() -> int | None:
    """Swap in use on macOS, page files in use on Windows."""
    if sys.platform == 'win32':
        from . import _winproc
        return _winproc.swap_used_mb()
    if sys.platform != 'darwin':
        return None
    found = re.search(r'used = ([\d.]+)([KMGT])',
                      _run(['sysctl', '-n', 'vm.swapusage']))
    if not found:
        return None
    scale = {'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}[found.group(2)]
    return round(float(found.group(1))*scale)
