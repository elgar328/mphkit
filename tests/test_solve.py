"""
Checks mk.problem_size, mk.log_progress and mk.progress: the parser on
real COMSOL progress logs (tests/data/progress), the process facts on
plain child processes, and the COMSOL side on the example's model.
"""
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import mphkit as mk
from conftest import java_export
from mphkit import _solve
from test_example import plate_with_holes

data = Path(__file__).parent/'data'/'progress'
posix = pytest.mark.skipif(os.name == 'nt', reason='POSIX processes')
macos = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')


def parse(name, lines=None):
    text = (data/name).read_bytes().decode('utf-8')
    if lines is not None:
        text = '\n'.join(text.split('\n')[:lines]) + '\n'
    return _solve._parse_log(text)


def wait_for(condition, timeout=10):
    """Waits until `condition()` is true."""
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, 'timed out'
        time.sleep(0.05)


def state(pid):
    """The ps state of a process, '' if there is none."""
    return _solve._run(['ps', '-o', 'stat=', '-p', str(pid)]).strip()


def write_info(log, pid, started):
    Path(str(log) + '.json').write_text(json.dumps({'pid': pid,
                                                    'started': started}))


def now(offset=0.0):
    moment = datetime.datetime.now().astimezone()
    return (moment + datetime.timedelta(seconds=offset)).isoformat()


##########
# Parser #
##########

def test_transient():
    values = parse('transient.log')
    assert values['time'] == 2000
    assert values['block_open'] is False
    assert values['percent'] == 100
    assert values['block'] == ('Time-Dependent Solver 1 in '
                               'transient/Solution 1')


@pytest.mark.parametrize('name', ['transient_strict.log',
                                  'transient_intermediate.log'])
def test_output_times_on_step_rows(name):
    # strict and intermediate time stepping print output times on step
    # rows too
    assert parse(name)['time'] == 2000


def test_solver_time():
    # free time stepping prints few output times: the stepper is ahead
    values = parse('transient_free.log')
    assert values['time'] == 2000
    assert values['solver_time'] == 2199.2
    assert values['step_size'] == 200


def test_first_step_row():
    # the first row has '-' as its step size
    values = parse('transient_free.log', 42)
    assert values['solver_time'] == 0
    assert values['step_size'] is None


def test_time_in_seconds():
    # a study in minutes: 30 min
    assert parse('transient_minutes.log')['time'] == 1800


def test_sweeps():
    values = parse('study_sweep.log')
    assert values['parameter'] == 'hm = 1.3'
    assert values['dofs'] == 206794 + 29658
    assert values['percent'] == 100
    assert parse('auxiliary_sweep.log')['parameter'] == 'hh = 300'


def test_empty_task():
    values = parse('iterative.log')
    assert values['task'] is None
    assert values['percent'] == 100


def test_killed():
    values = parse('killed.log')
    assert values['block_open'] is True
    assert values['percent'] == 55
    assert values['task'] == 'Matrix factorization'
    assert values['dofs'] == 961286
    assert values['memory_mb'] == values['peak_memory_mb'] == 9182
    assert values['comsol_cores'] == 4


def test_batch():
    values = parse('batch.log')
    assert (values['percent'], values['task']) == (100, 'Done')
    assert values['block_open'] is False


def test_new_block():
    # the values of the block before do not carry over
    values = parse('batch_cut.log')
    assert values['block'] == 'Stationary Solver 1 in static/Solution 1'
    assert values['block_open'] is True
    assert values['percent'] is None
    assert values['dofs'] is None


def test_nonlinear():
    values = parse('nonlinear.log')
    assert values['time'] is None
    assert values['solver_time'] is None


def test_eigenfrequency_log():
    values = parse('eigenfrequency.log')
    assert values['dofs'] == values['solved_dofs'] == 1512
    assert values['block'].startswith('Eigenvalue Solver 1')


def test_estimate_skipped():
    # the blocks of problem_size, and the progress lines before them
    values = parse('estimate_then_solve.log')
    assert (values['percent'], values['task']) == (100, None)
    assert (values['dofs'], values['solved_dofs']) == (4767, 2849)


def test_last_lines():
    values = parse('killed.log')
    assert values['last_lines'][0].startswith('Started at')
    assert values['last_lines'][-1] == 'Orthonormal null-space function used.'
    for name in data.glob('*.log'):
        lines = parse(name.name)['last_lines']
        assert 0 < len(lines) <= 10
        assert not any('Current Progress' in line or
                       line.startswith('Memory:') for line in lines)


def test_last_lines_before_solving():
    # only problem_size's blocks so far: their lines are left out
    lines = parse('estimate_then_solve.log', 56)['last_lines']
    assert lines[-1] == 'Minimum element quality: 0.0647'
    assert not any(line.startswith('Started at') for line in lines)


def test_line_ends():
    text = (data/'killed.log').read_text()
    crlf = _solve._parse_log(text.replace('\n', '\r\n'))
    assert crlf == _solve._parse_log(text)


def test_line_being_written():
    text = (data/'batch_cut.log').read_text()
    values = _solve._parse_log(text + '           Current Progress:  99 % - x')
    assert values['percent'] is None
    # a character cut in half at the end
    raw = (text + 'Current Progress:  7 % - é').encode()[:-1]
    values = _solve._parse_log(raw.decode('utf-8', errors='replace'))
    assert values['percent'] is None


############
# progress #
############

def test_no_log(tmp_path):
    result = mk.progress(tmp_path/'solve.log')
    assert result['exists'] is False
    assert all(result[key] is None for key in _solve.LOG_KEYS)
    assert result['pid'] is None and result['alive'] is None
    assert result['processes'] == []


def test_log_file(tmp_path):
    log = tmp_path/'solve.log'
    log.write_bytes((data/'killed.log').read_bytes())
    result = mk.progress(str(log))
    assert result['exists'] is True
    assert result['percent'] == 55
    assert result['updated_s_ago'] >= 0
    assert result['started'] is None and result['elapsed_s'] is None
    json.dumps(result)


def test_arguments(tmp_path):
    with pytest.raises(TypeError):
        mk.progress(None)
    with pytest.raises(ValueError):
        mk.progress(tmp_path/'solve.log', pid=0)
    with pytest.raises(TypeError):
        mk.progress(tmp_path/'solve.log', pid='1')
    with pytest.raises(TypeError):
        mk.progress(tmp_path/'solve.log', pid=True)


@posix
def test_no_such_process(tmp_path):
    for pid in (2**22 + 12345, 2**40):
        result = mk.progress(tmp_path/'solve.log', pid=pid)
        assert result['alive'] is False and result['processes'] == []


@posix
def test_this_process(tmp_path):
    result = mk.progress(tmp_path/'solve.log', pid=os.getpid())
    assert result['alive'] is True
    assert os.getpid() in [p['pid'] for p in result['processes']]
    assert result['rss_mb'] > 0
    if sys.platform == 'darwin':
        assert isinstance(result['cpu_percent'], float)


@posix
def test_ended(tmp_path):
    child = subprocess.Popen(['true'])
    # not yet reaped: a zombie, still there for os.kill
    wait_for(lambda: state(child.pid).startswith('Z'))
    result = mk.progress(tmp_path/'solve.log', pid=child.pid)
    assert result['alive'] is False
    assert result['processes'] == []
    child.wait()
    assert mk.progress(tmp_path/'solve.log', pid=child.pid)['alive'] is False


def session(tmp_path):
    """A shell in a new session with two children, as a launched solve."""
    # 'true' at the end: a shell may run its last command in its place
    leader = subprocess.Popen(['sh', '-c', 'sleep 30 & sleep 30; true'],
                              start_new_session=True)
    wait_for(lambda: len(_solve._run(['pgrep', '-g', str(leader.pid)])
                         .split()) == 3)
    return leader


@posix
def test_group(tmp_path):
    leader = session(tmp_path)
    try:
        result = mk.progress(tmp_path/'solve.log', pid=leader.pid)
        commands = sorted(p['command'] for p in result['processes'])
        assert commands == ['sh', 'sleep', 'sleep']
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait()


@posix
def test_group_outlives_leader(tmp_path):
    # as a batch job whose script ended: its group computes on
    log = tmp_path/'solve.log'
    leader = session(tmp_path)
    write_info(log, leader.pid, now(-1))
    try:
        leader.kill()
        leader.wait()
        result = mk.progress(log)
        assert result['pid'] == leader.pid
        assert result['alive'] is False
        assert [p['command'] for p in result['processes']] == ['sleep',
                                                               'sleep']
    finally:
        os.killpg(leader.pid, signal.SIGKILL)


@posix
def test_process_id_used_again(tmp_path):
    log = tmp_path/'solve.log'
    write_info(log, os.getpid(), now())
    assert mk.progress(log)['alive'] is True
    # a process that started after the info file: not the one it means
    write_info(log, os.getpid(), '2000-01-01T00:00:00+00:00')
    result = mk.progress(log)
    assert result['alive'] is False
    assert result['processes'] == []
    assert result['elapsed_s'] > 0
    # pid= is taken as given
    assert mk.progress(log, pid=os.getpid())['alive'] is True


@posix
@pytest.mark.parametrize('locale', ['de_DE.UTF-8', 'ko_KR.UTF-8'])
def test_locale(tmp_path, monkeypatch, locale):
    # ps and sysctl write decimal commas and local dates in some locales
    monkeypatch.setenv('LC_ALL', locale)
    monkeypatch.setenv('LANG', locale)
    log = tmp_path/'solve.log'
    write_info(log, os.getpid(), now())
    result = mk.progress(log)
    assert result['alive'] is True
    if sys.platform == 'darwin':
        assert isinstance(result['cpu_percent'], float)
        assert isinstance(result['swap_used_mb'], int)


def test_info_files(tmp_path):
    log = tmp_path/'solve.log'
    # a time without a time zone is local time
    write_info(log, os.getpid(), datetime.datetime.now().isoformat())
    result = mk.progress(log)
    assert result['started'].endswith(now()[-6:])
    assert 0 <= result['elapsed_s'] < 60
    for broken in ('', '{"pid": ', '[1, 2]', '{"pid": true}'):
        Path(str(log) + '.json').write_text(broken)
        result = mk.progress(log)
        assert result['pid'] is None and result['alive'] is None


@macos
def test_machine(tmp_path):
    result = mk.progress(tmp_path/'solve.log')
    assert result['free_memory_mb'] > 0
    assert result['swap_used_mb'] >= 0
    machine = _solve._machine()
    assert machine['memory_mb'] > machine['free_memory_mb']
    assert machine['cores'] == os.cpu_count()


def test_elapsed():
    assert _solve._elapsed('00:05') == 5
    assert _solve._elapsed('01:02:03') == 3723
    assert _solve._elapsed('2-01:00:00\n') == 2*86400 + 3600
    assert _solve._elapsed('') is None


##########
# COMSOL #
##########

@pytest.fixture
def plate(client):
    """The example's model with one hole, meshed."""
    model, geom, _ = plate_with_holes.build_model(client, 1)
    model.mesh()
    yield model, geom
    client.remove(model)


@pytest.fixture
def logging():
    """Switches the progress log off afterwards: it holds client-wide."""
    yield
    mk.log_progress(None)


def leaves_nothing(model, tmp_path):
    """The model's state that problem_size must leave as it was."""
    java = model.java
    return (java_export(model, tmp_path/'state.java'), model.datasets(),
            [str(t) for t in java.sol().tags()],
            [str(t) for t in java.batch().tags()],
            {str(t): int(java.mesh(t).getNumElem())
             for t in java.mesh().tags()})


def test_plate(client, plate, tmp_path):
    model, geom = plate
    before = leaves_nothing(model, tmp_path)
    size = mk.problem_size(model)
    assert size['study'] == 'static'
    [step] = size['steps']
    assert step['step'] == 'stationary' and step['type'] == 'Stationary'
    assert step['fields'] == {'comp1.T': step['dofs']}
    assert (step['solver'], step['linear_solver']) == ('direct', None)
    assert list(size['mesh_elements']) == ['mesh']
    assert size['mesh_elements']['mesh'] > 0
    assert size['machine']['comsol_cores'] == client.cores
    assert size['machine']['cores'] == os.cpu_count()
    assert mk.problem_size(model, 'static')['steps'] == size['steps']
    assert mk.problem_size(model, model/'studies'/'static')['steps'] == \
        size['steps']
    assert leaves_nothing(model, tmp_path) == before
    json.dumps(size)


def test_two_plate(client, tmp_path, logging):
    # the size, and the same numbers in the solve's log (the mesh, and so
    # the numbers, vary a little from run to run: 4767 = 2849 + 1918)
    model, geom, _ = plate_with_holes.build_model(client, 2)
    try:
        model.mesh()
        size = mk.problem_size(model)
        dofs = size['steps'][0]['dofs']
        log = mk.log_progress(tmp_path/'solve.log')
        assert log == (tmp_path/'solve.log').resolve() and log.is_file()
        info = json.loads(Path(str(log) + '.json').read_text())
        assert info['pid'] == os.getpid()
        model.solve()
        result = mk.progress(log)
        assert result['dofs'] == dofs
        assert 0.5*dofs < result['solved_dofs'] < dofs
        assert (result['percent'], result['block_open']) == (100, False)
        assert result['pid'] == os.getpid() and result['alive'] is True
        assert result['started'] == info['started']
        assert result['comsol_cores'] == size['machine']['comsol_cores']
        # again in this process: the start time stays
        mk.log_progress(log)
        assert json.loads(Path(str(log) + '.json').read_text()) == info
        assert log.stat().st_size == 0
        # switched off: the log stays as it is
        mk.log_progress(None)
        length = log.stat().st_size
        model.solve()
        assert log.stat().st_size == length
    finally:
        client.remove(model)


def test_log_progress_arguments(tmp_path):
    # checked before COMSOL is needed
    with pytest.raises(TypeError):
        mk.log_progress(1)
    with pytest.raises(FileNotFoundError):
        mk.log_progress(tmp_path/'missing'/'solve.log')


def test_arguments_comsol(client, plate):
    model, geom = plate
    with pytest.raises(TypeError, match='takes a model'):
        mk.problem_size(geom)
    with pytest.raises(LookupError, match='No study "nothing"'):
        mk.problem_size(model, 'nothing')
    (model/'studies').create(name='second').create('Stationary')
    with pytest.raises(ValueError, match='"static", "second"'):
        mk.problem_size(model)
    assert mk.problem_size(model, 'second')['study'] == 'second'


def test_no_study(model):
    with pytest.raises(ValueError, match='no study'):
        mk.problem_size(model)


def test_empty_mesh(client):
    model, geom, _ = plate_with_holes.build_model(client, 1)
    try:
        with pytest.raises(RuntimeError, match='is empty; run model.mesh'):
            mk.problem_size(model)
        assert model.java.mesh('mesh1').isEmpty()
    finally:
        client.remove(model)


def test_changed_mesh(plate):
    model, geom = plate
    mesh = model.java.mesh('mesh1')
    elements = mesh.getNumElem()
    mk.set(model/'meshes'/'mesh'/'Size', hauto=3)
    with pytest.raises(RuntimeError, match='changed since it was built'):
        mk.problem_size(model)
    assert mesh.getNumElem() == elements


def test_mesh_parameter(client):
    # a mesh built with a parameter that changed since
    model, geom, _ = plate_with_holes.build_model(client, 1)
    try:
        model.parameter('hm', '20')
        mk.set(model/'meshes'/'mesh'/'Size', custom=True, hmax='hm')
        (model/'meshes'/'mesh').create('FreeTet')
        model.mesh()
        assert mk.problem_size(model)['mesh_elements']['mesh'] > 0
        model.parameter('hm', '10')
        with pytest.raises(RuntimeError, match='changed since'):
            mk.problem_size(model)
    finally:
        client.remove(model)


def test_disabled_mesh_feature(plate):
    # a disabled feature stays unbuilt for good (a feature added by hand
    # removes the one COMSOL added: the tetrahedra are added too)
    model, geom = plate
    (model/'meshes'/'mesh').create('FreeTet')
    size = (model/'meshes'/'mesh').create('Size', name='spare')
    size.java.active(False)
    model.mesh()
    assert mk.problem_size(model)['mesh_elements']['mesh'] > 0


def test_unused_mesh(plate, tmp_path):
    # an empty mesh the study does not use, also with a sweep
    model, geom = plate
    spare = model.java.component('comp1').mesh().create('spare')
    model.parameter('p', '1')
    sweep = model.java.study('std1').create('param', 'Parametric')
    sweep.set('pname', ['p'])
    sweep.set('plistarr', ['1 2'])
    sweep.set('punit', [''])
    before = leaves_nothing(model, tmp_path)
    assert list(mk.problem_size(model)['mesh_elements']) == ['mesh']
    assert spare.isEmpty()
    assert leaves_nothing(model, tmp_path) == before


def test_second_component(plate):
    # COMSOL would build the empty mesh of another component silently
    model, geom = plate
    java = model.java
    component = java.component().create('comp2', True)
    other = component.geom().create('geom2', 3)
    other.create('blk1', 'Block')
    other.run()
    component.physics().create('ht2', 'HeatTransfer', 'geom2')
    mesh = component.mesh().create('mesh2')
    with pytest.raises(RuntimeError, match='is empty'):
        mk.problem_size(model)
    assert mesh.isEmpty()
    model.mesh()
    assert sorted(mk.problem_size(model)['mesh_elements']) == ['Mesh 2',
                                                              'mesh']


def test_no_mesh(model):
    # physics without a mesh: no degrees of freedom, an empty solution
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    model.build(geom)
    (model/'physics').create('HeatTransfer', geom)
    (model/'studies').create().create('Stationary')
    with pytest.raises(RuntimeError, match='has physics but no mesh'):
        mk.problem_size(model)


def test_changed_geometry(plate):
    model, geom = plate
    geom.java.feature('blk1').set('size', ['101', '40', '5'])
    with pytest.raises(RuntimeError, match='run model.build'):
        mk.problem_size(model)


def test_compile_error(plate, tmp_path):
    model, geom = plate
    # compiling catches syntax errors (not undefined names)
    (model/'physics'/'heat'/'hot end').property('T0', '1+*2')
    before = leaves_nothing(model, tmp_path)
    with pytest.raises(RuntimeError, match='could not compile'):
        mk.problem_size(model)
    assert leaves_nothing(model, tmp_path) == before


def test_two_steps(plate):
    model, geom = plate
    step = (model/'studies'/'static').create('Transient', name='later')
    step.property('tlist', 'range(0,10,50)')
    first, second = mk.problem_size(model)['steps']
    assert first['dofs'] > 0
    assert (second['step'], second['type']) == ('later', 'Transient')
    assert second['dofs'] is None and second['fields'] is None
    assert second['solver'] == 'direct'


def test_step_added_later(plate):
    # the solver sequence predates the second step: COMSOL's choice fills in
    model, geom = plate
    model.java.study('std1').createAutoSequences('all')
    step = (model/'studies'/'static').create('Transient', name='later')
    step.property('tlist', 'range(0,10,50)')
    first, second = mk.problem_size(model)['steps']
    assert (first['solver'], second['solver']) == ('direct', 'direct')


def test_solver_choices(plate):
    model, geom = plate
    java = model.java
    java.study('std1').createAutoSequences('all')
    solver = java.sol('sol1').feature('s1')
    solver.feature('i1').active(True)
    step = mk.problem_size(model)['steps'][0]
    assert (step['solver'], step['linear_solver']) == ('iterative', 'gmres')
    solver.feature().create('se1', 'Segregated')
    step = mk.problem_size(model)['steps'][0]
    assert (step['solver'], step['linear_solver']) == ('segregated', None)


def test_eigenfrequency(model):
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 0.1, 0.1))
    model.build(geom)
    mk.material(geom, 'Structural steel')
    solid = (model/'physics').create('SolidMechanics', geom)
    solid.create('Fixed', 2).select(mk.sel.box(geom, 'boundary', x=0))
    (model/'meshes').create(geom)
    model.mesh()
    (model/'studies').create().create('Eigenfrequency')
    step = mk.problem_size(model)['steps'][0]
    assert step['solver'] == 'direct'
    assert sum(step['fields'].values()) == step['dofs']


########
# Docs #
########

def test_documented(client, tmp_path, monkeypatch, logging):
    # the README's lines, on the example's model
    import re
    readme = (Path(__file__).parents[1]/'README.md').read_text()
    code = re.search(r'Before a long solve.*?```python\n(.*?)```', readme,
                     re.S).group(1).splitlines()
    assert len(code) == 3, 'update the checks below with the README'
    monkeypatch.chdir(tmp_path)
    model, geom, _ = plate_with_holes.build_model(client, 2)
    try:
        model.mesh()
        namespace = {'mk': mk, 'model': model}
        size, log, before = [eval(line.split('#')[0], namespace)
                             for line in code]
        assert log == tmp_path.resolve()/'solve.log'
        assert before['exists'] and before['pid'] == os.getpid()
        model.solve()
        after = eval(code[2].split('#')[0], namespace)
        assert after['percent'] == 100 and after['alive'] is True
        assert after['dofs'] == size['steps'][0]['dofs']
    finally:
        client.remove(model)
