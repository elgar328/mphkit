"""
Checks the guidance for guessed names and the documentation's references.

Runs without COMSOL, except the tests that execute the documented examples.
"""
import inspect
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mphkit as mk
from conftest import read
from mphkit import _hints

root = Path(__file__).parents[1]
readme = read(root/'README.md')


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
    (mk.sel, 'all_boundaries', "Did you mean mphkit.sel.all(geom, 'boundary')?"),
    (mk.sel, 'box_faces', "Did you mean mphkit.sel.box(geom, 'boundary', ...)?"),
    (mk.sel, 'adjacent_boundaries',
     "Did you mean mphkit.sel.adjacent(geom, input, 'boundary')?"),
    (mk, 'result_faces', "Did you mean mphkit.sel.result(geom, feature, 'boundary')?"),
    (mk.sel, 'cumulative_edges', "Did you mean mphkit.sel.cumulative(geom, group, 'edge')?"),
    (mk.sel, 'entities_domains', 'Did you mean mphkit.sel.entities?'),
    (mk.sel, 'layer_domains', 'Did you mean mphkit.sel.layer?'),
    # a module prefix, as in `sel_box`
    (mk, 'sel_box', 'Did you mean mphkit.sel.box (select)'),
    (mk.sel, 'sel_box', 'Did you mean mphkit.sel.box (select)'),
    (mk, 'sel_faces', "mphkit.sel.box(geom, 'boundary', ...)"),
    (mk, 'sel_xyzzy', 'Did you mean mphkit.sel?'),
    (mk, 'sel_blok', 'Did you mean mphkit.sel?'),
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
    (mk, 'picture', 'Did you mean mphkit.image? Pictures of the geometry'),
    (mk, 'mphplot', 'Did you mean mphkit.plot?'),
    (mk, 'slice', 'Did you mean mphkit.plot?'),
    (mk, 'surface_plot', 'Did you mean mphkit.plot?'),
    (mk, 'plot_temperature', 'Did you mean mphkit.plot?'),
    (mk, 'result_plot', 'Did you mean mphkit.plot?'),
    (mk, 'mphmesh', 'Did you mean mphkit.image?'),
    (mk, 'plot_mesh', 'Did you mean mphkit.image?'),
    (mk, 'mesh_plot', 'Did you mean mphkit.image?'),
    (mk, 'plot_mesh_quality', 'Pass mesh=True for a picture of the mesh.'),
    (mk, 'plot_mesh_quality', 'mphkit.mesh_quality gives the numbers.'),
    (mk, 'show_mesh', 'Did you mean mphkit.image?'),
    (mk, 'mesh_stats', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'quality_of_mesh', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'get_mesh_quality', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'check_mesh_quality', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'mesh_skewness', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'mphmeshstats', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'meshstat', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'min_quality', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'quality_histogram', 'Did you mean mphkit.mesh_quality?'),
    (mk, 'element_count', 'Did you mean mphkit.mesh_quality?'),
    (mk.sel, 'mesh_image', 'Did you mean mphkit.image?'),
    (mk, 'info', 'Did you mean mphkit.summary?'),
    (mk, 'coords', 'Did you mean mphkit.coordinates?'),
    (mk, 'adjacency', 'mphkit.sel.adjacent makes a selection instead'),
    (mk.sel, 'adjacency', 'mphkit.sel.adjacent makes a selection instead'),
    # the names of COMSOL's LiveLink for MATLAB
    (mk, 'mphgeominfo', 'Did you mean mphkit.summary?'),
    (mk, 'mphgetadj', 'Did you mean mphkit.sel.neighbors?'),
    (mk, 'mphgetcoords', 'Did you mean mphkit.coordinates?'),
    (mk, 'mphviewselection', 'Did you mean mphkit.image?'),
    (mk, 'mphint2', 'Did you mean mphkit.integral?'),
    (mk, 'mphinterp', 'Did you mean mphkit.value?'),
    # results: a last word for a result wins, max and min count first only
    (mk, 'volume_integral', 'Did you mean mphkit.integral?'),
    (mk, 'line_integral', 'Did you mean mphkit.integral?'),
    (mk, 'surface_average', 'Did you mean mphkit.average?'),
    (mk, 'point_values', 'Did you mean mphkit.value?'),
    (mk, 'max_temperature', 'Did you mean mphkit.maximum?'),
    (mk, 'bbox_max', 'Did you mean mphkit.bounding_box?'),
    (mk, 'max_value', 'Did you mean mphkit.maximum?'),
    (mk, 'mean_value', 'Did you mean mphkit.average?'),
    (mk, 'mean', 'Did you mean mphkit.average?'),
    (mk, 'probe', 'Did you mean mphkit.value?'),
    (mk, 'evaluate_at', 'Did you mean mphkit.value?'),
    (mk, 'evaluate', "MPh's model.evaluate(expression, unit) gives global"),
    (mk.sel, 'eval', "MPh's model.evaluate(expression, unit)"),
    (mk, 'translate', 'Did you mean mphkit.move?'),
    (mk, 'cone', "Did you mean mphkit.feature(geom, 'Cone', ...)?"),
    (mk, 'bounding', 'Did you mean mphkit.bounding_box?'),
    (mk, 'fillet_edges', 'Did you mean mphkit.fillet?'),
    # COMSOL's names
    (mk, 'features', 'Did you mean mphkit.feature_types or mphkit.feature?'),
    (mk, 'feature_type', 'Did you mean mphkit.feature_types?'),
    (mk, 'get_feature_types', 'Did you mean mphkit.feature_types?'),
    (mk, 'list_features', 'Did you mean mphkit.feature_types?'),
    (mk, 'feature_list', 'Did you mean mphkit.feature_types?'),
    (mk, 'physics_features', 'Did you mean mphkit.feature_types?'),
    (mk, 'feature_properties', 'Did you mean mphkit.properties?'),
    (mk, 'property_values', 'Did you mean mphkit.properties?'),
    (mk, 'get_properties', 'Did you mean mphkit.properties?'),
    (mk, 'list_properties', 'Did you mean mphkit.properties?'),
    (mk, 'get_props', 'Did you mean mphkit.properties?'),
    (mk, 'props', 'Did you mean mphkit.properties?'),
    (mk, 'settings', 'Did you mean mphkit.properties or mphkit.set?'),
    (mk, 'options', 'Did you mean mphkit.properties or mphkit.set?'),
    (mk.sel, 'settings', 'Did you mean mphkit.properties or mphkit.set?'),
    (mk, 'list_variables', 'Did you mean mphkit.variables?'),
    (mk, 'vars', 'Did you mean mphkit.variables?'),
    (mk, 'get_variables', 'Did you mean mphkit.variables?'),
    (mk, 'find_variable', 'Did you mean mphkit.variables?'),
    (mk, 'expressions', 'Did you mean mphkit.variables?'),
    (mk, 'physics', 'Did you mean mphkit.physics_types?'),
    (mk, 'physics_type', 'Did you mean mphkit.physics_types?'),
    (mk, 'interfaces', 'Did you mean mphkit.physics_types?'),
    (mk, 'list_physics', 'Did you mean mphkit.physics_types?'),
    (mk, 'physics_list', 'Did you mean mphkit.physics_types?'),
    (mk, 'physics_interfaces', 'Did you mean mphkit.physics_types?'),
    (mk, 'physics_feature', 'Did you mean mphkit.feature_types?'),
    # the check before solving
    (mk, 'validate', 'Did you mean mphkit.check?'),
    (mk, 'diagnose', 'Did you mean mphkit.check?'),
    (mk, 'precheck', 'Did you mean mphkit.check?'),
    (mk, 'check_model', 'Did you mean mphkit.check?'),
    (mk, 'lint', 'Did you mean mphkit.check?'),
    (mk, 'verify', 'Did you mean mphkit.check?'),
    (mk, 'sanity_check', 'Did you mean mphkit.check?'),
    # materials
    (mk, 'matlib', 'Did you mean mphkit.materials or mphkit.material?'),
    (mk, 'material_library',
     'Did you mean mphkit.materials or mphkit.material?'),
    (mk, 'library_materials', 'Did you mean mphkit.materials?'),
    (mk, 'list_materials', 'Did you mean mphkit.materials?'),
    (mk, 'get_materials', 'Did you mean mphkit.materials?'),
    (mk, 'insert_material', 'Did you mean mphkit.material?'),
    (mk, 'add_material', 'Did you mean mphkit.material?'),
    (mk, 'mesh_types', 'Did you mean mphkit.feature_types?'),
    (mk, 'study_types', 'Did you mean mphkit.feature_types?'),
    (mk, 'mesh_features', 'Did you mean mphkit.feature_types?'),
    (mk, 'study_steps', 'Did you mean mphkit.feature_types?'),
    (mk, 'types', 'Did you mean mphkit.feature_types?'),
    (mk, 'list_types', 'Did you mean mphkit.feature_types?'),
    (mk, 'geometry_features',
     'Did you mean mphkit.feature_types or mphkit.feature?'),
    (mk, 'geometry_types',
     'Did you mean mphkit.feature_types or mphkit.feature?'),
    (mk, 'geom_types', 'Did you mean mphkit.feature_types or mphkit.feature?'),
    (mk, 'outer_solutions', 'Did you mean mphkit.outer_values?'),
    (mk, 'outer_parameters', 'Did you mean mphkit.outer_values?'),
    (mk, 'sweep_values', 'Did you mean mphkit.outer_values?'),
    (mk, 'mphsolinfo', 'Did you mean mphkit.outer_values?'),
    (mk, 'parameter_values',
     'Did you mean mphkit.outer_values or mphkit.step_values?'),
    (mk, 'param_values',
     'Did you mean mphkit.outer_values or mphkit.step_values?'),
    (mk, 'time_values', 'Did you mean mphkit.step_values?'),
    (mk, 'get_time_values', 'Did you mean mphkit.step_values?'),
    (mk, 'time_steps', 'Did you mean mphkit.step_values?'),
    (mk, 'timesteps', 'Did you mean mphkit.step_values?'),
    (mk, 'frequencies', 'Did you mean mphkit.step_values?'),
    (mk, 'eigenvalues', 'Did you mean mphkit.step_values?'),
    (mk, 'eigen_values', 'Did you mean mphkit.step_values?'),
    (mk, 'eigenfrequencies', 'Did you mean mphkit.step_values?'),
    (mk, 'frequency_values', 'Did you mean mphkit.step_values?'),
    (mk, 'freq_values', 'Did you mean mphkit.step_values?'),
    (mk, 'sweep_steps', 'Did you mean mphkit.step_values?'),
    (mk, 'outer_value', 'Did you mean mphkit.outer_values?'),
    (mk, 'step_value', 'Did you mean mphkit.step_values?'),
])
def test_suggestion(module, name, expected):
    text = message(module, name)
    assert text.startswith(f"module '{module.__name__}' has no attribute {name!r}.")
    assert expected in text
    assert text.endswith(f'See help({module.__name__}) for all helpers.')


@pytest.mark.parametrize('module, name', [
    (mk.sel, 'inner'), (mk.sel, 'outer'), (mk, 'inner_boundaries'),
    (mk, 'inner')])
def test_no_sweep_suggestion(module, name):
    # inner and outer boundaries are no sweeps
    assert 'values' not in message(module, name)


@pytest.mark.parametrize('name, expected', [
    ('outer', 'outer= is an argument of the results helpers'),
    ('sweep', "Did you mean mphkit.feature(geom, 'Sweep', ...) (a geometry "
              "sweep)? A parametric sweep is plain MPh"),
    ('parametric_sweep', "sweep = study.create('Parametric')"),
    ('param_sweep', 'Read it with outer= or step='),
    ('set_parameter', "model.parameter('L', '0.1[m]')"),
    ('parameters', "model.parameter("),
    ('datasets', 'take dataset= (a dataset or its study)'),
    ('solve', "model.solve('s')"),
    ('mesh', "(model/'meshes').create(geom)"),
    ('create_study', "study.create('Stationary')"),
    ('results', 'The results helpers are mphkit.integral'),
    ('flux', "Did you mean mphkit.integral? e.g. mk.integral(geom, "
             "'boundary', 'ht.ntflux'"),
    ('animate', 'mphkit.plot draws one step per call'),
    ('outer_sweep_values', 'Did you mean mphkit.outer_values?'),
    ('steps_values', 'Did you mean mphkit.step_values?'),
    ('steps', 'Did you mean mphkit.step_values?'),
    ('times', 'Did you mean mphkit.step_values?'),
    ('maxval', 'Did you mean mphkit.maximum?'),
    ('minval', 'Did you mean mphkit.minimum?')])
def test_guessed_notes(name, expected):
    assert expected in message(mk, name)


def test_sel_result_stays():
    assert mk.sel.result is not None
    assert 'results helpers' not in message(mk.sel, 'results')


def test_meanings_name_helpers():
    for key in _hints.MEANINGS:
        for meant in _hints._meant(key):
            target = mk
            for part in meant.split('.')[1:]:
                target = getattr(target, part)
    assert 'Did you mean' not in message(mk, 'evaluate')


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


def test_suggested_calls_match_signatures():
    # the call shapes the hints suggest follow the helpers' own arguments
    kinds = (inspect.Parameter.POSITIONAL_ONLY,
             inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def positional(name):
        parameters = inspect.signature(getattr(mk.sel, name)).parameters.values()
        return [p.name for p in parameters if p.kind in kinds]

    assert set(_hints.SEL_CALLS) <= set(mk.sel.__all__)
    for name in mk.sel.__all__:
        call = _hints.SEL_CALLS.get(name, _hints.SEL_CALL)
        shown = call.strip('()').replace('{kind}', 'entity').split(', ')
        if shown[-1] == '...':
            shown.pop()
            assert positional(name)[:len(shown)] == shown, name
        else:
            assert positional(name) == shown, name


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


def run_example(code, client, monkeypatch, tmp_path, check=None):
    """
    Runs documented example code with the test session's client, and
    `check` on its namespace before the models are removed.
    """
    import mph
    monkeypatch.setattr(mph, 'start', lambda *args, **kwargs: client)
    code = code.replace("'demo.mph'", repr(str(tmp_path/'demo.mph')))
    before = client.models()
    try:
        namespace: dict = {}
        exec(code, namespace)
        if check:
            check(namespace)
    finally:
        for model in client.models():
            if model not in before:
                client.remove(model)


def test_readme_example(client, monkeypatch, tmp_path):
    code = re.search(r'## Example\n\n```python\n(.*?)```', readme, re.S).group(1)
    queries = re.search(r'Queries on the plate.*?```python\n(.*?)```',
                        readme, re.S).group(1).splitlines()
    assert len(queries) == 7, 'update the checks below with the README'

    def check(namespace):
        # the queries in the README, with the values their comments show
        (entities, found, volume, bbox, summary, neighbors, vertices) = [
            eval(line.split('#')[0], namespace) for line in queries]
        assert entities == [3]
        assert found == [1]
        assert volume == pytest.approx(100*100*10 - math.pi*5**2*10, rel=1e-4)
        assert bbox['x'] == pytest.approx((0, 100))
        assert bbox['y'] == pytest.approx((0, 100))
        assert bbox['z'] == pytest.approx((0, 0))
        assert summary['domains'] == 1
        assert neighbors == [1]
        assert vertices[1] == (0, 0, 0) and vertices[3] == (0, 100, 0)

    run_example(code, client, monkeypatch, tmp_path, check)
    assert (tmp_path/'demo.mph').exists()


def test_readme_results(client, tmp_path):
    # the results in the README, on the example script it links to
    from test_example import plate_with_holes
    lines = re.search(r'Results of the solved.*?```python\n(.*?)```', readme,
                      re.S).group(1).splitlines()
    pictures = [re.sub(r"'(\w+\.png)'", lambda m: repr(str(tmp_path/m[1])),
                       line.split('#')[0])
                for line in re.search(r'Pictures, written to a file:\n\n'
                                      r'```python\n(.*?)```', readme,
                                      re.S).group(1).splitlines()
                if line.startswith('mk.plot') or 'mesh=True' in line]
    assert len(pictures) == 3
    assert len(lines) == 4, 'update the checks below with the README'
    model, geom, selections = plate_with_holes.build_model(client, 2)
    try:
        model.solve()
        namespace = {'mk': mk, 'geom': geom, 'selections': selections}
        heat, mean, (low, where), points = [
            eval(line.split('#')[0], namespace) for line in lines]
        assert heat == pytest.approx(-2.74, abs=0.005)
        assert mean == pytest.approx(90.2, abs=0.05)
        assert low == pytest.approx(83.4, abs=0.05)
        assert where[:2] == pytest.approx([88, 20])   # any z: T is flat in z
        assert points == pytest.approx([89.8, 84.1], abs=0.05)
        for line in pictures:
            assert eval(line, namespace).exists()
    finally:
        client.remove(model)


def test_help_example(client, monkeypatch, tmp_path):
    lines = [line[4:] for line in mk.__doc__.splitlines()
             if line.startswith('    ') or not line.strip()]
    joined = '\n'.join(lines)
    code = re.search(r'import mph\n.*?\.select\(bottom\)[^\n]*\n',
                     joined, re.S).group(0)
    # the name lookups, on the example's model
    lookups = re.search(r'(mk\.physics_types\(geom.*?)\n\n', joined,
                        re.S).group(1).splitlines()
    assert len(lookups) == 6, 'update the checks below with the help'

    def check(namespace):
        found = [eval(line.split('#')[0], namespace) for line in lookups]
        assert 'T0' in found[4]
        assert any(v['name'] == 'ht.ntflux' for v in found[5])

    run_example(code, client, monkeypatch, tmp_path, check)


def test_help_sweep_example(client, monkeypatch, tmp_path):
    # the plain MPh lines up to the solve, then the sweep lines
    lines = [line[4:] for line in mk.__doc__.splitlines()
             if line.startswith('    ') or not line.strip()]
    joined = '\n'.join(lines)
    code = re.search(r"import mph\n.*?model\.solve\('heating'\)\n", joined,
                     re.S).group(0)
    reading = re.search(r"(mk\.average\(geom, 'domain', 'T', unit='degC', "
                        r"outer='all'.*?\n)\n", joined, re.S).group(1)
    reading = reading.replace("'T_{outer}.png'",
                              repr(str(tmp_path/'T_{outer}.png')))

    def check(namespace):
        exec(reading, namespace)
        table = namespace['table']
        assert [row['Th'] for row in table] == \
            pytest.approx([373.15, 473.15, 573.15])
        assert all('Tmax' in row for row in table)
    run_example(code, client, monkeypatch, tmp_path, check)
    assert len(list(tmp_path.glob('T_*.png'))) == 3


def test_help_java_export(client, model, tmp_path):
    # the route from an existing model that help(mphkit) describes
    geom = mk.geometry(model, 3)
    mk.block(geom, (1, 1, 1))
    mk.cylinder(geom, 0.1, 1, (2, 0, 0)).remove()   # history to compact
    model.build(geom)
    bottom = mk.sel.find(geom, 'boundary', z=0)
    assert len(bottom) == 1
    heat = (model/'physics').create('HeatTransfer', geom)
    temp = heat.create('TemperatureBoundary', 2)
    temp.select(bottom)                              # a numbered selection
    model.save(tmp_path/'demo.mph')
    saved = (tmp_path/'demo.mph').read_bytes()
    old = client.load(tmp_path/'demo.mph')
    try:
        g = (old/'geometries').children()[0]
        assert mk.sel.find(g, 'boundary', z=0) == bottom   # loaded, built
        old.save(tmp_path/'before.java')
        old.reset()
        old.save(tmp_path/'old.java')
        before = read(tmp_path/'before.java')
        text = read(tmp_path/'old.java')
        assert '"Cylinder"' in before, 'no history to compact'
        assert '"Block"' in text and '"Cylinder"' not in text
        assert f'feature("{temp.tag()}").selection().set({bottom[0]})' in text
        # the documented round trip: look up in old, select in the rebuild
        box = mk.bounding_box(g, 'boundary', bottom[0])
        assert mk.sel.find(g, 'boundary', **box) == bottom
        rebuilt = mk.sel.box(geom, 'boundary', **box)
        assert mk.sel.entities(geom, rebuilt) == bottom
        assert (tmp_path/'demo.mph').read_bytes() == saved
    finally:
        client.remove(old)
