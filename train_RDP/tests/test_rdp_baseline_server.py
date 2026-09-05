"""The server launcher keeps four tasks and two GPU queues independent."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import io

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_four_task_dry_run_uses_exact_sources_and_isolated_baseline(tmp_path):
    env = dict(os.environ, DRY_RUN='1', WORK_ROOT=str(tmp_path / 'work'),
               RUN_ID='server_test', PYTHON_BIN='/usr/bin/python3')
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/server_ljl_baseline_four_tasks.sh'), 'all'],
        cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for task, names, gpu in [
        ('insert', ['insert_01', 'insert_02'], '0'),
        ('press', ['press_01', 'press_02'], '1'),
        ('two_tubes', [f'two_tubes_0{i}' for i in range(1, 5)], '0'),
        ('bread', ['bread_01', 'bread_02', 'bread_03'], '1'),
    ]:
        precompute = next(line for line in lines if 'precompute_pick_tube_v21' in line and f'--datasets {names[0]}' in line)
        assert all(name in precompute for name in names)
        assert f'CUDA_VISIBLE_DEVICES={gpu}' in precompute
        training = next(line for line in lines if 'train_rdp_baseline.sh' in line and f'task.name={task}' in line)
        assert 'NUM_WORKERS=32' in training
        assert f'/{task}/raw_tactile_manifest.json' in training
        assert f'/{task}/tactile_pca.npz' in training
        assert f'/{task}' in training
    assert 'convert_pick_tube_lerobot_to_rdp_zarr.py' not in result.stdout
    assert 'deployable' not in result.stdout
    assert not (tmp_path / 'work').exists()


def test_task_selection_only_schedules_requested_task(tmp_path):
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/server_ljl_baseline_four_tasks.sh'), 'train', 'bread'],
        env=dict(os.environ, DRY_RUN='1', GPU_IDS='1', PYTHON_BIN='/usr/bin/python3'),
        cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'task.name=bread' in result.stdout
    assert 'CUDA_VISIBLE_DEVICES=1' in result.stdout
    assert 'task.name=insert' not in result.stdout
    assert 'precompute_pick_tube_v21' not in result.stdout


def test_failed_task_stops_its_queue_while_other_gpu_finishes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('baseline_server', ROOT / 'rdp_baseline/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fake_commands(stage, task, gpu, environ, run_id):
        program = ('import pathlib,sys; '
                   'pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); '
                   'sys.exit(7 if sys.argv[3] == "insert" else 0)')
        return [([sys.executable, '-c', program, str(tmp_path / task), gpu, task], {}, [])]
    monkeypatch.setattr(module, 'commands', fake_commands)
    monkeypatch.setenv('WORK_ROOT', str(tmp_path / 'work'))
    monkeypatch.setenv('GPU_IDS', '0,1')
    monkeypatch.delenv('DRY_RUN', raising=False)
    monkeypatch.setattr(sys, 'argv', ['server.py', 'train'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    assert (tmp_path / 'insert').read_text() == '0'
    assert not (tmp_path / 'two_tubes').exists()
    assert (tmp_path / 'press').read_text() == '1'
    assert (tmp_path / 'bread').read_text() == '1'


def test_live_output_copies_progress_stderr_and_reports_silent_process():
    from rdp_baseline.server import stream_process
    program = ('import sys,time; '
               'print("encoder ready", flush=True); '
               'sys.stderr.write("progress 25%\\rprogress 50%\\r"); sys.stderr.flush(); '
               'time.sleep(0.15); print("finished", flush=True)')
    events = []
    log = io.StringIO()
    with subprocess.Popen([sys.executable, '-u', '-c', program], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        assert stream_process(process, log, '[GPU 0][insert][tactile]',
                              events.append, heartbeat_seconds=.04) == 0
    merged = '\n'.join(events)
    for value in ['encoder ready', 'progress 25%', 'progress 50%', 'finished']:
        assert value in merged and value in log.getvalue()
    assert 'RUNNING' in merged
    assert all('[GPU 0][insert][tactile]' in event for event in events)


def test_at_and_ldp_have_separate_progress_stages(tmp_path):
    from rdp_baseline.server import commands, command_stage
    plan = commands('train', 'insert', '0', {'WORK_ROOT':str(tmp_path)}, 'test')
    assert [command_stage(item) for item in plan] == ['AT', 'LDP']
    assert all(item[1]['PYTHONUNBUFFERED'] == '1' for item in plan)
