"""Run four independent original-RDP pipelines in one queue per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import threading
import time

RDP_DIR = Path(__file__).resolve().parents[1]
DATASETS = {
    'insert': ['insert_01', 'insert_02'],
    'press': ['press_01', 'press_02'],
    'two_tubes': [f'two_tubes_0{i}' for i in range(1, 5)],
    'bread': ['bread_01', 'bread_02', 'bread_03'],
}


def commands(stage, task, gpu, environ, run_id):
    work = Path(environ.get('WORK_ROOT', '/DATA/ljl/substage/rdp_original'))
    dataset_root = environ.get('LEROBOT_ROOT', '/DATA/ljl/substage/lerobot_v21/KaiyueChen')
    cache = environ.get('TACTILE_CACHE_ROOT', str(work / 'tactile_embeddings_encoder0824'))
    data = work / 'datasets' / task
    pca = work / 'pca' / task / 'tactile_pca.npz'
    python = environ.get('PYTHON_BIN', str(RDP_DIR / '.venv/bin/python'))
    jax_default = RDP_DIR / '.venv-jax/bin/python'
    if not jax_default.is_file():
        jax_default = Path('/home/ljl/RDP_vitamin/.venv-jax/bin/python')
    encoder_default = Path('/DATA/ljl/substage/rdp_single_right/encoder_ckpt_0824')
    if not (encoder_default / 'checkpoint.json').is_file():
        encoder_default = Path('/home/ljl/RDP_vitamin/data/encoder_ckpt_0824')
    arms = environ.get(f'{task.upper()}_ARMS', 'right' if task in ('insert', 'press') else 'both')
    if arms not in ('right', 'both'):
        raise ValueError(f'{task.upper()}_ARMS must be right or both')
    workers = environ.get('NUM_WORKERS', '32')
    environment = {
        'CUDA_VISIBLE_DEVICES': gpu,
        'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': environ.get('OMP_NUM_THREADS', '4'),
        'MKL_NUM_THREADS': environ.get('MKL_NUM_THREADS', '4'),
    }
    result = []
    if stage in ('all', 'prepare', 'precompute'):
        env = dict(environment, XLA_PYTHON_CLIENT_PREALLOCATE='false')
        cmd = [environ.get('JAX_PYTHON', str(jax_default)),
               str(RDP_DIR / 'precompute_pick_tube_v21_tactile_embeddings.py'),
               '--dataset-root', dataset_root, '--cache-root', cache,
               '--encoder-path', environ.get('ENCODER_DIR', str(encoder_default)),
               '--batch-size', environ.get('PRECOMPUTE_BATCH', '512'),
               '--num-workers', environ.get('PRECOMPUTE_WORKERS', workers),
               '--datasets', *DATASETS[task]]
        result.append((cmd, env, ['LD_LIBRARY_PATH']))
    if stage in ('all', 'prepare'):
        if not pca.is_file():
            result.append(([python, str(RDP_DIR / 'fit_pick_tube_tactile_pca.py'),
                            '--tactile-cache-root', cache, '--output', str(pca),
                            '--components-per-arm', '15', '--datasets', *DATASETS[task]], environment, []))
        result.append(([python, str(RDP_DIR / 'prepare_baseline_data.py'),
                        '--dataset-root', dataset_root, '--datasets', *DATASETS[task],
                        '--tactile-cache-root', cache, '--output-dir', str(data),
                        '--arms', arms, '--num-workers', environ.get('CONVERT_WORKERS', workers)], environment, []))
    if stage in ('all', 'train', 'at', 'ldp'):
        env = dict(environment, PYTHON_BIN=python, DATASET_PATH=str(data / 'replay_buffer.zarr'),
                   TACTILE_CACHE_PATH=str(data / 'raw_tactile_manifest.json'),
                   TACTILE_PCA_PATH=str(pca), OUTPUT_ROOT=str(work / 'outputs' / task), RUN_ID=run_id,
                   NUM_WORKERS=workers, DEVICE='cuda:0', MIXED_PRECISION=environ.get('MIXED_PRECISION', 'bf16'),
                   AT_BATCH=environ.get('AT_BATCH', '64'), LDP_BATCH=environ.get('LDP_BATCH', '64'),
                   AT_EPOCHS=environ.get('AT_EPOCHS', '60'), LDP_EPOCHS=environ.get('LDP_EPOCHS', '40'),
                   WANDB_MODE=environ.get('WANDB_MODE', 'offline'),
                   AT_CKPT=str(work / 'outputs' / task / run_id / 'at/checkpoints/latest.ckpt'))
        for training_stage in (('at', 'ldp') if stage in ('all', 'train') else (stage,)):
            cmd = ['bash', str(RDP_DIR / 'scripts/train_rdp_baseline.sh'), training_stage,
                   'task=single_right' if arms == 'right' else 'task=dual_arm', f'task.name={task}',
                   f'training.resume={environ.get("RESUME", "false")}',
                   f'training.checkpoint_every={environ.get("CHECKPOINT_EVERY", "10")}']
            result.append((cmd, env, []))
    return result


def format_command(command):
    argv, env, unset = command
    return shlex.join(['env', *[v for key in unset for v in ('-u', key)],
                       *[f'{key}={value}' for key, value in env.items()], *argv])


def command_stage(command):
    argv = command[0]
    for argument in argv:
        name = Path(argument).name
        if name == 'precompute_pick_tube_v21_tactile_embeddings.py':
            return 'tactile'
        if name == 'fit_pick_tube_tactile_pca.py':
            return 'PCA'
        if name == 'prepare_baseline_data.py':
            return 'convert'
        if name == 'train_rdp_baseline.sh':
            return argv[argv.index(argument) + 1].upper()
    return 'process'


def stream_process(process, log, prefix, emit, heartbeat_seconds=30):
    """Tee stdout/stderr, including CR progress bars, while reporting silence."""
    lines = queue.Queue(maxsize=256)
    started = time.monotonic()

    def reader():
        try:
            # Popen text mode translates tqdm's carriage returns to newlines.
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    while True:
        try:
            line = lines.get(timeout=heartbeat_seconds)
        except queue.Empty:
            message = f'RUNNING elapsed={time.monotonic()-started:.0f}s; waiting for subprocess output'
        else:
            if line is None:
                break
            message = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
            if not message:
                continue
        formatted = f'[{datetime.now():%H:%M:%S}]{prefix} {message}'
        log.write(formatted + '\n')
        log.flush()
        emit(formatted)
    thread.join()
    process.stdout.close()
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'prepare', 'precompute', 'train', 'at', 'ldp'])
    parser.add_argument('tasks', nargs='*', choices=None, help='Default: insert press two_tubes bread')
    args = parser.parse_args()
    tasks = args.tasks or list(DATASETS)
    if len(set(tasks)) != len(tasks) or any(task not in DATASETS for task in tasks):
        parser.error('choose distinct tasks from insert, press, two_tubes, bread')
    env = dict(os.environ)
    gpus = [gpu.strip() for gpu in env.get('GPU_IDS', '0,1').split(',')]
    if not gpus or any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU_IDS must contain distinct comma-separated GPU indices')
    run_id = env.get('RUN_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
    heartbeat = float(env.get('LOG_HEARTBEAT_SECONDS', '30'))
    if heartbeat <= 0:
        parser.error('LOG_HEARTBEAT_SECONDS must be positive')
    lanes = [(gpu, tasks[index::len(gpus)]) for index, gpu in enumerate(gpus)]
    print(f'RDP baseline run: {run_id}', flush=True)
    for gpu, selected in lanes:
        print(f'GPU {gpu}: {" -> ".join(selected)}', flush=True)
    print(f'AT: epochs={env.get("AT_EPOCHS", "60")} batch={env.get("AT_BATCH", "64")} FP32; '
          f'LDP: epochs={env.get("LDP_EPOCHS", "40")} batch={env.get("LDP_BATCH", "64")} '
          f'precision={env.get("MIXED_PRECISION", "bf16")}; workers={env.get("NUM_WORKERS", "32")}', flush=True)
    if env.get('DRY_RUN') == '1':
        for gpu, selected in lanes:
            for task in selected:
                for cmd in commands(args.stage, task, gpu, env, run_id):
                    print(format_command(cmd))
        return

    work = Path(env.get('WORK_ROOT', '/DATA/ljl/substage/rdp_original'))
    active = set()
    lock = threading.Lock()
    console_lock = threading.Lock()
    stop = threading.Event()

    def emit(message):
        with console_lock:
            print(message, flush=True)

    def run_lane(gpu, selected):
        for task in selected:
            if stop.is_set():
                return
            output = work / 'outputs' / task / run_id
            output.mkdir(parents=True, exist_ok=True)
            plan = commands(args.stage, task, gpu, env, run_id)
            (output / 'pipeline.json').write_text(json.dumps({
                'task': task, 'datasets': DATASETS[task], 'gpu': gpu, 'run_id': run_id,
                'commands': [format_command(cmd) for cmd in plan]}, indent=2) + '\n')
            emit(f'[{task}] GPU {gpu}; datasets={",".join(DATASETS[task])}; log: {output / "pipeline.log"}')
            with (output / 'pipeline.log').open('a', buffering=1) as log:
                for index, cmd in enumerate(plan, 1):
                    if stop.is_set():
                        return
                    argv, overrides, unset = cmd
                    child_env = dict(env, **overrides)
                    for key in unset:
                        child_env.pop(key, None)
                    log.write(f'\n{datetime.now().isoformat()} {format_command(cmd)}\n')
                    prefix = f'[GPU {gpu}][{task}][{command_stage(cmd)} {index}/{len(plan)}]'
                    started = time.monotonic()
                    message = f'[{datetime.now():%H:%M:%S}]{prefix} START'
                    log.write(message + '\n')
                    emit(message)
                    with lock:
                        if stop.is_set():
                            return
                        process = subprocess.Popen(argv, env=child_env, cwd=RDP_DIR,
                                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, encoding='utf-8', errors='replace', bufsize=1,
                                                   start_new_session=True)
                        active.add(process)
                    result = stream_process(process, log, prefix, emit, heartbeat)
                    with lock:
                        active.discard(process)
                    status = 'DONE' if result == 0 else 'FAILED'
                    message = f'[{datetime.now():%H:%M:%S}]{prefix} {status} elapsed={time.monotonic()-started:.1f}s exit={result}'
                    log.write(message + '\n')
                    emit(message)
                    if result:
                        raise RuntimeError(f'{task} failed (exit {result}); see {output / "pipeline.log"}')
            emit(f'[{task}] completed')

    pool = ThreadPoolExecutor(max_workers=len(gpus))
    failed = False
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        futures = [pool.submit(run_lane, gpu, selected) for gpu, selected in lanes if selected]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failed = True
                print(str(error), flush=True)
    except KeyboardInterrupt:
        stop.set()
        with lock:
            for process in active:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit(130)
    finally:
        pool.shutdown(wait=True)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
