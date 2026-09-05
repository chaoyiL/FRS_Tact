#!/usr/bin/env python3
"""Prepare raw LeRobot episodes for the original RDP chunk-relative adapter.

Only data readers are reused from the existing converter. No canonical action,
idle threshold, repeat weighting, PCA projection, or training code is executed.
Raw tactile arrays remain in their source caches and are referenced in order.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc

from convert_pick_tube_lerobot_to_rdp_zarr import (
    decode_images, extract_float32_matrix, load_episode_lengths, parquet_path,
)


def _stamp(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def prepare_data(*, dataset_root, datasets, tactile_cache_root, output_dir,
                 arms='right', num_workers=0, max_episodes_per_dataset=None,
                 overwrite=False):
    dataset_root, tactile_cache_root, output_dir = map(Path, (dataset_root, tactile_cache_root, output_dir))
    if arms not in ('right', 'both'):
        raise ValueError('arms must be right or both')
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError('datasets must be nonempty and unique')
    if num_workers < 0 or (max_episodes_per_dataset is not None and max_episodes_per_dataset < 1):
        raise ValueError('num-workers must be nonnegative; episode limit must be positive')
    sources, episodes, shards = [], [], []
    total_frames = 0
    for dataset in datasets:
        dataset_dir = dataset_root / dataset
        records, offsets = load_episode_lengths(dataset_dir)
        selected = records[:max_episodes_per_dataset]
        if not selected or any(int(item['length']) < 1 for item in selected):
            raise ValueError(f'{dataset}: expected nonempty episodes')
        cache_path = tactile_cache_root / 'KaiyueChen' / dataset / 'embeddings.npy'
        cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        cache_metadata_path = cache_path.with_name('metadata.json')
        if not cache_metadata_path.is_file():
            raise ValueError(f'{dataset}: raw tactile cache has no completion metadata: {cache_metadata_path}')
        cache_metadata = json.loads(cache_metadata_path.read_text())
        count = sum(int(item['length']) for item in selected)
        if cache.ndim != 3 or cache.shape[1:] != (4, 512) or len(cache) < count:
            raise ValueError(f'{dataset}: expected aligned [N,4,512] raw tactile cache, got {cache.shape}')
        if int(cache_metadata['total_frames']) != len(cache):
            raise ValueError(f'{dataset}: raw tactile completion metadata/frame count mismatch')
        source_episodes = []
        for item in selected:
            episode_index, length = int(item['episode_index']), int(item['length'])
            source_episodes.append({'episode_index': episode_index, 'length': length,
                                    'cache_start': offsets[episode_index],
                                    'parquet': _stamp(parquet_path(dataset_dir, episode_index))})
            episodes.append((dataset_dir, episode_index, length))
        sources.append({'dataset': dataset, 'root': str(dataset_dir.resolve()),
                        'cache': _stamp(cache_path), 'cache_metadata': _stamp(cache_metadata_path),
                        'episodes': source_episodes})
        shards.append({'dataset': dataset, 'path': str(cache_path.resolve()), 'start': 0, 'stop': count,
                       'episode_indices': [int(item['episode_index']) for item in selected],
                       'episode_lengths': [int(item['length']) for item in selected]})
        total_frames += count
    manifest = {'version': 1, 'arms': arms, 'total_frames': total_frames, 'sources': sources}
    marker = output_dir / 'prepare_manifest.json'
    generated = ('replay_buffer.zarr', 'raw_tactile_manifest.json', 'prepare_manifest.json')
    if marker.is_file() and not overwrite:
        if json.loads(marker.read_text()) != manifest:
            raise ValueError(f'{output_dir}: existing baseline data has different inputs; choose another output or --overwrite')
        if all((output_dir / name).exists() for name in generated):
            print(f'Baseline data already prepared: {output_dir}', flush=True)
            return manifest
    if not overwrite and any((output_dir / name).exists() for name in generated):
        raise ValueError(f'{output_dir}: incomplete baseline preparation; use --overwrite')
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.baseline-prepare-', dir=output_dir))
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers else None
    try:
        replay = zarr.open_group(str(temporary / 'replay_buffer.zarr'), mode='w')
        replay.attrs.update({'baseline_raw_version': 1, 'arms': arms,
                             'action_source': 'actions (unmodified local increments)',
                             'source_datasets': list(datasets)})
        data = replay.create_group('data')
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        state_dim, action_dim = (20, 20) if arms == 'both' else (7, 10)
        arrays = {
            key: data.create_dataset(key, shape=(total_frames, width), chunks=(min(total_frames, 4096), width),
                                     dtype='f4', compressor=compressor)
            for key, width in [('observation_state', state_dim), ('action_raw', action_dim)]
        }
        cameras = [('camera1', 'observation.images.camera0'), ('camera2', 'observation.images.camera1')]
        if arms == 'right':
            cameras = cameras[1:]
        ends, offset = [], 0
        for dataset_dir, episode_index, length in episodes:
            columns = ['observation.state', 'actions', *(source for _, source in cameras)]
            table = pq.read_table(parquet_path(dataset_dir, episode_index), columns=columns)
            if len(table) != length:
                raise ValueError(f'{dataset_dir.name} episode {episode_index}: metadata/parquet frame count mismatch')
            values = {
                'observation_state': extract_float32_matrix(table['observation.state'], expected_width=state_dim, name='observation.state'),
                'action_raw': extract_float32_matrix(table['actions'], expected_width=action_dim, name='actions'),
            }
            for key, source in cameras:
                values[key] = decode_images(table[source].to_pylist(), dataset_dir, executor)
                if key not in arrays:
                    # The training adapter samples RGB frames independently.
                    # Keep each compressed chunk to one frame for random reads.
                    arrays[key] = data.create_dataset(key, shape=(total_frames, *values[key].shape[1:]),
                                                      chunks=(1, *values[key].shape[1:]),
                                                      dtype='u1', compressor=compressor)
                if arrays[key].shape[1:] != values[key].shape[1:]:
                    raise ValueError(f'{dataset_dir.name} episode {episode_index}: inconsistent RGB dimensions')
            for key, value in values.items():
                arrays[key][offset:offset+length] = value
            offset += length
            ends.append(offset)
            print(f'{dataset_dir.name} episode {episode_index}: {offset}/{total_frames} frames', flush=True)
        replay.create_group('meta').create_dataset('episode_ends', data=np.asarray(ends, dtype=np.int64))
        (temporary / 'raw_tactile_manifest.json').write_text(json.dumps(
            {'version': 1, 'total_frames': total_frames, 'shards': shards}, indent=2)+'\n')
        (temporary / 'prepare_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        for name in generated:
            destination = output_dir / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            (temporary / name).replace(destination)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        shutil.rmtree(temporary)
    print(f'Prepared {len(episodes)} episodes, {total_frames} frames: {output_dir}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--tactile-cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arms', choices=('right', 'both'), default='right')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-episodes-per-dataset', type=int)
    parser.add_argument('--overwrite', action='store_true')
    prepare_data(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
