import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image


class PrepareBaselineDataTest(unittest.TestCase):
    def test_multi_source_raw_and_cache_order_and_reuse(self):
        self._check_multi_source_raw_and_cache_order_and_reuse('right')

    def test_dual_arm_raw_state_actions_and_both_cameras(self):
        self._check_multi_source_raw_and_cache_order_and_reuse('both')

    def _check_multi_source_raw_and_cache_order_and_reuse(self, arms):
        state_dim, action_dim = (20, 20) if arms == 'both' else (7, 10)
        expected_states, expected_actions = {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source, marker in [('a', 10), ('b', 20)]:
                dataset = root / 'sources' / source
                (dataset / 'meta').mkdir(parents=True)
                (dataset / 'data/chunk-000').mkdir(parents=True)
                (dataset / 'meta/episodes.jsonl').write_text(''.join(
                    json.dumps({'episode_index': ep, 'length': 3})+'\n' for ep in [1, 0]))
                for ep in [0, 1]:
                    states = np.tile(np.arange(state_dim, dtype=np.float32), (3, 1))
                    states[:, 0] = marker + ep
                    actions = np.zeros((3, action_dim), np.float32)
                    for arm_offset in range(0, action_dim, 10):
                        actions[:, arm_offset:arm_offset+3] = .000001 * (arm_offset + 1)
                        actions[:, arm_offset+3:arm_offset+9] = [1,0,0,0,1,0]
                        actions[:, arm_offset+9] = marker + arm_offset
                    actions[-1] = 0
                    if ep == 0:
                        expected_states[source], expected_actions[source] = states, actions
                    columns = {'observation.state': pa.array(states.tolist(), type=pa.list_(pa.float32())),
                               'actions': pa.array(actions.tolist(), type=pa.list_(pa.float32()))}
                    for camera in [0, 1]:
                        stream = io.BytesIO()
                        Image.fromarray(np.full((2,2,3), marker+camera, np.uint8)).save(stream, format='PNG')
                        columns[f'observation.images.camera{camera}'] = pa.array([{'bytes': stream.getvalue()}]*3)
                    pq.write_table(pa.table(columns), dataset / f'data/chunk-000/episode_{ep:06d}.parquet')
                cache = root / 'cache/KaiyueChen' / source
                cache.mkdir(parents=True)
                np.save(cache / 'embeddings.npy', np.full((6,4,512), marker, np.float16))
                (cache / 'metadata.json').write_text(json.dumps({'total_frames': 6}))
            try:
                prepare = importlib.import_module('prepare_baseline_data').prepare_data
            except ModuleNotFoundError:
                self.fail('Baseline raw-data preparation implementation is missing')
            args = dict(dataset_root=root/'sources', datasets=['b','a'], tactile_cache_root=root/'cache',
                        output_dir=root/'output', arms=arms, max_episodes_per_dataset=1)
            prepare(**args)
            replay = zarr.open_group(str(root/'output/replay_buffer.zarr'), mode='r')
            np.testing.assert_array_equal(replay['meta/episode_ends'][:], [3, 6])
            np.testing.assert_array_equal(replay['data/observation_state'][:, 0], [20,20,20,10,10,10])
            np.testing.assert_array_equal(replay['data/camera2'][:, 0,0,0], [21,21,21,11,11,11])
            np.testing.assert_array_equal(replay['data/observation_state'][:],
                                          np.concatenate([expected_states[s] for s in ['b','a']]))
            np.testing.assert_array_equal(replay['data/action_raw'][:],
                                          np.concatenate([expected_actions[s] for s in ['b','a']]))
            for key in ['observation_state', 'action_raw']:
                self.assertEqual(replay['data'][key].dtype, np.dtype('float32'))
            cameras = ['camera1','camera2'] if arms == 'both' else ['camera2']
            for camera in cameras:
                self.assertEqual(replay['data'][camera].dtype, np.dtype('uint8'))
                self.assertEqual(replay['data'][camera].chunks, (1, 2, 2, 3))
            if arms == 'both':
                np.testing.assert_array_equal(replay['data/camera1'][:, 0,0,0], [20,20,20,10,10,10])
            else:
                self.assertNotIn('camera1', replay['data'])
            self.assertNotIn('action', replay['data'])
            np.testing.assert_array_equal(replay['data/action_raw'][2], 0)
            self.assertAlmostEqual(float(replay['data/action_raw'][0,0]), .000001)
            manifest = json.loads((root/'output/raw_tactile_manifest.json').read_text())
            self.assertEqual([Path(item['path']).parent.name for item in manifest['shards']], ['b','a'])
            self.assertEqual([(item['start'],item['stop']) for item in manifest['shards']], [(0,3),(0,3)])
            stat = (root/'output/replay_buffer.zarr/.zgroup').stat().st_mtime_ns
            prepare(**args)
            self.assertEqual((root/'output/replay_buffer.zarr/.zgroup').stat().st_mtime_ns, stat)
            with self.assertRaisesRegex(ValueError, 'different|mismatch'):
                prepare(**{**args, 'datasets':['a','b']})


if __name__ == '__main__':
    unittest.main()
