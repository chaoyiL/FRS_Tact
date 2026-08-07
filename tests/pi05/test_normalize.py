"""Tests for pi0.5 normalization and the borrowed-norm-stats dimension padding.

SKIPPED unless the pi0.5 deps are importable. `pi05_jax.normalize` imports `download`
(filelock/fsspec/tqdm_loggable) and `modalities_eval.pi05_utils` imports jax, so these run on the
training server but not on a plain dev box -- same situation as tests/jax/*. The numbers asserted
below were verified by hand against the real
gs://openpi-assets/checkpoints/pi05_base/assets/trossen/norm_stats.json before being written down
here; see pi05_frs_plan.md.

What matters here: the configured setup borrows pi05_base's `trossen` stats (14-dim) for
pick_tube (20-dim), so `_match_norm_stats_dim` pads the extra 6 dims to an identity transform.
If that padding is ever wrong, state/actions get silently mis-scaled -- and for pi0.5 the state
also feeds the tokenized prompt, so it would corrupt the prompt too.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from lerobot.policies.pi05_jax.normalize import NormStats
    from lerobot.policies.pi05_jax.normalize import apply as normalize_apply
    from lerobot.policies.pi05_jax.normalize import unapply as normalize_unapply

    from modalities_eval.pi05_utils import _match_norm_stats_dim

    DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    DEPS_AVAILABLE = False


def make_stats(dim: int, *, quantiles: bool = True) -> "NormStats":
    rng = np.random.default_rng(0)
    return NormStats(
        mean=rng.normal(size=dim).astype(np.float32),
        std=(rng.uniform(0.5, 2.0, size=dim)).astype(np.float32),
        q01=(-rng.uniform(0.5, 2.0, size=dim)).astype(np.float32) if quantiles else None,
        q99=(rng.uniform(0.5, 2.0, size=dim)).astype(np.float32) if quantiles else None,
    )


@unittest.skipUnless(DEPS_AVAILABLE, "pi0.5 deps (jax/fsspec/filelock) not installed")
class MatchNormStatsDimTest(unittest.TestCase):
    def test_same_dim_is_returned_unchanged(self):
        stats = make_stats(20)
        self.assertIs(_match_norm_stats_dim(stats, 20, label="state"), stats)

    def test_narrower_stats_are_padded_to_an_identity_transform(self):
        padded = _match_norm_stats_dim(make_stats(14), 20, label="state")
        self.assertEqual(padded.mean.shape[-1], 20)
        np.testing.assert_array_equal(padded.mean[14:], np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(padded.std[14:], np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(padded.q01[14:], -np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(padded.q99[14:], np.ones(6, dtype=np.float32))

    def test_padded_dims_pass_through_both_norm_modes(self):
        """Identity only up to the 1e-6 epsilon apply() adds to the denominator, hence atol."""
        padded = _match_norm_stats_dim(make_stats(14), 20, label="state")
        x = np.random.default_rng(1).uniform(-5, 5, 20).astype(np.float32)
        for use_quantiles in (True, False):
            with self.subTest(use_quantiles=use_quantiles):
                y = normalize_apply(x, padded, use_quantiles=use_quantiles)
                np.testing.assert_allclose(y[14:], x[14:], atol=1e-5)
                # The real dims must actually be transformed, or the padding is masking a no-op.
                self.assertFalse(np.allclose(y[:14], x[:14]))

    def test_missing_quantiles_are_left_none(self):
        padded = _match_norm_stats_dim(make_stats(14, quantiles=False), 20, label="state")
        self.assertIsNone(padded.q01)
        self.assertIsNone(padded.q99)

    def test_wider_than_dataset_stats_are_rejected(self):
        """Silently truncating would mean normalizing with another robot's stats for those dims."""
        with self.assertRaises(ValueError):
            _match_norm_stats_dim(make_stats(20), 14, label="state")


@unittest.skipUnless(DEPS_AVAILABLE, "pi0.5 deps (jax/fsspec/filelock) not installed")
class NormalizeRoundTripTest(unittest.TestCase):
    def test_round_trip_1d(self):
        stats = make_stats(20)
        x = np.random.default_rng(2).uniform(-3, 3, 20).astype(np.float32)
        for use_quantiles in (True, False):
            with self.subTest(use_quantiles=use_quantiles):
                y = normalize_apply(x, stats, use_quantiles=use_quantiles)
                back = normalize_unapply(y, stats, use_quantiles=use_quantiles)
                np.testing.assert_allclose(back, x, atol=1e-5)

    def test_round_trip_2d_action_chunk_broadcasts_over_horizon(self):
        stats = make_stats(20)
        actions = np.random.default_rng(3).uniform(-3, 3, (50, 20)).astype(np.float32)
        y = normalize_apply(actions, stats, use_quantiles=True)
        self.assertEqual(y.shape, (50, 20))
        np.testing.assert_allclose(
            normalize_unapply(y, stats, use_quantiles=True), actions, atol=1e-5
        )

    def test_unapply_leaves_columns_beyond_the_stats_untouched(self):
        """Model output is action_dim (32) wide while the stats cover the dataset's real width;
        the zero-padded tail must not be rescaled."""
        stats = make_stats(20)
        wide = np.random.default_rng(4).uniform(-1, 1, (50, 32)).astype(np.float32)
        out = normalize_unapply(wide, stats, use_quantiles=True)
        self.assertEqual(out.shape, (50, 32))
        np.testing.assert_array_equal(out[:, 20:], wide[:, 20:])

    def test_quantile_norm_maps_the_quantile_range_onto_minus_one_to_one(self):
        """pi0.5 discretizes state into the prompt assuming [-1, 1]; that is why the config uses
        quantile norm (see pi05_jax/tokenizer.py)."""
        stats = make_stats(20)
        np.testing.assert_allclose(
            normalize_apply(stats.q01, stats, use_quantiles=True), -np.ones(20), atol=1e-5
        )
        np.testing.assert_allclose(
            normalize_apply(stats.q99, stats, use_quantiles=True), np.ones(20), atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
