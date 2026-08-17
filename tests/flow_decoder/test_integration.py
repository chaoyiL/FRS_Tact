from __future__ import annotations

import unittest

import jax.numpy as jnp

from utils.integration import euler_integrate_velocity
from utils.integration import fireflow_integrate_velocity
from utils.integration import slerpflow_integrate_velocity


class EulerIntegrationTest(unittest.TestCase):
    def test_constant_positive_velocity_integrates_forward(self):
        x = jnp.zeros((2, 3, 4), dtype=jnp.float32)
        result = euler_integrate_velocity(lambda value, time: jnp.ones_like(value), x, num_steps=20)
        self.assertTrue(jnp.allclose(result, 1.0, atol=1e-6))

    def test_time_starts_at_zero_and_stops_before_one(self):
        x = jnp.zeros((1, 1, 1), dtype=jnp.float32)

        def velocity(value, time):
            return jnp.broadcast_to(time[:, None, None], value.shape)

        result = euler_integrate_velocity(velocity, x, num_steps=10)
        # Forward Euler samples t=0,.1,...,.9, whose mean is .45.
        self.assertTrue(jnp.allclose(result, 0.45, atol=1e-6))

    def test_invalid_step_count(self):
        with self.assertRaises(ValueError):
            euler_integrate_velocity(lambda value, time: value, jnp.zeros((1, 1, 1)), num_steps=0)


class FireFlowIntegrationTest(unittest.TestCase):
    def test_shape_preserved(self):
        x = jnp.zeros((2, 3, 4), dtype=jnp.float32)
        for num_steps in (1, 4, 10):
            with self.subTest(num_steps=num_steps):
                result = fireflow_integrate_velocity(
                    lambda value, time: jnp.ones_like(value),
                    x,
                    num_steps=num_steps,
                )
                self.assertEqual(result.shape, x.shape)

    def test_nfe_equals_num_steps_plus_one(self):
        for num_steps in (1, 5, 10):
            with self.subTest(num_steps=num_steps):
                x = jnp.zeros((2, 3, 4), dtype=jnp.float32)
                _, nfe = fireflow_integrate_velocity(
                    lambda value, time: jnp.ones_like(value),
                    x,
                    num_steps=num_steps,
                    return_nfe=True,
                )
                self.assertEqual(int(nfe), num_steps + 1)

    def test_invalid_step_count(self):
        with self.assertRaises(ValueError):
            fireflow_integrate_velocity(
                lambda value, time: value,
                jnp.zeros((1, 1, 1)),
                num_steps=0,
            )


class SlerpFlowIntegrationTest(unittest.TestCase):
    def test_shape_and_finiteness_are_preserved(self):
        x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4) / 24.0

        def velocity(value, time):
            return 0.1 * value + time[:, None, None]

        for num_steps in (1, 4, 10):
            with self.subTest(num_steps=num_steps):
                result = slerpflow_integrate_velocity(velocity, x, num_steps=num_steps)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(bool(jnp.all(jnp.isfinite(result))))

    def test_nfe_equals_num_steps_plus_one(self):
        x = jnp.ones((2, 3, 4), dtype=jnp.float32)
        for num_steps in (1, 5, 10):
            with self.subTest(num_steps=num_steps):
                _, nfe = slerpflow_integrate_velocity(
                    lambda value, time: jnp.ones_like(value),
                    x,
                    num_steps=num_steps,
                    return_nfe=True,
                )
                self.assertEqual(int(nfe), num_steps + 1)

    def test_constant_radial_velocity_integrates_forward(self):
        x = jnp.ones((2, 3, 4), dtype=jnp.float32)
        direction = x / jnp.linalg.norm(x, axis=(1, 2), keepdims=True)
        result = slerpflow_integrate_velocity(
            lambda value, time: direction,
            x,
            num_steps=10,
        )
        self.assertTrue(jnp.allclose(result, x + direction, atol=1e-5))

    def test_invalid_step_count(self):
        with self.assertRaises(ValueError):
            slerpflow_integrate_velocity(
                lambda value, time: value,
                jnp.ones((1, 1, 1)),
                num_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
