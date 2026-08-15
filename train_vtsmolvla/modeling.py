from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import traverse_util
from train_encoder.utils.resnet import encode_resnet18

from train_smolvla.modeling import JaxSmolVLA, Params, PrefixContext

from .configuration import VTSmolVLAConfig

Array = jax.Array
_TACTILE_ENCODER_PREFIX = "model.tactile_encoder."


def normalize_tactile_embeddings(embeddings: Array, eps: float = 1e-6) -> Array:
    """Normalize each frozen ResNet embedding to unit RMS before projection."""

    embeddings = jnp.asarray(embeddings, dtype=jnp.float32)
    inverse_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(embeddings), axis=-1, keepdims=True) + eps
    )
    return embeddings * inverse_rms


class VTJaxSmolVLA(JaxSmolVLA):
    """SmolVLA visual core extended with tactile prefix-token fusion."""

    config: VTSmolVLAConfig

    @staticmethod
    def _tactile_prefix_inputs(
        tactile_images: Array | None,
        tactile_embeddings: Array | None,
        tactile_masks: Array | None,
    ) -> Mapping[str, Array | None]:
        return {
            "tactile_images": tactile_images,
            "tactile_embeddings": tactile_embeddings,
            "tactile_masks": tactile_masks,
        }

    def _tactile_encoder_variables(self, params: Params) -> dict:
        flat = {
            tuple(name.removeprefix(_TACTILE_ENCODER_PREFIX).split("/")): value
            for name, value in params.items()
            if name.startswith(_TACTILE_ENCODER_PREFIX)
        }
        if not flat:
            raise KeyError("Missing tactile encoder parameters; expected model.tactile_encoder.*")
        return traverse_util.unflatten_dict(flat)

    def embed_tactile(
        self,
        params: Params,
        tactile_images: Array | None = None,
        *,
        tactile_embeddings: Array | None = None,
    ) -> Array:
        if not self.config.use_tactile_encoder:
            raise ValueError("embed_tactile called while use_tactile_encoder=False")
        if (tactile_images is None) == (tactile_embeddings is None):
            raise ValueError("provide exactly one of tactile_images or tactile_embeddings")
        if tactile_embeddings is None:
            assert tactile_images is not None
            if tactile_images.ndim != 5:
                raise ValueError(
                    f"tactile_images must be [B,N,H,W,C], got {tactile_images.shape}"
                )
            batch, token_count = tactile_images.shape[:2]
            flat_images = tactile_images.reshape(
                (batch * token_count,) + tactile_images.shape[2:]
            )
            tactile_tokens, _ = encode_resnet18(
                self._tactile_encoder_variables(params),
                jnp.asarray(flat_images, dtype=jnp.float32),
                train=False,
                embedding_dim=self.config.tactile_embedding_dim,
            )
            if self.config.freeze_tactile_encoder:
                tactile_tokens = jax.lax.stop_gradient(tactile_tokens)
            tactile_tokens = tactile_tokens.reshape(
                batch,
                token_count,
                self.config.tactile_embedding_dim,
            )
        else:
            tactile_tokens = jnp.asarray(tactile_embeddings, dtype=jnp.float32)
            if tactile_tokens.ndim != 3:
                raise ValueError(
                    "tactile_embeddings must be [B,N,D], got "
                    f"{tactile_tokens.shape}"
                )
            _, token_count, embedding_dim = tactile_tokens.shape
            if embedding_dim != self.config.tactile_embedding_dim:
                raise ValueError(
                    f"Expected tactile embedding dim {self.config.tactile_embedding_dim}, "
                    f"got {embedding_dim}"
                )
        if token_count != self.config.tactile_num_tokens:
            raise ValueError(
                f"Expected {self.config.tactile_num_tokens} tactile tokens, got {token_count}"
            )
        tactile_tokens = normalize_tactile_embeddings(tactile_tokens)
        return self._linear(params, "model.tactile_proj", tactile_tokens, bias=True)

    def prefix_inputs_from_batch(
        self,
        batch: Mapping[str, Array],
    ) -> Mapping[str, Array] | None:
        if not self.config.use_tactile_encoder:
            return None
        return {
            key: batch[key]
            for key in ("tactile_images", "tactile_embeddings", "tactile_masks")
            if key in batch
        }

    def _embed_prefix_extension(
        self,
        params: Params,
        prefix_inputs: Mapping[str, Array] | None,
    ) -> tuple[Sequence[Array], Sequence[Array], Sequence[Array]]:
        if not self.config.use_tactile_encoder:
            return (), (), ()
        inputs = dict(prefix_inputs or {})
        tactile_masks = inputs.get("tactile_masks")
        if tactile_masks is None:
            raise ValueError("tactile_masks are required for VT-SmolVLA")
        tactile_embedding = self.embed_tactile(
            params,
            inputs.get("tactile_images"),
            tactile_embeddings=inputs.get("tactile_embeddings"),
        )
        tactile_masks = jnp.asarray(tactile_masks, dtype=jnp.bool_)
        if tactile_masks.shape != tactile_embedding.shape[:2]:
            raise ValueError(
                f"tactile_masks must have shape {tactile_embedding.shape[:2]}, "
                f"got {tactile_masks.shape}"
            )
        segment = jnp.zeros(tactile_embedding.shape[1], dtype=jnp.bool_)
        return (tactile_embedding,), (tactile_masks,), (segment,)

    def embed_prefix(
        self,
        params: Params,
        images: Array | Sequence[Array],
        image_masks: Array | Sequence[Array],
        language_tokens: Array,
        language_masks: Array,
        state: Array,
        state_mask: Array | None = None,
        tactile_images: Array | None = None,
        tactile_embeddings: Array | None = None,
        tactile_masks: Array | None = None,
        prefix_inputs: Mapping[str, Array] | None = None,
    ) -> tuple[Array, Array, Array]:
        if prefix_inputs is None and self.config.use_tactile_encoder:
            prefix_inputs = self._tactile_prefix_inputs(
                tactile_images,
                tactile_embeddings,
                tactile_masks,
            )
        return super().embed_prefix(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            state_mask=state_mask,
            prefix_inputs=prefix_inputs,
        )

    def flow_velocity(
        self,
        params: Params,
        images: Array,
        image_masks: Array,
        language_tokens: Array,
        language_masks: Array,
        state: Array,
        noisy_actions: Array,
        timestep: Array,
        state_mask: Array | None = None,
        tactile_images: Array | None = None,
        tactile_embeddings: Array | None = None,
        tactile_masks: Array | None = None,
        prefix_inputs: Mapping[str, Array] | None = None,
    ) -> Array:
        if prefix_inputs is None and self.config.use_tactile_encoder:
            prefix_inputs = self._tactile_prefix_inputs(
                tactile_images,
                tactile_embeddings,
                tactile_masks,
            )
        return super().flow_velocity(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            noisy_actions,
            timestep,
            state_mask=state_mask,
            prefix_inputs=prefix_inputs,
        )

    def build_prefix_context(
        self,
        params: Params,
        images: Array,
        image_masks: Array,
        language_tokens: Array,
        language_masks: Array,
        state: Array,
        state_mask: Array | None = None,
        tactile_images: Array | None = None,
        tactile_embeddings: Array | None = None,
        tactile_masks: Array | None = None,
        prefix_inputs: Mapping[str, Array] | None = None,
    ) -> PrefixContext:
        if prefix_inputs is None and self.config.use_tactile_encoder:
            prefix_inputs = self._tactile_prefix_inputs(
                tactile_images,
                tactile_embeddings,
                tactile_masks,
            )
        return super().build_prefix_context(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            state_mask=state_mask,
            prefix_inputs=prefix_inputs,
        )

    def sample_actions(
        self,
        params: Params,
        images: Array,
        image_masks: Array,
        language_tokens: Array,
        language_masks: Array,
        state: Array,
        rng: Array,
        *,
        tactile_images: Array | None = None,
        tactile_embeddings: Array | None = None,
        tactile_masks: Array | None = None,
        noise: Array | None = None,
        num_steps: int | None = None,
        previous_chunk: Array | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        prefix_inputs: Mapping[str, Array] | None = None,
    ) -> Array:
        if prefix_inputs is None and self.config.use_tactile_encoder:
            prefix_inputs = self._tactile_prefix_inputs(
                tactile_images,
                tactile_embeddings,
                tactile_masks,
            )
        return super().sample_actions(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            rng,
            prefix_inputs=prefix_inputs,
            noise=noise,
            num_steps=num_steps,
            previous_chunk=previous_chunk,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
