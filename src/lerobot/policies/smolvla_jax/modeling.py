from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .configuration import JaxSmolVLAConfig
from .functional import (
    apply_rope,
    eager_attention,
    gelu_pytorch_tanh,
    layer_norm,
    linear,
    make_att_2d_masks,
    pad_last_dim,
    rms_norm,
    silu_pytorch,
    sinusoidal_time_embedding,
)
from .rtc import rtc_guided_velocity

Array = jax.Array
type Params = Mapping[str, Array]
type KVCache = tuple[tuple[Array, Array], ...]


@dataclass(frozen=True)
class PrefixContext:
    pad_mask: Array
    cache: KVCache


class JaxSmolVLA:
    """Functional JAX implementation of the PyTorch SmolVLA checkpoint."""

    def __init__(self, config: JaxSmolVLAConfig):
        self.config = config

    @staticmethod
    def _p(params: Params, name: str) -> Array:
        try:
            return params[name]
        except KeyError as error:
            raise KeyError(f"Missing SmolVLA parameter: {name}") from error

    def _linear(self, params: Params, prefix: str, x: Array, *, bias: bool = False) -> Array:
        weight = self._p(params, f"{prefix}.weight")
        bias_value = self._p(params, f"{prefix}.bias") if bias else None
        return linear(x, weight, bias_value)

    def _vision_layer(self, params: Params, hidden: Array, layer_index: int) -> Array:
        prefix = f"model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{layer_index}"
        residual = hidden
        hidden = layer_norm(
            hidden,
            self._p(params, f"{prefix}.layer_norm1.weight"),
            self._p(params, f"{prefix}.layer_norm1.bias"),
            self.config.vision_layer_norm_eps,
        )
        query = self._linear(params, f"{prefix}.self_attn.q_proj", hidden, bias=True)
        key = self._linear(params, f"{prefix}.self_attn.k_proj", hidden, bias=True)
        value = self._linear(params, f"{prefix}.self_attn.v_proj", hidden, bias=True)
        batch, length, _ = query.shape
        shape = (batch, length, self.config.vision_num_heads, -1)
        mask = jnp.ones((batch, length, length), dtype=jnp.bool_)
        attention = eager_attention(query.reshape(shape), key.reshape(shape), value.reshape(shape), mask)
        hidden = residual + self._linear(params, f"{prefix}.self_attn.out_proj", attention, bias=True)

        residual = hidden
        hidden = layer_norm(
            hidden,
            self._p(params, f"{prefix}.layer_norm2.weight"),
            self._p(params, f"{prefix}.layer_norm2.bias"),
            self.config.vision_layer_norm_eps,
        )
        hidden = self._linear(params, f"{prefix}.mlp.fc1", hidden, bias=True)
        hidden = gelu_pytorch_tanh(hidden)
        hidden = self._linear(params, f"{prefix}.mlp.fc2", hidden, bias=True)
        return residual + hidden

    def embed_image(self, params: Params, image: Array) -> Array:
        """Encode normalized BCHW images into 64 SmolVLM tokens."""

        prefix = "model.vlm_with_expert.vlm.model.vision_model"
        patch_weight = self._p(params, f"{prefix}.embeddings.patch_embedding.weight")
        patch_bias = self._p(params, f"{prefix}.embeddings.patch_embedding.bias")
        image = image.astype(patch_weight.dtype)
        patches = jax.lax.conv_general_dilated(
            image,
            patch_weight,
            window_strides=(self.config.vision_patch_size, self.config.vision_patch_size),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            preferred_element_type=jnp.float32,
        ).astype(patch_weight.dtype)
        patches = patches + patch_bias[None, :, None, None]
        hidden = jnp.transpose(patches, (0, 2, 3, 1)).reshape(image.shape[0], -1, patches.shape[1])
        positions = self._p(params, f"{prefix}.embeddings.position_embedding.weight")
        hidden = hidden + positions[: hidden.shape[1]][None, :, :]

        for layer_index in range(self.config.vision_num_layers):
            hidden = self._vision_layer(params, hidden, layer_index)

        hidden = layer_norm(
            hidden,
            self._p(params, f"{prefix}.post_layernorm.weight"),
            self._p(params, f"{prefix}.post_layernorm.bias"),
            self.config.vision_layer_norm_eps,
        )
        return self._connector(params, hidden)

    def _connector(self, params: Params, hidden: Array) -> Array:
        scale = self.config.connector_scale_factor
        batch, sequence, channels = hidden.shape
        side = int(sequence**0.5)
        if side * side != sequence or side % scale:
            raise ValueError(f"connector expects a square patch grid divisible by {scale}, got {sequence}")
        hidden = hidden.reshape(batch, side, side, channels)
        hidden = hidden.reshape(batch, side, side // scale, channels * scale)
        hidden = jnp.transpose(hidden, (0, 2, 1, 3))
        hidden = hidden.reshape(batch, side // scale, side // scale, channels * scale * scale)
        hidden = jnp.transpose(hidden, (0, 2, 1, 3))
        hidden = hidden.reshape(batch, sequence // (scale * scale), channels * scale * scale)
        return self._linear(
            params,
            "model.vlm_with_expert.vlm.model.connector.modality_projection.proj",
            hidden,
        )

    def embed_language(self, params: Params, tokens: Array) -> Array:
        embedding = self._p(params, "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")
        return embedding[tokens]

    def embed_prefix(
        self,
        params: Params,
        images: Array | Sequence[Array],
        image_masks: Array | Sequence[Array],
        language_tokens: Array,
        language_masks: Array,
        state: Array,
    ) -> tuple[Array, Array, Array]:
        if isinstance(images, jax.Array):
            if images.ndim != 5:
                raise ValueError(f"images must be [B,N,C,H,W], got {images.shape}")
            image_list = [images[:, index] for index in range(images.shape[1])]
        else:
            image_list = list(images)
        if isinstance(image_masks, jax.Array):
            mask_list = [image_masks[:, index] for index in range(image_masks.shape[1])]
        else:
            mask_list = list(image_masks)
        if len(image_list) != len(mask_list):
            raise ValueError("images and image_masks must contain the same number of cameras")

        embeddings: list[Array] = []
        pad_masks: list[Array] = []
        attention_segments: list[Array] = []
        batch = state.shape[0]
        for image, mask in zip(image_list, mask_list, strict=True):
            if self.config.add_image_special_tokens:
                start_tokens = jnp.broadcast_to(
                    jnp.asarray(
                        [self.config.fake_image_token_id, self.config.global_image_token_id],
                        dtype=jnp.int32,
                    )[None, :],
                    (batch, 2),
                )
                start_embedding = self.embed_language(params, start_tokens)
                embeddings.append(start_embedding)
                pad_masks.append(jnp.ones(start_embedding.shape[:2], dtype=jnp.bool_))
                attention_segments.append(jnp.zeros(start_embedding.shape[1], dtype=jnp.bool_))
            image_embedding = self.embed_image(params, image)
            image_embedding = image_embedding * jnp.sqrt(
                jnp.asarray(image_embedding.shape[-1], dtype=image_embedding.dtype)
            )
            embeddings.append(image_embedding)
            pad_masks.append(jnp.broadcast_to(mask[:, None].astype(jnp.bool_), image_embedding.shape[:2]))
            attention_segments.append(jnp.zeros(image_embedding.shape[1], dtype=jnp.bool_))
            if self.config.add_image_special_tokens:
                end_tokens = jnp.full((batch, 1), self.config.fake_image_token_id, dtype=jnp.int32)
                end_embedding = self.embed_language(params, end_tokens)
                embeddings.append(end_embedding)
                pad_masks.append(jnp.ones(end_embedding.shape[:2], dtype=jnp.bool_))
                attention_segments.append(jnp.zeros(end_embedding.shape[1], dtype=jnp.bool_))

        language_embedding = self.embed_language(params, language_tokens)
        language_embedding = language_embedding * jnp.sqrt(
            jnp.asarray(language_embedding.shape[-1], dtype=language_embedding.dtype)
        )
        embeddings.append(language_embedding)
        pad_masks.append(language_masks.astype(jnp.bool_))
        attention_segments.append(jnp.zeros(language_embedding.shape[1], dtype=jnp.bool_))

        state = pad_last_dim(state, self.config.max_state_dim)
        state_embedding = self._linear(params, "model.state_proj", state, bias=True)[:, None, :]
        embeddings.append(state_embedding)
        pad_masks.append(jnp.ones((batch, 1), dtype=jnp.bool_))
        attention_segments.append(jnp.ones(1, dtype=jnp.bool_))

        embedding = jnp.concatenate(embeddings, axis=1)
        pad_mask = jnp.concatenate(pad_masks, axis=1)
        attention_ar = jnp.concatenate(attention_segments)[None, :]
        attention_ar = jnp.broadcast_to(attention_ar, pad_mask.shape)
        if embedding.shape[1] < self.config.prefix_length:
            pad = self.config.prefix_length - embedding.shape[1]
            embedding = jnp.pad(embedding, ((0, 0), (0, pad), (0, 0)))
            pad_mask = jnp.pad(pad_mask, ((0, 0), (0, pad)))
            attention_ar = jnp.pad(attention_ar, ((0, 0), (0, pad)))
        return embedding, pad_mask, attention_ar

    def embed_suffix(
        self, params: Params, noisy_actions: Array, timestep: Array
    ) -> tuple[Array, Array, Array]:
        action_embedding = self._linear(params, "model.action_in_proj", noisy_actions, bias=True)
        time_embedding = sinusoidal_time_embedding(
            timestep,
            self.config.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
        ).astype(action_embedding.dtype)
        time_embedding = jnp.broadcast_to(time_embedding[:, None, :], action_embedding.shape)
        hidden = jnp.concatenate((action_embedding, time_embedding), axis=-1)
        hidden = self._linear(params, "model.action_time_mlp_in", hidden, bias=True)
        hidden = jax.nn.silu(hidden)
        hidden = self._linear(params, "model.action_time_mlp_out", hidden, bias=True)
        pad_mask = jnp.ones(hidden.shape[:2], dtype=jnp.bool_)
        attention_ar = jnp.ones(hidden.shape[:2], dtype=jnp.bool_)
        return hidden, pad_mask, attention_ar

    def _expert_layer_index(self, layer_index: int) -> int | None:
        multiple = self.config.num_vlm_layers // self.config.num_expert_layers
        if multiple > 0 and layer_index > 0 and layer_index % multiple != 0:
            return None
        return layer_index // multiple if multiple > 0 else layer_index

    def _layer_prefix(self, expert: bool, layer_index: int) -> str:
        if expert:
            expert_index = self._expert_layer_index(layer_index)
            if expert_index is None:
                raise ValueError(f"VLM layer {layer_index} has no corresponding expert layer")
            return f"model.vlm_with_expert.lm_expert.layers.{expert_index}"
        return f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_index}"

    def _project_qkv(
        self, params: Params, hidden: Array, layer_index: int, *, expert: bool
    ) -> tuple[Array, Array, Array]:
        prefix = self._layer_prefix(expert, layer_index)
        hidden = rms_norm(
            hidden,
            self._p(params, f"{prefix}.input_layernorm.weight"),
            self.config.text_rms_norm_eps,
        )
        q_weight = self._p(params, f"{prefix}.self_attn.q_proj.weight")
        hidden = hidden.astype(q_weight.dtype)
        query = linear(hidden, q_weight).reshape(*hidden.shape[:2], -1, self.config.head_dim)
        key = self._linear(params, f"{prefix}.self_attn.k_proj", hidden).reshape(
            *hidden.shape[:2], -1, self.config.head_dim
        )
        value = self._linear(params, f"{prefix}.self_attn.v_proj", hidden).reshape(
            *hidden.shape[:2], -1, self.config.head_dim
        )
        return query, key, value

    def _finish_decoder_layer(
        self, params: Params, hidden: Array, attention: Array, layer_index: int, *, expert: bool
    ) -> Array:
        prefix = self._layer_prefix(expert, layer_index)
        output_weight = self._p(params, f"{prefix}.self_attn.o_proj.weight")
        attention = attention.astype(output_weight.dtype)
        projected = linear(attention, output_weight)
        # The reference uses an in-place ``out_emb += hidden_states``.  PyTorch
        # therefore casts the residual to the projection dtype (BF16 here)
        # instead of promoting the projection to FP32.
        hidden = (projected + hidden.astype(projected.dtype)).astype(projected.dtype)
        residual = hidden
        hidden = rms_norm(
            hidden,
            self._p(params, f"{prefix}.post_attention_layernorm.weight"),
            self.config.text_rms_norm_eps,
        )
        gate = self._linear(params, f"{prefix}.mlp.gate_proj", hidden)
        up = self._linear(params, f"{prefix}.mlp.up_proj", hidden)
        hidden = self._linear(params, f"{prefix}.mlp.down_proj", silu_pytorch(gate) * up)
        return (hidden + residual.astype(hidden.dtype)).astype(hidden.dtype)

    def _self_attention_layer(
        self,
        params: Params,
        prefix_hidden: Array | None,
        expert_hidden: Array | None,
        layer_index: int,
        position_ids: Array,
        attention_mask: Array,
        cached_kv: tuple[Array, Array] | None,
    ) -> tuple[Array | None, Array | None, tuple[Array, Array]]:
        queries: list[Array] = []
        keys: list[Array] = []
        values: list[Array] = []
        lengths: list[int] = []
        if prefix_hidden is not None:
            q, k, v = self._project_qkv(params, prefix_hidden, layer_index, expert=False)
            queries.append(q)
            keys.append(k)
            values.append(v)
            lengths.append(prefix_hidden.shape[1])
        if expert_hidden is not None:
            q, k, v = self._project_qkv(params, expert_hidden, layer_index, expert=True)
            queries.append(q)
            keys.append(k)
            values.append(v)
            lengths.append(expert_hidden.shape[1])
        query = jnp.concatenate(queries, axis=1)
        key = jnp.concatenate(keys, axis=1)
        value = jnp.concatenate(values, axis=1)
        local_positions = (
            position_ids[:, : query.shape[1]] if query.shape[1] < position_ids.shape[1] else position_ids
        )
        local_mask = (
            attention_mask[:, : query.shape[1], : query.shape[1]]
            if query.shape[1] < position_ids.shape[1]
            else attention_mask
        )
        query = apply_rope(query, local_positions)
        key = apply_rope(key, local_positions)
        new_cache = (key, value)
        if cached_kv is not None:
            key = jnp.concatenate((cached_kv[0], key), axis=1)
            value = jnp.concatenate((cached_kv[1], value), axis=1)
            local_mask = attention_mask
        attention = eager_attention(query, key, value, local_mask)

        prefix_output: Array | None = None
        expert_output: Array | None = None
        offset = 0
        if prefix_hidden is not None:
            prefix_attention = attention[:, offset : offset + lengths[0]]
            prefix_output = self._finish_decoder_layer(
                params, prefix_hidden, prefix_attention, layer_index, expert=False
            )
            offset += lengths[0]
        if expert_hidden is not None:
            expert_attention = attention[:, offset : offset + expert_hidden.shape[1]]
            expert_output = self._finish_decoder_layer(
                params, expert_hidden, expert_attention, layer_index, expert=True
            )
        return prefix_output, expert_output, new_cache

    def _cross_attention_layer(
        self,
        params: Params,
        prefix_hidden: Array | None,
        expert_hidden: Array,
        layer_index: int,
        position_ids: Array,
        attention_mask: Array,
        cached_kv: tuple[Array, Array] | None,
    ) -> tuple[Array | None, Array, tuple[Array, Array]]:
        prefix_output: Array | None = None
        if cached_kv is None:
            if prefix_hidden is None:
                raise ValueError("cross attention needs prefix hidden states or a KV cache")
            prefix_length = prefix_hidden.shape[1]
            prefix_positions = position_ids[:, :prefix_length]
            prefix_mask = attention_mask[:, :prefix_length, :prefix_length]
            query, key, value = self._project_qkv(params, prefix_hidden, layer_index, expert=False)
            query = apply_rope(query, prefix_positions)
            key = apply_rope(key, prefix_positions)
            prefix_attention = eager_attention(query, key, value, prefix_mask)
            prefix_output = self._finish_decoder_layer(
                params, prefix_hidden, prefix_attention, layer_index, expert=False
            )
            cached_kv = (key, value)

        prefix_key, prefix_value = cached_kv
        expert_prefix = self._layer_prefix(True, layer_index)
        expert_norm = rms_norm(
            expert_hidden,
            self._p(params, f"{expert_prefix}.input_layernorm.weight"),
            self.config.text_rms_norm_eps,
        )
        q_weight = self._p(params, f"{expert_prefix}.self_attn.q_proj.weight")
        query = linear(expert_norm.astype(q_weight.dtype), q_weight).reshape(
            *expert_hidden.shape[:2], -1, self.config.head_dim
        )
        flat_key = prefix_key.reshape(*prefix_key.shape[:2], -1)
        flat_value = prefix_value.reshape(*prefix_value.shape[:2], -1)
        expert_key = self._linear(params, f"{expert_prefix}.self_attn.k_proj", flat_key).reshape(
            *flat_key.shape[:2], -1, self.config.head_dim
        )
        expert_value = self._linear(params, f"{expert_prefix}.self_attn.v_proj", flat_value).reshape(
            *flat_value.shape[:2], -1, self.config.head_dim
        )
        expert_positions = position_ids[:, -expert_hidden.shape[1] :]
        expert_positions = expert_positions - jnp.min(expert_positions, axis=1, keepdims=True)
        query = apply_rope(query, expert_positions)
        expert_mask = attention_mask[:, -expert_hidden.shape[1] :, : expert_key.shape[1]]
        attention = eager_attention(query, expert_key, expert_value, expert_mask)
        expert_output = self._finish_decoder_layer(params, expert_hidden, attention, layer_index, expert=True)
        return prefix_output, expert_output, cached_kv

    def transformer(
        self,
        params: Params,
        prefix_hidden: Array | None,
        expert_hidden: Array | None,
        attention_mask: Array,
        position_ids: Array,
        *,
        cache: KVCache | None = None,
        fill_cache: bool = False,
    ) -> tuple[Array | None, Array | None, KVCache]:
        cache_output: list[tuple[Array, Array]] = []
        for layer_index in range(self.config.num_vlm_layers):
            cached_kv = None if cache is None else cache[layer_index]
            if expert_hidden is not None and self._expert_layer_index(layer_index) is None:
                if prefix_hidden is not None:
                    prefix_hidden, _, new_cache = self._self_attention_layer(
                        params,
                        prefix_hidden,
                        None,
                        layer_index,
                        position_ids,
                        attention_mask,
                        cached_kv,
                    )
                elif cached_kv is not None:
                    new_cache = cached_kv
                else:
                    raise ValueError(f"layer {layer_index} has neither inputs nor cached prefix")
                cache_output.append(new_cache)
                continue
            use_self_attention = (
                fill_cache
                or "cross" not in self.config.attention_mode
                or self.config.self_attn_every_n_layers <= 0
                or layer_index % self.config.self_attn_every_n_layers == 0
            )
            if use_self_attention:
                prefix_hidden, expert_hidden, new_cache = self._self_attention_layer(
                    params,
                    prefix_hidden,
                    expert_hidden,
                    layer_index,
                    position_ids,
                    attention_mask,
                    cached_kv,
                )
            else:
                if expert_hidden is None:
                    raise ValueError("cross-attention layer requires expert hidden states")
                prefix_hidden, expert_hidden, new_cache = self._cross_attention_layer(
                    params,
                    prefix_hidden,
                    expert_hidden,
                    layer_index,
                    position_ids,
                    attention_mask,
                    cached_kv,
                )
            cache_output.append(new_cache)

        if prefix_hidden is not None:
            prefix_hidden = rms_norm(
                prefix_hidden,
                self._p(params, "model.vlm_with_expert.vlm.model.text_model.norm.weight"),
                self.config.text_rms_norm_eps,
            )
        if expert_hidden is not None:
            expert_hidden = rms_norm(
                expert_hidden,
                self._p(params, "model.vlm_with_expert.lm_expert.norm.weight"),
                self.config.text_rms_norm_eps,
            )
        return prefix_hidden, expert_hidden, tuple(cache_output)

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
    ) -> Array:
        prefix, prefix_pad, prefix_ar = self.embed_prefix(
            params, images, image_masks, language_tokens, language_masks, state
        )
        suffix, suffix_pad, suffix_ar = self.embed_suffix(params, noisy_actions, timestep)
        pad_mask = jnp.concatenate((prefix_pad, suffix_pad), axis=1)
        attention_ar = jnp.concatenate((prefix_ar, suffix_ar), axis=1)
        attention_mask = make_att_2d_masks(pad_mask, attention_ar)
        position_ids = jnp.cumsum(pad_mask, axis=1) - 1
        _, suffix, _ = self.transformer(
            params, prefix, suffix, attention_mask, position_ids, fill_cache=False
        )
        suffix = suffix[:, -self.config.chunk_size :].astype(jnp.float32)
        return self._linear(params, "model.action_out_proj", suffix, bias=True)

    def loss(
        self,
        params: Params,
        batch: Mapping[str, Array],
        rng: Array,
        *,
        noise: Array | None = None,
        time: Array | None = None,
        reduction: str = "mean",
    ) -> Array:
        actions = pad_last_dim(batch["actions"], self.config.max_action_dim)
        noise_rng, time_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape, dtype=jnp.float32)
        if time is None:
            time = jax.random.beta(time_rng, 1.5, 1.0, (actions.shape[0],), dtype=jnp.float32)
            time = time * 0.999 + 0.001
        expanded_time = time[:, None, None]
        x_t = expanded_time * noise + (1.0 - expanded_time) * actions
        target = noise - actions
        velocity = self.flow_velocity(
            params,
            batch["images"],
            batch["image_masks"],
            batch["language_tokens"],
            batch["language_masks"],
            batch["state"],
            x_t,
            time,
        )
        losses = jnp.square(target - velocity)[..., : self.config.action_dim]
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = (~action_is_pad).astype(losses.dtype)[..., None]
            losses = losses * valid
            denominator = jnp.maximum(jnp.sum(valid) * losses.shape[-1], 1)
            if reduction == "mean":
                return jnp.sum(losses) / denominator
            per_sample_denominator = jnp.maximum(jnp.sum(valid, axis=1) * losses.shape[-1], 1)
            return jnp.sum(losses, axis=(1, 2)) / per_sample_denominator[:, 0]
        if reduction == "none":
            return jnp.mean(losses, axis=(1, 2))
        return jnp.mean(losses)

    def build_prefix_context(
        self,
        params: Params,
        images: Array,
        image_masks: Array,
        language_tokens: Array,
        language_masks: Array,
        state: Array,
    ) -> PrefixContext:
        prefix, pad_mask, attention_ar = self.embed_prefix(
            params, images, image_masks, language_tokens, language_masks, state
        )
        attention_mask = make_att_2d_masks(pad_mask, attention_ar)
        position_ids = jnp.cumsum(pad_mask, axis=1) - 1
        _, _, cache = self.transformer(
            params,
            prefix,
            None,
            attention_mask,
            position_ids,
            fill_cache=True,
        )
        return PrefixContext(pad_mask=pad_mask, cache=cache)

    def denoise_step(
        self,
        params: Params,
        context: PrefixContext,
        x_t: Array,
        timestep: Array,
    ) -> Array:
        suffix, suffix_pad, suffix_ar = self.embed_suffix(params, x_t, timestep)
        batch, suffix_length = suffix_pad.shape
        prefix_length = context.pad_mask.shape[1]
        prefix_mask = jnp.broadcast_to(context.pad_mask[:, None, :], (batch, suffix_length, prefix_length))
        suffix_mask = make_att_2d_masks(suffix_pad, suffix_ar)
        attention_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=-1)
        offsets = jnp.sum(context.pad_mask, axis=-1)[:, None]
        position_ids = offsets + jnp.cumsum(suffix_pad, axis=1) - 1
        _, suffix, _ = self.transformer(
            params,
            None,
            suffix,
            attention_mask,
            position_ids,
            cache=context.cache,
            fill_cache=False,
        )
        return self._linear(
            params,
            "model.action_out_proj",
            suffix[:, -self.config.chunk_size :].astype(jnp.float32),
            bias=True,
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
        noise: Array | None = None,
        num_steps: int | None = None,
        previous_chunk: Array | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> Array:
        batch = state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng,
                (batch, self.config.chunk_size, self.config.max_action_dim),
                dtype=jnp.float32,
            )
        context = self.build_prefix_context(
            params, images, image_masks, language_tokens, language_masks, state
        )
        steps = self.config.num_steps if num_steps is None else num_steps
        dt = -1.0 / steps

        def body(step: int, x_t: Array) -> Array:
            time = 1.0 + step * dt
            timestep = jnp.full((batch,), time, dtype=jnp.float32)

            def velocity_fn(value: Array) -> Array:
                return self.denoise_step(params, context, value, timestep)

            if self.config.rtc_config is not None and self.config.rtc_config.enabled:
                if inference_delay is None:
                    raise ValueError("RTC inference requires inference_delay")
                horizon = (
                    self.config.rtc_config.execution_horizon
                    if execution_horizon is None
                    else execution_horizon
                )
                velocity = rtc_guided_velocity(
                    velocity_fn,
                    x_t,
                    previous_chunk,
                    time=jnp.asarray(time, dtype=jnp.float32),
                    inference_delay=inference_delay,
                    execution_horizon=horizon,
                    config=self.config.rtc_config,
                )
            else:
                velocity = velocity_fn(x_t)
            return x_t + dt * velocity

        actions = jax.lax.fori_loop(0, steps, body, noise)
        return actions[..., : self.config.action_dim]
