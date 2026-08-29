import einops
import numpy as np
import torch
import tqdm
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.common.sampler import canonical_action_padding_value
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class RealImageTactileLatentDiffusionDataset(RealImageTactileDataset):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    @torch.no_grad()
    def get_normalizer(self, latent_batch_size=4096, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []
        action_dim = self.shape_meta['action']['shape'][0]
        action_data = np.asarray(self.replay_buffer['action'][:, :action_dim])
        indices = self.sampler.indices
        sequence_length = self.sampler.sequence_length

        progress = tqdm.tqdm(
            range(0, len(indices), latent_batch_size),
            leave=False,
            desc='Calculating latent action for normalizer',
            unit='batch')
        for batch_start in progress:
            batch_indices = indices[batch_start:batch_start + latent_batch_size]
            actions = np.empty(
                (len(batch_indices), sequence_length, action_dim),
                dtype=action_data.dtype)
            for batch_idx, (buffer_start, buffer_end, sample_start, sample_end) in enumerate(batch_indices):
                sample = action_data[buffer_start:buffer_end]
                actions[batch_idx, sample_start:sample_end] = sample
                if sample_start > 0:
                    prefix = sample[0]
                    if self.has_v2_action_contract:
                        prefix = canonical_action_padding_value(prefix)
                    actions[batch_idx, :sample_start] = prefix
                if sample_end < sequence_length:
                    suffix = sample[-1]
                    if self.has_v2_action_contract:
                        suffix = canonical_action_padding_value(suffix)
                    actions[batch_idx, sample_end:] = suffix

            action = torch.from_numpy(actions).to(self.at.device)
            action = normalizer['action'].normalize(action)
            latent_action = self.at.encoder(
                self.at.preprocess(action / self.at.act_scale)
            )
            if self.at.use_vq:
                if not self.use_latent_action_before_vq:
                    latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
            else:
                latent_action, _ = self.at.quant_state_without_vq(
                    latent_action,
                    sample=False,
                )
            if self.at.use_conv_encoder:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
            else:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
            latent_action_all.append(latent_action.cpu().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer
