from reactive_diffusion_policy.model.vae.model import VAE


def _vae(**kwargs):
    return VAE(
        horizon=1,
        shape_meta={"action": {"shape": [10]}, "extended_obs": {}},
        n_latent_dims=4,
        mlp_layer_num=0,
        n_embed=2,
        use_vq=False,
        eval=False,
        device="cpu",
        **kwargs,
    )


def test_vae_maps_explicit_micro_motion_weight_to_physical_weights():
    vae = _vae(micro_motion_weight=1.0)

    assert vae.physical_loss_weights["micro_motion_weight"] == 1.0


def test_vae_defaults_micro_motion_weight_to_zero():
    vae = _vae()

    assert vae.physical_loss_weights["micro_motion_weight"] == 0.0
