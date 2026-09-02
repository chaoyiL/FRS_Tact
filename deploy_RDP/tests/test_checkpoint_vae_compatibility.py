import hydra
from omegaconf import OmegaConf


def test_checkpoint_policy_at_hydra_instantiates_with_micro_motion_weight():
    checkpoint_cfg = OmegaConf.create(
        {
            "policy": {
                "at": {
                    "_target_": "reactive_diffusion_policy.model.vae.model.VAE",
                    "horizon": 1,
                    "shape_meta": {"action": {"shape": [10]}, "extended_obs": {}},
                    "n_latent_dims": 4,
                    "mlp_layer_num": 0,
                    "n_embed": 2,
                    "use_vq": False,
                    "eval": False,
                    "device": "cpu",
                    "micro_motion_weight": 1.0,
                }
            }
        }
    )

    vae = hydra.utils.instantiate(checkpoint_cfg.policy.at)

    assert vae.physical_loss_weights["micro_motion_weight"] == 1.0

    legacy_cfg = OmegaConf.create(OmegaConf.to_container(checkpoint_cfg, resolve=True))
    del legacy_cfg.policy.at.micro_motion_weight
    legacy_vae = hydra.utils.instantiate(legacy_cfg.policy.at)
    assert legacy_vae.physical_loss_weights["micro_motion_weight"] == 0.0
    legacy_state = legacy_vae.state_dict()
    weighted_state = vae.state_dict()
    assert legacy_state.keys() == weighted_state.keys()
    for component, component_state in legacy_state.items():
        assert component_state.keys() == weighted_state[component].keys()
        for name, value in component_state.items():
            assert value.shape == weighted_state[component][name].shape
    vae.load_state_dict(legacy_state)
