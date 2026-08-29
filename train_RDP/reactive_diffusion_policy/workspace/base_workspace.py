from typing import Optional
import pathlib
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading

from reactive_diffusion_policy.common.artifact_manifest import (
    ArtifactManifest,
    normalizer_identity_digest,
)


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    def advance_training_state_for_resume(self):
        """Advance counters stored by end-of-epoch checkpoints.

        Training checkpoints are saved after the last optimizer step of an
        epoch, but before global_step and epoch are incremented. A resumed run
        must therefore start at the following step and epoch.
        """
        self.global_step = int(self.global_step) + 1
        self.epoch = int(self.epoch) + 1

    def get_remaining_epochs(self, target_num_epochs):
        """Return epochs left when target_num_epochs is a total target."""
        target_num_epochs = int(target_num_epochs)
        if target_num_epochs < 0:
            raise ValueError("target_num_epochs must be non-negative")
        return max(0, target_num_epochs - int(self.epoch))

    def should_save_checkpoint(self, checkpoint_every, local_epoch_idx, num_epochs_to_run):
        """Save on the configured cadence and after the final requested epoch."""
        checkpoint_every = int(checkpoint_every)
        local_epoch_idx = int(local_epoch_idx)
        num_epochs_to_run = int(num_epochs_to_run)
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if num_epochs_to_run < 1 or not 0 <= local_epoch_idx < num_epochs_to_run:
            raise ValueError("local_epoch_idx must identify an epoch in this run")
        return (
            int(self.epoch) % checkpoint_every == 0
            or local_epoch_idx == num_epochs_to_run - 1
        )

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def bind_checkpoint_artifacts(
        self,
        signature,
        *,
        normalizer,
        normalizer_path,
        role,
    ):
        """Bind the training inputs that this checkpoint is safe to use with."""
        if not pathlib.Path(normalizer_path).is_file():
            raise FileNotFoundError(f"normalizer cache does not exist: {normalizer_path}")
        manifest = ArtifactManifest.from_cache_signature(
            signature,
            normalizer_sha256=normalizer_identity_digest(normalizer),
            role=role,
        )
        OmegaConf.update(
            self.cfg,
            "artifacts",
            manifest.to_dict(),
            merge=False,
            force_add=True,
        )
        return manifest

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        artifacts = OmegaConf.select(self.cfg, "artifacts")
        if artifacts is not None:
            role = "LDP" if OmegaConf.select(self.cfg, "artifacts.at_sha256") else "AT"
            ArtifactManifest.from_dict(artifacts, role=role)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, weights_only=False, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill, weights_only=False)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill, weights_only=False)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
