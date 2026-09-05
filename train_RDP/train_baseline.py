"""Train the original RDP baseline using the local single-right dataset."""
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

os.environ.setdefault('RDP_BASELINE_REPO_ROOT', str(Path(__file__).resolve().parents[1]))
OmegaConf.register_new_resolver('eval', eval, replace=True)


@hydra.main(version_base=None, config_path='rdp_baseline/config', config_name='train_at')
def main(cfg):
    OmegaConf.resolve(cfg)
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
