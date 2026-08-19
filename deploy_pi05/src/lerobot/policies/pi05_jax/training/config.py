"""See _CONFIGS for the list of available configs.

Adapted from openpi's `src/openpi/training/config.py` (commit 15a9616). The `AssetsConfig` /
`DataConfig` / `GroupFactory` / `ModelTransformFactory` / `DataConfigFactory` / `TrainConfig`
scaffolding and the `cli()` / `get_config()` entry points are upstream's; what changed:

  * Upstream's concrete robot configs (Aloha/Libero/DROID) and the RLDS/DROID plumbing are gone,
    replaced by `LeRobotPickTubeDataConfig` -- this repo trains pi0.5 on the bimanual pick_tube
    datasets. `LeRobotPickTubeDataConfig` is modelled directly on upstream's
    `LeRobotLiberoDataConfig`.
  * `DataConfig` gains `sources` / `image_keys` / `video_backend`, because this repo trains on
    *several* LeRobot datasets concatenated together (four pick_tube captures), while upstream
    assumes exactly one `repo_id`. See `data_loader.create_torch_dataset`.
  * `TrainConfig` drops upstream's `pytorch_weight_path` / `pytorch_training_precision`: the
    PyTorch mirror model those feed is not vendored (see ../README.md).
"""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

from lerobot.datasets.dataset_sources import DatasetSource

from .. import download as _download
from .. import model as _model
from .. import normalize as _normalize
from .. import pi0_config
from .. import tokenizer as _tokenizer
from .. import transforms as _transforms
from ..policies import pick_tube_policy
from . import optimizer as _optimizer
from . import weight_loaders

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # --- Not in upstream openpi. ---
    # The LeRobot datasets to concatenate. Upstream trains on the single `repo_id` above; this
    # repo's pick_tube capture is split across four datasets that share a task and a robot, so
    # `data_loader.create_torch_dataset` builds one `LeRobotDataset` per entry and concatenates
    # them. `repo_id` above then only names the run (and, by default, the norm-stats asset id).
    sources: Sequence[DatasetSource] = ()
    # Dataset image keys (*after* each source's `rename_map` is applied) that the repack transform
    # reads. Used to restrict video decoding to the cameras the model actually consumes.
    image_keys: Sequence[str] = ()
    # Video decoding backend passed through to `LeRobotDataset`.
    video_backend: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                # Upstream builds a FAST tokenizer group here. pi0-FAST's model code
                # (`models/pi0_fast.py`) is not vendored, so this branch is unreachable.
                raise NotImplementedError("pi0-FAST is not vendored; see ../README.md")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPickTubeDataConfig(DataConfigFactory):
    """Data config for the bimanual pick_tube datasets.

    Same shape as upstream's `LeRobotLiberoDataConfig`: a repack transform that renames this
    dataset's keys into the policy's key space, a data transform pair from
    `policies/pick_tube_policy.py`, and the standard model transforms. The one structural
    addition is `sources`, because the capture is split across four LeRobot datasets.
    """

    # The datasets to concatenate. Each entry's `rename_map` is applied before `camera_map` is
    # resolved, so `camera_map` below is written in post-rename key space.
    sources: tyro.conf.Suppress[Sequence[DatasetSource]] = ()
    # pi0.5 image slot -> dataset image key (post-rename). Slots left out are zero-filled and
    # masked off by `PickTubeInputs` -- pick_tube has no third-person camera.
    camera_map: tyro.conf.Suppress[Mapping[str, str]] = dataclasses.field(
        default_factory=lambda: {
            "left_wrist_0_rgb": "observation.images.camera1",
            "right_wrist_0_rgb": "observation.images.camera2",
        }
    )
    # Real action/state width of the dataset, used by `PickTubeOutputs` to strip model padding.
    action_dim: int = 20
    # Video decoding backend passed through to `LeRobotDataset`.
    video_backend: str | None = "torchcodec"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        camera_map = dict(self.camera_map)
        # `data_loader.create_torch_dataset` renames each source's own action column to this key
        # before the repack runs, so the two must agree.
        action_key = (self.base_config or DataConfig()).action_sequence_keys[0]
        # The repack transform is *only* applied to data coming from the dataset. It maps the
        # LeRobot column names onto the keys `PickTubeInputs` reads.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": dict(camera_map),
                        "state": "observation.state",
                        "actions": action_key,
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[pick_tube_policy.PickTubeInputs(model_type=model_config.model_type)],
            outputs=[pick_tube_policy.PickTubeOutputs(action_dim=self.action_dim)],
        )

        # Model transforms include things like tokenizing the prompt and padding to action_dim.
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            sources=tuple(self.sources),
            image_keys=tuple(camera_map.values()),
            video_backend=self.video_backend,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# The four pick_tube captures. `rename_map` shifts the camera numbering by one (`camera0` is the
# left arm, `camera1` the right); `camera_map` in `LeRobotPickTubeDataConfig` is written against
# the post-rename names. See pi05_frs_plan.md.
_PICK_TUBE_RENAME_MAP = {
    "observation.images.camera0": "observation.images.camera1",
    "observation.images.camera1": "observation.images.camera2",
}

_PICK_TUBE_SOURCES = tuple(
    DatasetSource(
        repo_id=f"KaiyueChen/pick_tube_{index:02d}",
        root=f"/workspace/lerobot_v30/KaiyueChen/pick_tube_{index:02d}",
        action_key="actions",
        rename_map=_PICK_TUBE_RENAME_MAP,
    )
    for index in (1, 2, 3, 4)
)

# action_horizon=50 is not a placeholder -- it is what the official pi05_base checkpoint restores
# with (verified on the training server, see pi05_frs_plan.md). `get_freeze_filter()` derives the
# LoRA freeze filter from *this* model config, so the two can never drift apart.
_PICK_TUBE_LORA_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
    max_token_len=200,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # pi0.5 LoRA fine-tune on the four pick_tube datasets (the FRS base model on this branch).
    #
    TrainConfig(
        name="pi05_pick_tube",
        project_name="pick_tube",
        model=_PICK_TUBE_LORA_MODEL,
        data=LeRobotPickTubeDataConfig(
            repo_id="KaiyueChen/pick_tube",
            sources=_PICK_TUBE_SOURCES,
            assets=AssetsConfig(asset_id="pick_tube"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=_PICK_TUBE_LORA_MODEL.get_freeze_filter(),
        # LoRA fine-tuning disables EMA, matching openpi's own LoRA configs.
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        batch_size=32,
        num_workers=8,
        num_train_steps=40_000,
        log_interval=20,
        save_interval=2_000,
        fsdp_devices=2,
    ),
    #
    # Same data, full fine-tune (no LoRA): every weight trainable, EMA on.
    #
    TrainConfig(
        name="pi05_pick_tube_full",
        project_name="pick_tube",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
        ),
        data=LeRobotPickTubeDataConfig(
            repo_id="KaiyueChen/pick_tube",
            sources=_PICK_TUBE_SOURCES,
            assets=AssetsConfig(asset_id="pick_tube"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=0.999,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        batch_size=32,
        num_workers=8,
        num_train_steps=30_000,
        log_interval=20,
        save_interval=2_000,
        fsdp_devices=2,
    ),
    #
    # Smoke-test config: random data, no dataset or checkpoint needed.
    #
    TrainConfig(
        name="debug",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200),
        data=FakeDataConfig(),
        batch_size=2,
        num_workers=0,
        num_train_steps=10,
        log_interval=1,
        save_interval=5,
        wandb_enabled=False,
        overwrite=True,
        exp_name="debug",
    ),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
