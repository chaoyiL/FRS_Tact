"""pick_tube (bimanual) input/output transforms, written against openpi's policy-transform contract.

Structured exactly like upstream's `src/openpi/policies/libero_policy.py`: an `Inputs` transform
that converts one dataset sample into the model's `{"image", "image_mask", "state", "actions",
"prompt"}` format, and an `Outputs` transform that slices the model's padded action vector back
down to the robot's real action dimension.

Dataset facts this encodes (see pi05_frs_plan.md for how they were established from
`KaiyueChen/pick_tube_0X`'s `meta/info.json`):

  * `robot_type: "bimanual"`, two RGB cameras only -- `camera0` is the **left** arm's wrist
    camera, `camera1` the right arm's. There is no third-person view, so pi0.5's `base_0_rgb`
    slot is filled with zeros and masked off, the same way `LiberoInputs` masks off its unused
    `right_wrist_0_rgb`.
  * `observation.state` and `actions` are both 20-dimensional; the model pads them to
    `action_dim` (32) later, in `transforms.PadStatesAndActions`.
"""

import dataclasses
from collections.abc import Mapping

import einops
import numpy as np

from .. import transforms
from ..model import IMAGE_KEYS, ModelType


def _parse_image(image) -> np.ndarray:
    """Dataset frame -> uint8 HWC, the layout pi0.5's SigLIP tower expects.

    Byte-for-byte the same helper as upstream's `libero_policy._parse_image`, and for the same
    reason: LeRobot stores images as float32 CHW in [0, 1], while the model wants HWC. (Getting
    this wrong is silent -- the model runs fine and produces garbage features.)
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PickTubeInputs(transforms.DataTransformFn):
    """Convert a repacked pick_tube sample into pi0.5's model input format.

    Runs on the output of the config's `repack_transforms`, which has already mapped this
    dataset's keys into `{"image": {<pi0.5 slot>: <frame>}, "state": ..., "actions": ...,
    "prompt": ...}`. Any of pi0.5's three image slots the repack did not fill is zero-filled and
    masked off here.
    """

    # Determines which model will be used.
    model_type: ModelType

    def __call__(self, data: dict) -> dict:
        images: Mapping = data["image"]
        parsed = {key: _parse_image(value) for key, value in images.items()}
        if not parsed:
            raise ValueError(f"repack produced no images; expected a subset of {IMAGE_KEYS}")
        unknown = set(parsed) - set(IMAGE_KEYS)
        if unknown:
            raise ValueError(f"image keys must be a subset of {IMAGE_KEYS}, got extra {sorted(unknown)}")

        # Shape template for the missing views. All pick_tube cameras share a resolution, and
        # `transforms.ResizeImages` normalizes them to 224x224 downstream regardless.
        template = next(iter(parsed.values()))
        # pi0-FAST does not mask padding images; pi0/pi0.5 do. Same rule as `LiberoInputs`.
        pad_mask = np.True_ if self.model_type == ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["state"],
            "image": {key: parsed.get(key, np.zeros_like(template)) for key in IMAGE_KEYS},
            "image_mask": {key: np.True_ if key in parsed else pad_mask for key in IMAGE_KEYS},
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PickTubeOutputs(transforms.DataTransformFn):
    """Slice the model's padded actions back down to the robot's real action dimension.

    Inference-only, like every `*Outputs` transform in openpi.
    """

    # Real action dimension of the dataset (pick_tube: 20). Everything above this index is the
    # zero padding added by `transforms.PadStatesAndActions`.
    action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
