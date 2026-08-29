import inspect
import math

import pytest

from reactive_diffusion_policy.workspace import train_at_workspace
from reactive_diffusion_policy.workspace import train_diffusion_unet_image_workspace


class _Counter:
    def __init__(self):
        self.count = 0

    def step(self, *args):
        self.count += 1


@pytest.mark.parametrize(
    ("num_batches", "accumulate_every"),
    [(7, 1), (7, 2), (7, 3), (6, 3)],
)
def test_optimizer_scheduler_and_ema_step_once_per_complete_or_partial_group(
    num_batches,
    accumulate_every,
):
    optimizer = _Counter()
    scheduler = _Counter()
    ema = _Counter()

    for batch_idx in range(num_batches):
        if train_at_workspace.should_optimizer_step(
            batch_idx,
            num_batches,
            accumulate_every,
        ):
            optimizer.step()
            scheduler.step()
            ema.step(object())

    expected = math.ceil(num_batches / accumulate_every)
    assert optimizer.count == expected
    assert scheduler.count == expected
    assert ema.count == expected


def test_at_and_ldp_workspaces_share_the_accumulation_boundary_helper():
    assert (
        train_diffusion_unet_image_workspace.should_optimizer_step
        is train_at_workspace.should_optimizer_step
    )


@pytest.mark.parametrize(
    (
        "num_batches",
        "max_train_steps",
        "accumulate_every",
        "num_epochs",
        "expected",
    ),
    [
        (7, None, 2, 3, 12),
        (7, 5, 2, 3, 9),
        (7, 5, 3, 2, 4),
        (7, 1, 3, 4, 4),
    ],
)
def test_scheduler_steps_use_effective_batches_and_include_partial_group(
    num_batches,
    max_train_steps,
    accumulate_every,
    num_epochs,
    expected,
):
    assert train_at_workspace.get_num_training_steps(
        num_batches=num_batches,
        max_train_steps=max_train_steps,
        accumulate_every=accumulate_every,
        num_epochs=num_epochs,
    ) == expected


def test_at_and_ldp_workspaces_share_scheduler_step_count_helper():
    assert (
        train_diffusion_unet_image_workspace.get_num_training_steps
        is train_at_workspace.get_num_training_steps
    )


@pytest.mark.parametrize(
    "workspace_class",
    [
        train_at_workspace.TrainATWorkspace,
        train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace,
    ],
)
def test_workspaces_use_effective_scheduler_step_count(workspace_class):
    source = inspect.getsource(workspace_class.run)

    assert "num_training_steps=get_num_training_steps(" in source


@pytest.mark.parametrize(
    (
        "num_batches",
        "max_train_steps",
        "accumulate_every",
        "completed_epochs",
        "expected",
    ),
    [
        (7, None, 3, 4, 12),
        (7, 5, 3, 4, 8),
        (7, 5, 2, 3, 9),
    ],
)
def test_legacy_resume_optimizer_step_uses_effective_capped_batches(
    num_batches,
    max_train_steps,
    accumulate_every,
    completed_epochs,
    expected,
):
    assert train_at_workspace.get_legacy_optimizer_step(
        completed_epochs=completed_epochs,
        num_batches=num_batches,
        max_train_steps=max_train_steps,
        accumulate_every=accumulate_every,
    ) == expected


def test_at_and_ldp_workspaces_share_legacy_optimizer_step_helper():
    assert (
        train_diffusion_unet_image_workspace.get_legacy_optimizer_step
        is train_at_workspace.get_legacy_optimizer_step
    )


@pytest.mark.parametrize(
    "workspace_class",
    [
        train_at_workspace.TrainATWorkspace,
        train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace,
    ],
)
def test_workspaces_use_effective_legacy_optimizer_step(workspace_class):
    source = inspect.getsource(workspace_class.run)

    assert "self.optimizer_step = get_legacy_optimizer_step(" in source


def test_workspaces_persist_optimizer_step_separately_from_batch_logging_step():
    assert "global_step" in train_at_workspace.TrainATWorkspace.include_keys
    assert "optimizer_step" in train_at_workspace.TrainATWorkspace.include_keys
    assert (
        "global_step"
        in train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace.include_keys
    )
    assert (
        "optimizer_step"
        in train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace.include_keys
    )


@pytest.mark.parametrize("accumulate_every", [0, -1])
def test_accumulation_boundary_rejects_nonpositive_interval(accumulate_every):
    with pytest.raises(ValueError, match="positive"):
        train_at_workspace.should_optimizer_step(0, 1, accumulate_every)
