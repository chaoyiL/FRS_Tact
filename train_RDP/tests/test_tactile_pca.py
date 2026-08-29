import numpy as np
import pytest
import torch

from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA
from reactive_diffusion_policy.model.tactile_pca import group_tactile_embeddings


def sensor_values() -> np.ndarray:
    return np.stack(
        [np.full(512, sensor_index, dtype=np.float32) for sensor_index in range(4)]
    )


def test_group_tactile_embeddings_groups_both_fingers_of_each_robot() -> None:
    grouped = group_tactile_embeddings(sensor_values())

    assert grouped.shape == (2, 1024)
    np.testing.assert_array_equal(grouped[0, :512], 0.0)
    np.testing.assert_array_equal(grouped[0, 512:], 1.0)
    np.testing.assert_array_equal(grouped[1, :512], 2.0)
    np.testing.assert_array_equal(grouped[1, 512:], 3.0)


def test_torch_projection_uses_the_same_robot_grouping() -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 15, 1024), dtype=np.float32)
    components[0, 0, 0] = 1.0
    components[0, 1, 512] = 1.0
    components[1, 0, 0] = 1.0
    components[1, 1, 512] = 1.0
    model = BimanualTactilePCA(means, components)

    projected = model(torch.from_numpy(sensor_values())).detach().numpy()

    assert projected.shape == (30,)
    np.testing.assert_array_equal(projected[[0, 1, 15, 16]], [0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("components_per_arm", [8, 15, 30])
def test_projection_dimension_is_inferred_from_components(
    components_per_arm: int,
) -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, components_per_arm, 1024), dtype=np.float32)
    model = BimanualTactilePCA(means, components)

    assert model.components_per_arm == components_per_arm
    assert model.output_dim == components_per_arm * 2
    assert model(torch.from_numpy(sensor_values())).shape == (model.output_dim,)
    assert model.transform_numpy(sensor_values()).shape == (model.output_dim,)
    flat_batch = np.stack([sensor_values().reshape(-1)] * 2)
    assert model(torch.from_numpy(flat_batch)).shape == (2, model.output_dim)
    assert model.transform_numpy(flat_batch).shape == (2, model.output_dim)


@pytest.mark.parametrize("field", ["means", "components"])
def test_pca_rejects_non_finite_values(field: str) -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 8, 1024), dtype=np.float32)
    if field == "means":
        means[0, 0] = np.nan
    else:
        components[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match=f"PCA {field} must contain only finite values"):
        BimanualTactilePCA(means, components)


@pytest.mark.parametrize("field", ["means", "components"])
@pytest.mark.parametrize("dtype", [np.complex64, object])
def test_pca_rejects_complex_values_before_float32_conversion(field: str, dtype) -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 8, 1024), dtype=np.float32)
    if field == "means":
        means = means.astype(dtype)
        means[0, 0] = 1.0 + np.inf * 1j
    else:
        components = components.astype(dtype)
        components[0, 0, 0] = 1.0 + np.inf * 1j

    with pytest.raises(ValueError, match=f"PCA {field} must contain real values"):
        BimanualTactilePCA(means, components)
