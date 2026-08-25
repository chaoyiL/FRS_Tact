import numpy as np
import pytest

from deploy_deco.policy import DECOPolicy


def test_uint8_image_is_converted_to_unit_float():
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    converted = DECOPolicy._image(image, "camera")
    assert converted.dtype == np.float32
    assert np.all(converted == 1.0)


def test_out_of_range_float_image_is_rejected():
    image = np.full((4, 5, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        DECOPolicy._image(image, "camera")
