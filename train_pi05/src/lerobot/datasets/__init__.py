"""Read-only LeRobot v3 dataset surface used by the Pi0.5 cache producer."""

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset


def check_video_encoder_parameters_pyav(*args, **kwargs):
    """Load the optional PyAV encoder checks only when video support is used."""

    from .pyav_utils import check_video_encoder_parameters_pyav as _check

    return _check(*args, **kwargs)


def detect_available_encoders_pyav(*args, **kwargs):
    """Load the optional PyAV encoder probe only when video support is used."""

    from .pyav_utils import detect_available_encoders_pyav as _detect

    return _detect(*args, **kwargs)

__all__ = [
    "CODEBASE_VERSION",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "check_video_encoder_parameters_pyav",
    "detect_available_encoders_pyav",
]
