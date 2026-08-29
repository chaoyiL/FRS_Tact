from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dependency_names(project: Path) -> set[str]:
    config = tomllib.loads(project.read_text(encoding="utf-8"))
    names = set()
    for requirement in config["project"]["dependencies"]:
        name = requirement.split(";", 1)[0]
        for delimiter in ("[", "<", ">", "=", "!", "~"):
            name = name.split(delimiter, 1)[0]
        names.add(name.strip().lower())
    return names


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


def test_pi05_training_environment_has_no_video_or_device_stack() -> None:
    dependencies = _dependency_names(ROOT / "train_pi05" / "pyproject.toml")
    assert dependencies.isdisjoint(
        {
            "av",
            "draccus",
            "msgpack",
            "nvidia-ml-py",
            "opencv-python",
            "pytest",
            "safetensors",
            "torchcodec",
            "torchvision",
            "websockets",
        }
    )


def test_image_only_conversion_environment_has_no_ml_or_video_stack() -> None:
    dependencies = _dependency_names(ROOT / "data_tools" / "pyproject.toml")
    assert dependencies.isdisjoint(
        {
            "av",
            "draccus",
            "einops",
            "opencv-python-headless",
            "safetensors",
            "termcolor",
            "torch",
            "torchvision",
        }
    )


def test_image_training_import_surface_does_not_eagerly_import_video_modules() -> None:
    dataset_root = ROOT / "train_pi05" / "src" / "lerobot" / "datasets"
    for relative_path in (
        "__init__.py",
        "dataset_metadata.py",
        "dataset_reader.py",
        "lerobot_dataset.py",
    ):
        imports = _top_level_imports(dataset_root / relative_path)
        assert imports.isdisjoint({"av", "torchcodec", "torchvision"}), relative_path


def test_converter_does_not_eagerly_import_video_stack() -> None:
    converter = ROOT / "lerobot" / "datasets" / "v30" / "convert_dataset_v21_to_v30.py"
    source = converter.read_text(encoding="utf-8")
    top_level_imports = _top_level_imports(converter)
    assert top_level_imports.isdisjoint({"av", "torch", "torchvision"})
    assert "from lerobot.datasets.video_utils import" not in "\n".join(
        line for line in source.splitlines() if not line.startswith("    ")
    )


def test_data_tools_package_initializers_do_not_eagerly_import_torch() -> None:
    for relative_path in ("lerobot/utils/__init__.py", "lerobot/datasets/__init__.py"):
        imports = _top_level_imports(ROOT / relative_path)
        assert "torch" not in imports, relative_path


def test_private_lerobot_utility_parses_as_python_311() -> None:
    source = (ROOT / "train_pi05" / "src" / "lerobot" / "utils" / "io_utils.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source, feature_version=(3, 11))


def test_jax_model_import_does_not_eagerly_load_pytorch_model_or_pytest() -> None:
    model_path = ROOT / "train_pi05" / "src" / "openpi" / "models" / "model.py"
    imports = _top_level_imports(model_path)
    assert imports.isdisjoint({"pytest", "safetensors"})

    gemma_pytorch = (
        ROOT / "train_pi05" / "src" / "openpi" / "models_pytorch" / "gemma_pytorch.py"
    ).read_text(encoding="utf-8")
    assert "import pytest" not in gemma_pytorch
    assert "pytest.Cache" not in gemma_pytorch


def test_smoke_training_schedule_is_valid_for_two_steps() -> None:
    train_path = ROOT / "train_pi05" / "train.py"
    spec = importlib.util.spec_from_file_location("pi05_yaml_train", train_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._fit_schedule_steps(2, 1_000) == (1, 2)
    assert module._fit_schedule_steps(30_000, 1_000) == (1_000, 30_000)
