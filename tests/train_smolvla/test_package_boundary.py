import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_visual_package_is_discoverable():

    assert importlib.util.find_spec("train_smolvla") is not None


def test_all_visual_production_modules_are_isolated():
    package_spec = importlib.util.find_spec("train_smolvla")
    assert package_spec is not None and package_spec.origin is not None
    package_dir = Path(package_spec.origin).parent
    expected = []
    for path in package_dir.rglob("*.py"):
        relative = path.relative_to(package_dir).with_suffix("")
        if "__init__" in relative.parts or {"train", "launcher"}.intersection(relative.parts):
            continue
        expected.append("train_smolvla." + ".".join(relative.parts))
    expected.sort()
    script = """
import importlib
import json
import pkgutil
import sys

package = importlib.import_module("train_smolvla")
modules = sorted(
    module.name
    for module in pkgutil.walk_packages(package.__path__, prefix="train_smolvla.")
    if module.name.rsplit(".", 1)[-1] not in {"train", "launcher"}
)
for module_name in modules:
    importlib.import_module(module_name)
for forbidden_prefix in ("tactile_encoder", "train_vtsmolvla"):
    loaded = sorted(
        name
        for name in sys.modules
        if name == forbidden_prefix or name.startswith(forbidden_prefix + ".")
    )
    assert not loaded, (forbidden_prefix, loaded)
print(json.dumps(modules))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == expected


def test_visual_production_sources_have_no_forbidden_coupling():
    package_spec = importlib.util.find_spec("train_smolvla")
    assert package_spec is not None and package_spec.origin is not None
    package_dir = Path(package_spec.origin).parent
    for path in package_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "lerobot.policies.smolvla_jax" not in source, path
        assert "from tactile_encoder" not in source, path
        assert "import tactile_encoder" not in source, path
        assert ".tactile_cache" not in source, path
        tree = ast.parse(source, filename=str(path))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"
                    )
                )
            ):
                imported_modules.add(node.args[0].value)
        assert not {
            module
            for module in imported_modules
            if module == "train_vtsmolvla" or module.startswith("train_vtsmolvla.")
        }, path


def test_visual_config_has_no_tactile_fields():
    from dataclasses import fields

    from train_smolvla import JaxSmolVLAConfig

    assert not {field.name for field in fields(JaxSmolVLAConfig) if "tactile" in field.name}
