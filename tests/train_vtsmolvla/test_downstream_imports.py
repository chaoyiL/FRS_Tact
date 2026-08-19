import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "train_smolvla_frs/utils/data.py",
    "tools/precompute_tactile_embeddings.py",
    "tools/convert_smolvla_pt_to_jax.py",
    "modalities_eval/utils.py",
    "utils/source_model.py",
)


def test_simple_consumers_do_not_import_the_legacy_mixed_package():
    for relative in FILES:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not {
            module
            for module in modules
            if module == "lerobot.policies.smolvla_jax"
            or module.startswith("lerobot.policies.smolvla_jax.")
        }, path
