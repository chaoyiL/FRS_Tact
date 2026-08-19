"""The pi0.5 runtime path must not drag in unrelated policy code.

The original version of this file only grepped a couple of runtime scripts for
`import lerobot.policies.smolvla_jax`. That check passed while the real hazard sat one level up:
`lerobot/policies/__init__.py` re-exported SmolVLA, so *any* `import lerobot.policies.pi05_jax`
executed it first and pulled in a whole second model stack -- on a branch whose jax/flax pins come
from openpi and were never expected to keep that stack importable.

SmolVLA is gone from this branch now, but the structural rule it exposed is what matters and is
what these tests pin: importing pi0.5 must not require the `lerobot.policies` package to import
any model.
"""

from pathlib import Path

from lerobot.datasets.dataset_sources import parse_dataset_sources, resolve_source_visual_keys

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_source_helpers_are_model_neutral() -> None:
    sources = parse_dataset_sources(
        {
            "datasets": [
                {
                    "repo_id": "org/demo",
                    "rename_map": {"observation.images.camera0": "left_wrist_0_rgb"},
                }
            ]
        }
    )

    assert sources[0].repo_id == "org/demo"
    assert resolve_source_visual_keys(
        ("left_wrist_0_rgb",),
        sources[0].rename_map,
        ("observation.images.camera0",),
    ) == ["observation.images.camera0"]


def test_policies_package_init_imports_no_model() -> None:
    """`import lerobot.policies.pi05_jax` runs this file first -- it must stay inert.

    A re-export here makes every pi0.5 entry point pay for (and depend on) an unrelated policy's
    import-time correctness. That is exactly how the SmolVLA re-export used to be able to take
    pi0.5 down before it ran a single line.
    """
    source = (ROOT / "src" / "lerobot" / "policies" / "__init__.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert "import" not in code, "lerobot/policies/__init__.py must not import any policy"


def test_pi05_runtime_path_imports_no_other_policy() -> None:
    """The shared (base-model-agnostic) stages must not reach into a policy package at all."""
    runtime_sources = (
        ROOT / "tools" / "precompute_tactile_embeddings.py",
        ROOT / "train_pi05_frs" / "utils" / "data.py",
        ROOT / "tools" / "train_frs.py",
        ROOT / "src" / "lerobot" / "datasets" / "sample_utils.py",
        ROOT / "src" / "lerobot" / "datasets" / "tactile_cache.py",
        ROOT / "utils" / "cache.py",
    )
    for path in runtime_sources:
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "lerobot.policies" not in stripped, f"{path}: {stripped}"
