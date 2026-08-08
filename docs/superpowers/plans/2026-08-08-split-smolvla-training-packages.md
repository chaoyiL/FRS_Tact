# Independent SmolVLA Training Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace lerobot.policies.smolvla_jax with a visual train_smolvla package and a tactile train_vtsmolvla extension package, each with YAML-only configuration and a one-command launcher.

**Architecture:** train_smolvla owns the visual JAX core. train_vtsmolvla imports it and adds tactile config, preprocessing, cache, fusion, checkpoint, validation, and training behavior. Old package and entrypoint paths are deleted.

**Tech Stack:** Python 3.12, JAX, Flax, Optax, LeRobot datasets, PyYAML, Bash, tmux, pytest, uv.

## Global Constraints

- Use top-level packages train_smolvla and train_vtsmolvla.
- Do not retain the old package or a compatibility shim.
- train_smolvla must not import tactile_encoder or train_vtsmolvla and must have no tactile config fields.
- train_vtsmolvla extends train_smolvla without duplicating visual core code.
- Both packages contain configs/train.yaml and scripts/train.sh.
- All adjustable training and launcher values live in YAML; CLI accepts only --config.
- Launchers never install, log in, or download datasets/checkpoints.
- Preserve checkpoint tensor names and old visual/VT checkpoint readability.
- Preserve unrelated user changes, especially the current deploy_smolvla restructuring and deleted files.
- Keep FRS-owned config and launcher assets in `train_frs/`; its launcher uses
  `python -m train_frs.<module>` for FRS stages while retaining shared SmolVLA merge and tactile
  embedding precomputation under `tools/`.

---

### Task 1: Establish package boundaries

**Files:**
- Create: tests/train_smolvla/test_package_boundary.py
- Create: tests/train_vtsmolvla/test_package_boundary.py
- Modify: pyproject.toml
- Create: train_smolvla/__init__.py
- Create: train_vtsmolvla/__init__.py

**Interfaces:**
- Produces importable top-level namespace packages before concrete implementations move in Tasks 2 and 3.
- Produces setuptools discovery rules for both package trees.

- [ ] **Step 1: Write failing import tests**

~~~python
def test_visual_package_is_discoverable():
    import importlib.util
    assert importlib.util.find_spec("train_smolvla") is not None


def test_vt_package_is_discoverable():
    import importlib.util
    assert importlib.util.find_spec("train_vtsmolvla") is not None
~~~

- [ ] **Step 2: Verify RED**

Run: .venv/bin/python -m pytest -q tests/train_smolvla/test_package_boundary.py tests/train_vtsmolvla/test_package_boundary.py

Expected: ModuleNotFoundError for the new package names.

- [ ] **Step 3: Add package discovery and empty namespaces**

Add train_smolvla* and train_vtsmolvla* to setuptools discovery. Create documented empty namespaces; concrete lazy exports are added with their implementations in Tasks 2 and 3.

- [ ] **Step 4: Commit**

~~~bash
git add pyproject.toml train_smolvla train_vtsmolvla tests/train_smolvla tests/train_vtsmolvla
git commit -m "refactor: establish SmolVLA package boundaries"
~~~

### Task 2: Move the visual core and remove tactile knowledge

**Files:**
- Create/modify from: src/lerobot/policies/smolvla_jax core modules into train_smolvla/
- Modify: visual tests under tests/jax/
- Modify: tests/train_smolvla/test_package_boundary.py

**Interfaces:**
- Produces visual-only implementations using current visual class/function names.
- Trainer passes a complete batch mapping to a neutral model loss hook.

- [ ] **Step 1: Add failing isolation tests**

~~~python
def test_visual_package_does_not_load_tactile_modules():
    import sys
    import train_smolvla
    assert "tactile_encoder" not in sys.modules
    assert "train_vtsmolvla" not in sys.modules


def test_visual_config_has_no_tactile_fields():
    from dataclasses import fields
    from train_smolvla import JaxSmolVLAConfig
    assert not {f.name for f in fields(JaxSmolVLAConfig) if "tactile" in f.name}
~~~

- [ ] **Step 2: Verify RED**

Run: .venv/bin/python -m pytest -q tests/train_smolvla/test_package_boundary.py

- [ ] **Step 3: Create the new modules and strip tactile branches**

Create the new visual modules from the current implementation, but retain the old package unchanged as a temporary migration source until Task 7. Remove tactile fields, encoder/cache imports, fusion initialization, tactile LoRA names, preprocessing, batch kwargs, and validation from the new package. Refactor the trainer call to:

~~~python
loss, metrics = self.model.compute_training_loss(params, batch=batch, rng=loss_rng)
~~~

- [ ] **Step 4: Verify GREEN**

~~~bash
.venv/bin/python -m pytest -q tests/train_smolvla/test_package_boundary.py tests/jax/test_functional.py tests/jax/test_checkpoint.py tests/jax/test_lora.py tests/jax/test_modality_dropout.py tests/jax/test_training.py
~~~

- [ ] **Step 5: Commit visual core**

~~~bash
git add train_smolvla src/lerobot/policies/smolvla_jax tests/jax tests/train_smolvla
git commit -m "refactor: move visual SmolVLA JAX core"
~~~

### Task 3: Extract tactile behavior into VT

**Files:**
- Create: train_vtsmolvla/checkpoint.py, configuration.py, data.py, lora.py, modeling.py, policy.py, preprocessing.py, tactile_cache.py, training.py, validation.py
- Modify: tests/train_vtsmolvla/ and tactile tests under tests/jax/

**Interfaces:**
- Produces VTSmolVLAConfig, VTJaxSmolVLA, VTJaxSmolVLAPolicy, VTJaxSmolVLATrainer, VTLeRobotJaxDataLoader.
- Preserves model.tactile_encoder.* and model.tactile_proj.* keys.

- [ ] **Step 1: Write failing extension test**

~~~python
def test_vt_config_extends_visual_config():
    from train_smolvla import JaxSmolVLAConfig
    from train_vtsmolvla import VTSmolVLAConfig
    cfg = VTSmolVLAConfig(tactile_keys=("observation.images.tactile",), tactile_num_tokens=1)
    assert isinstance(cfg, JaxSmolVLAConfig)
~~~

- [ ] **Step 2: Verify RED**

Run: .venv/bin/python -m pytest -q tests/train_vtsmolvla tests/jax/test_tactile_cache.py tests/jax/test_tactile_integration.py

- [ ] **Step 3: Implement VT extensions**

Move tactile_cache.py. Add the frozen VT config subclass, data/preprocessing wrappers, tactile fusion model, checkpoint initializer, LoRA classification, VT contract, policy, and trainer hooks by importing visual primitives.

- [ ] **Step 4: Verify GREEN with visual isolation**

~~~bash
.venv/bin/python -m pytest -q tests/train_smolvla tests/train_vtsmolvla tests/jax/test_tactile_cache.py tests/jax/test_tactile_integration.py tests/jax/test_data.py tests/jax/test_checkpoint_validation.py
~~~

- [ ] **Step 5: Commit VT extension**

~~~bash
git add train_smolvla train_vtsmolvla tests/train_vtsmolvla tests/jax
git commit -m "refactor: extract VT SmolVLA extensions"
~~~

### Task 4: Move entrypoints and YAML

**Files:**
- Move/modify: tools/train_smolvla_jax.py to train_smolvla/train.py
- Move/rewrite: tools/train_vtsmolvla_jax.py to train_vtsmolvla/train.py
- Move: configs/train_smolvla_jax.yaml to train_smolvla/configs/train.yaml
- Move: configs/train_vtsmolvla_jax.yaml to train_vtsmolvla/configs/train.yaml
- Create: tests/train_smolvla/test_train_entrypoint.py
- Create: tests/train_vtsmolvla/test_train_entrypoint.py
- Modify: tests/jax/test_train_script.py

**Interfaces:**
- Produces python -m train_smolvla.train --config PATH.
- Produces python -m train_vtsmolvla.train --config PATH.
- CLI exposes only --config.

- [ ] **Step 1: Write failing CLI/YAML tests**

~~~python
def test_visual_cli_has_no_parameter_overrides(run_module_help):
    result = run_module_help("train_smolvla.train")
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--batch-size" not in result.stdout


def test_visual_yaml_has_no_tactile_settings(load_training_yaml):
    cfg = load_training_yaml("train_smolvla/configs/train.yaml")
    assert "tactile_embedding_cache" not in cfg
    assert not any("tactile" in key for key in cfg.get("model", {}))
~~~

- [ ] **Step 2: Verify RED and move entrypoints**

Shared orchestration stays in train_smolvla.train behind components. VT supplies VT factories and does not mutate sys.argv or import a script by filename.

- [ ] **Step 3: Normalize YAML**

Put checkpoint, datasets, output, steps, batching, workers, logging, validation, transforms, dropout, W&B, model, resume, and launcher values in YAML. Remove tactile-named visual examples.

- [ ] **Step 4: Verify and commit**

~~~bash
.venv/bin/python -m pytest -q tests/train_smolvla/test_train_entrypoint.py tests/train_vtsmolvla/test_train_entrypoint.py tests/jax/test_train_script.py
git add train_smolvla train_vtsmolvla tools configs tests
git commit -m "refactor: colocate SmolVLA training entrypoints"
~~~

### Task 5: Add one-command launchers

**Files:**
- Create: train_smolvla/launcher.py, scripts/train.sh, README.md
- Create: train_vtsmolvla/launcher.py, scripts/train.sh, README.md
- Delete: scripts/start_vtsmolvla_train.sh and train_for_agent.md
- Modify: scripts/setup_env.sh
- Create: tests/train_smolvla/test_launcher.py
- Create: tests/train_vtsmolvla/test_launcher.py

**Interfaces:**
- Produces bash train_smolvla/scripts/train.sh.
- Produces bash train_vtsmolvla/scripts/train.sh.
- Consumes YAML launcher settings for tmux, session, foreground, temp, and logs.

- [ ] **Step 1: Write failing launcher tests**

Cover root discovery from another cwd, YAML parsing, missing inputs, GPU preflight, tmux construction, overwrite protection, and no training constants in Shell.

- [ ] **Step 2: Verify RED**

Run: .venv/bin/python -m pytest -q tests/train_smolvla/test_launcher.py tests/train_vtsmolvla/test_launcher.py

- [ ] **Step 3: Implement thin wrappers**

~~~bash
exec "$UV_BIN" run --no-sync python -m train_smolvla.launcher \
  --config "$PROJECT_ROOT/train_smolvla/configs/train.yaml"
~~~

The VT wrapper uses its VT module and config.

- [ ] **Step 4: Implement launchers and README files**

Launchers read YAML, preflight GPU/inputs/resume/output, create tmux, and write logs. VT optionally precomputes tactile embeddings.

- [ ] **Step 5: Verify and commit**

~~~bash
bash -n train_smolvla/scripts/train.sh
bash -n train_vtsmolvla/scripts/train.sh
.venv/bin/python -m pytest -q tests/train_smolvla/test_launcher.py tests/train_vtsmolvla/test_launcher.py
git add train_smolvla train_vtsmolvla scripts train_for_agent.md tests
git commit -m "feat: add one-command SmolVLA launchers"
~~~

### Task 6: Migrate all consumers

**Files:**
- Modify: src/lerobot/policies/__init__.py
- Modify: train_frs/prepare.py, utils/source_model.py, train_frs/utils/data.py, modalities_eval/utils.py
- Modify: affected tools/*.py
- Modify carefully: deploy_smolvla/remote_client.py and only old references in the user's current deploy_smolvla layout
- Modify: affected JAX, FRS, publish, deploy tests
- Create: tests/train_smolvla/test_no_legacy_imports.py

**Interfaces:**
- Produces zero active-code imports of lerobot.policies.smolvla_jax.

- [ ] **Step 1: Write failing legacy-reference test**

~~~python
def test_repository_has_no_legacy_smolvla_imports(repository_python_files):
    offenders = [p for p in repository_python_files if "lerobot.policies.smolvla_jax" in p.read_text()]
    assert offenders == []
~~~

- [ ] **Step 2: Verify RED**

Run: .venv/bin/python -m pytest -q tests/train_smolvla/test_no_legacy_imports.py

- [ ] **Step 3: Update imports by responsibility**

Visual utilities import train_smolvla; tactile cache/policy consumers import train_vtsmolvla; FRS uses VT only for tactile cache. Deployment selects visual or VT policy from YAML.

- [ ] **Step 4: Verify downstream behavior**

~~~bash
.venv/bin/python -m pytest -q tests/flow_decoder/test_frs_safety.py tests/jax/test_publish_checkpoint.py tests/jax/test_deploy_launcher.py tests/train_smolvla/test_no_legacy_imports.py
~~~

Do not modify a pre-existing failure caused by the user's uncommitted deploy config.

- [ ] **Step 5: Commit**

~~~bash
git add src/lerobot/policies/__init__.py train_frs/prepare.py utils train_frs modalities_eval tools deploy_smolvla tests
git commit -m "refactor: migrate SmolVLA consumers"
~~~

### Task 7: Delete legacy paths and verify

**Files:**
- Delete: src/lerobot/policies/smolvla_jax/
- Delete: old training tools, configs, and launcher
- Modify: final path references and package metadata

- [ ] **Step 1: Add deletion assertions**

~~~python
def test_legacy_paths_are_removed(repository_root):
    assert not (repository_root / "src/lerobot/policies/smolvla_jax").exists()
    assert not (repository_root / "tools/train_smolvla_jax.py").exists()
    assert not (repository_root / "tools/train_vtsmolvla_jax.py").exists()
    assert not (repository_root / "configs/train_smolvla_jax.yaml").exists()
    assert not (repository_root / "configs/train_vtsmolvla_jax.yaml").exists()
~~~

- [ ] **Step 2: Remove only legacy targets and fix last references**

Do not delete or restore unrelated user files shown by baseline git status.

- [ ] **Step 3: Run focused verification**

~~~bash
.venv/bin/python -m pytest -q tests/train_smolvla tests/train_vtsmolvla
bash -n train_smolvla/scripts/train.sh
bash -n train_vtsmolvla/scripts/train.sh
.venv/bin/python -m train_smolvla.train --help
.venv/bin/python -m train_vtsmolvla.train --help
~~~

- [ ] **Step 4: Run shared regressions**

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q tests/jax tests/flow_decoder tests/datasets
~~~

- [ ] **Step 5: Run static and packaging checks**

~~~bash
rg -n 'lerobot\.policies\.smolvla_jax|tools/train_smolvla_jax|tools/train_vtsmolvla_jax|configs/train_smolvla_jax|configs/train_vtsmolvla_jax|scripts/start_vtsmolvla_train' --glob '!docs/superpowers/**' --glob '!.git/**' .
git diff --check
uv build
~~~

Expected: no active legacy matches, clean whitespace, and both packages in the wheel.

- [ ] **Step 6: Commit final cleanup**

~~~bash
git add -A train_smolvla train_vtsmolvla src/lerobot/policies tools configs scripts tests pyproject.toml
git commit -m "refactor: complete independent SmolVLA migration"
~~~
