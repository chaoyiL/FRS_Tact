# SmolVLA Image Input Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one offline PNG comparing SmolVLA training and saved real-deployment images.

**Architecture:** Add one small command-line visualization tool that reuses the validated parquet and saved-observation loaders from `tools/analyze_smolvla_online_run.py`. It selects six evenly spaced frames per source, renders four camera rows, and computes full-dataset RGB and luminance histograms.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, Pillow, pytest

## Global Constraints

- Read `data/pick_two_tube` and `eval_obs_20260901_011323` offline only.
- Preserve source RGB values; do not apply augmentation, normalization, or exposure correction.
- Do not start or connect to the robot service.
- Write the final PNG to `outputs/smolvla_online_diagnosis/eval_obs_20260901_011323/image_input_comparison.png`.

---

### Task 1: Generate the comparison figure

**Files:**
- Create: `tools/visualize_smolvla_image_inputs.py`
- Create: `tests/test_visualize_smolvla_image_inputs.py`
- Generate: `outputs/smolvla_online_diagnosis/eval_obs_20260901_011323/image_input_comparison.png`

**Interfaces:**
- Consumes: `load_training_parquets(Path) -> TrainingCorpus` and `load_saved_observations(Path) -> list[SavedObservation]`
- Produces: `select_even_indices(length: int, count: int) -> np.ndarray`
- Produces: `write_image_input_comparison(training_root: Path, obs_dir: Path, output: Path, sample_count: int = 6) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
def test_select_even_indices_includes_endpoints():
    assert module.select_even_indices(13, 6).tolist() == [0, 2, 5, 7, 10, 12]


def test_write_comparison_creates_readable_png(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_training_parquets", lambda _: fake_training())
    monkeypatch.setattr(module, "load_saved_observations", lambda _: fake_observations())
    result = module.write_image_input_comparison(
        tmp_path / "training", tmp_path / "obs", tmp_path / "comparison.png", sample_count=2
    )
    with Image.open(result) as image:
        assert image.format == "PNG"
        assert image.width > image.height > 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache \
  uv run --no-sync pytest -q tests/test_visualize_smolvla_image_inputs.py
```

Expected: failure because `tools/visualize_smolvla_image_inputs.py` does not exist.

- [ ] **Step 3: Implement the minimal visualization**

```python
def select_even_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.linspace(0, length - 1, min(length, count), dtype=int)


def write_image_input_comparison(training_root, obs_dir, output, sample_count=6):
    training = load_training_parquets(Path(training_root))
    saved = load_saved_observations(Path(obs_dir))
    # Render four sampled camera rows followed by full-dataset RGB and
    # luminance histogram axes. Titles include source, camera, frame/step,
    # resolution, and mean luminance.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return Path(output)
```

The CLI accepts required `--training-root`, `--obs-dir`, and `--output` arguments plus optional `--sample-count` defaulting to 6.

- [ ] **Step 4: Run focused tests and syntax validation**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache \
  uv run --no-sync pytest -q tests/test_visualize_smolvla_image_inputs.py
python -m py_compile tools/visualize_smolvla_image_inputs.py
```

Expected: all tests pass and `py_compile` exits 0.

- [ ] **Step 5: Generate and inspect the real artifact**

Run:

```bash
PYTHONPATH=.:src:tests UV_CACHE_DIR=/tmp/frs_tact_uv_cache uv run --no-sync \
  python tools/visualize_smolvla_image_inputs.py \
  --training-root data/pick_two_tube \
  --obs-dir /home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260901_011323 \
  --output outputs/smolvla_online_diagnosis/eval_obs_20260901_011323/image_input_comparison.png
```

Expected: a nonempty PNG with four correctly labelled RGB image rows and two histogram panels. Open it once for visual inspection.

- [ ] **Step 6: Commit the tool and test**

```bash
git add tools/visualize_smolvla_image_inputs.py tests/test_visualize_smolvla_image_inputs.py
git commit -m "feat: visualize SmolVLA image inputs"
```
