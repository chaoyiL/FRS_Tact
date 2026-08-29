from __future__ import annotations

import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train_RDP"
DEPLOY = ROOT / "deploy_RDP"
REVISION = "RDP_vitamin 7a5bc24 branch agent/rdp-pick-tube-deployment"


def project_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


class RDPProjectBoundaryTest(unittest.TestCase):
    def test_projects_record_new_source_revision(self) -> None:
        for project in (TRAIN, DEPLOY):
            self.assertEqual(
                (project / "SOURCE_REVISION").read_text(encoding="utf-8").strip(),
                REVISION,
            )

    def test_projects_have_isolated_dependency_contracts(self) -> None:
        train_project = tomllib.loads((TRAIN / "pyproject.toml").read_text(encoding="utf-8"))
        deploy_project = tomllib.loads((DEPLOY / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(train_project["project"]["name"], "rdp-training")
        self.assertEqual(deploy_project["project"]["name"], "rdp-deployment")
        self.assertIn("numpy==1.26.4", train_project["project"]["dependencies"])
        self.assertIn("websockets==16.0", deploy_project["project"]["dependencies"])
        self.assertNotIn("websockets==16.0", train_project["project"]["dependencies"])

    def test_shared_rdp_package_is_identical(self) -> None:
        train_package = TRAIN / "reactive_diffusion_policy"
        deploy_package = DEPLOY / "reactive_diffusion_policy"
        train_files = project_files(train_package)
        deploy_files = project_files(deploy_package)
        self.assertEqual(train_files, deploy_files)
        for relative_path in train_files:
            self.assertEqual(
                (train_package / relative_path).read_bytes(),
                (deploy_package / relative_path).read_bytes(),
                f"shared RDP runtime drift: {relative_path}",
            )

    def test_native_pick_tube_deployment_chain_is_present(self) -> None:
        required = (
            DEPLOY / "deploy_pick_tube_rdp.py",
            DEPLOY / "configs" / "deploy_pick_tube_rdp.yaml",
            DEPLOY / "reactive_diffusion_policy" / "deploy" / "bridge_client.py",
            DEPLOY / "reactive_diffusion_policy" / "deploy" / "tactile_encoder_torch.py",
            DEPLOY / "reactive_diffusion_policy" / "common" / "artifact_manifest.py",
            DEPLOY / "reactive_diffusion_policy" / "common" / "pick_tube_action_contract.py",
            DEPLOY / "reactive_diffusion_policy" / "model" / "tactile_pca.py",
        )
        for path in required:
            self.assertTrue(path.is_file(), path)

    def test_wrong_source_custom_artifact_layer_is_absent(self) -> None:
        for relative_path in (
            "load_artifact.py",
            "check_artifact.py",
            "validate_pick_tube_contract.py",
            "configs/pick_tube_contract.yaml",
        ):
            self.assertFalse((DEPLOY / relative_path).exists(), relative_path)
        self.assertFalse((TRAIN / "export_artifact.py").exists())

    def test_deploy_defaults_are_operator_gated(self) -> None:
        config = (DEPLOY / "configs" / "deploy_pick_tube_rdp.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("artifact_verification: legacy-compatible", config)
        self.assertIn("auto_start: false", config)
        self.assertIn("max_iterations: 0", config)

    def test_deploy_requirements_extend_training_requirements(self) -> None:
        lines = [
            line.strip()
            for line in (DEPLOY / "requirements-rdp-deploy.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        self.assertEqual(lines[0], "-r requirements-rdp-training.txt")
        self.assertIn("msgpack==1.1.2", lines)
        self.assertIn("websockets==16.0", lines)

    def test_no_prebuilt_native_extensions_are_vendored(self) -> None:
        self.assertEqual(list(TRAIN.rglob("*.so")), [])
        self.assertEqual(list(DEPLOY.rglob("*.so")), [])

    def test_shell_entrypoints_parse(self) -> None:
        scripts = [*TRAIN.joinpath("scripts").glob("*.sh"), *DEPLOY.joinpath("scripts").glob("*.sh")]
        scripts.extend([TRAIN / "train_rdp.sh", TRAIN / "train_pick_tube_rdp.sh"])
        for script in scripts:
            result = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
