# SmolVLA 部署

客户端入口按模型类型拆分，脚本会固定选择对应 YAML，不接受 `--mode`：

| 模式 | 启动脚本 | 固定配置 |
| --- | --- | --- |
| 双手纯视觉 PyTorch SmolVLA | `scripts/start_smolvla.sh` | `configs/deploy_smolvla_pytorch.yaml` |
| 双手 SmolVLA + FRS | `scripts/start_smolvla_frs.sh` | `configs/deploy_frs.yaml` |
| 右手纯视觉 PyTorch SmolVLA | `scripts/start_smolvla_right.sh` | `configs/deploy_smolvla_pytorch_right.yaml` |

从仓库根目录启动：

```bash
bash deploy_smolvla/scripts/start_smolvla.sh
bash deploy_smolvla/scripts/start_smolvla_frs.sh
```

快速检查配置和环境：

```bash
bash deploy_smolvla/scripts/start_smolvla.sh --check
bash deploy_smolvla/scripts/start_smolvla_frs.sh --check
```

需要限制循环次数时，可直接追加 `--max-iterations N`。底层
`scripts/start_remote_client.sh` 仍接受显式 `--config`，供内部调试使用。
