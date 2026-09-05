# deploy_RDP

独立的 Pick Tube RDP 部署子项目，使用新源代码自带的 VB3 WebSocket bridge，详见
`DEPLOY_PICK_TUBE_RDP.md`。策略端接收两路视觉、四路触觉 RGB 和 20D state，在线
生成 PCA30 触觉特征，并以 `[1,20]` 单步相对动作发送给机器人服务器。

当前使用 `rdp_step_v3`，需要同时更新服务器。观测携带真实采集时间；策略按采集时间推进
动作索引，服务器以已成功入队的目标承接下一步增量。ACK明确区分调度与拒绝，不代表到达。
0902单右臂请使用 `scripts/start_pick_tube_rdp_right.sh`，通用launcher默认是双臂配置。

部署端保留完整 `reactive_diffusion_policy`，因为原生 checkpoint loader 会构造训练
workspace；但部署入口不会创建 dataset 或优化器。AT/LDP、PCA、normalizer 和数据
identity 会在策略构造前进行配对校验。

## 环境与权重

```bash
bash deploy_RDP/scripts/setup_rdp_env.sh
bash deploy_RDP/scripts/setup_pick_tube_data.sh weights
```

编辑 `configs/deploy_pick_tube_rdp.yaml`，确认 LDP、AT、触觉 encoder、PCA 和机器人
地址。默认配置保持 `runtime.auto_start: false`，warm-up 完成后需要人工按 Enter。

## 启动

机器人电脑先启动支持 RDP 单步相对动作合同的 VB3 server。策略电脑运行：

```bash
export VB_ROBOT_TOKEN='<与机器人服务器相同的 token>'
bash deploy_RDP/scripts/start_pick_tube_rdp_client.sh
```

上真机前至少执行：

```bash
cd deploy_RDP
.venv/bin/python -m pytest -q tests/test_pick_tube_rdp_deploy.py
```

默认 `artifact_verification: legacy-compatible` 是为当前已验证旧 checkpoint 对准备；
新训练产物应改用 `strict`。不要使用 smoke、identity 不匹配或 non-deployable 权重。
