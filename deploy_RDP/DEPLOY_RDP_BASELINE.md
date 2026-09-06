# 原始 RDP baseline（0906）部署

入口自动识别 checkpoint 的 `action_contract`：

- `single_right_chunk_relative10d_v1`：右臂状态、camera2、右臂触觉 15D，模型动作 10D。
- `dual_arm_chunk_relative20d_v1`：双臂状态、两路相机、触觉 30D，模型动作 20D。

PCA 文件仍保存两臂各 15 个分量；单右臂模型使用输出的 `[15:30]`。
不需要重新生成 15D PCA。状态使用各臂 TCP pose9 和夹爪宽度；每次慢推理以
当前 TCP 为固定基底，将历史观测和解码目标转换到正确坐标系。

加载器直接构造 `train_RDP/rdp_baseline` 策略并加载 LDP 的 EMA（或训练配置指定的
普通模型），不构造训练 workspace。部署需要保留仓库中的该训练包。
它核对 AT/LDP 合同、形状、时域配置，并逐张量核对指定 AT 与 LDP 内嵌 AT 的
网络权重。baseline 的 `latest.ckpt` 可直接加载，不需要
`--allow-unqualified-checkpoint`，也不使用旧 physical_v2 的 release qualification。

当前 baseline checkpoint 没有记录 PCA 文件内容哈希，只能核对 AT/LDP 记录的 PCA
路径和实际 PCA 的结构；请使用对应任务训练时生成的 PCA，结构相同不代表内容相同。

## 两端执行协议

异步 baseline 自动请求 `rdp_observation_deadline_v1`；同步对照仍使用
`rdp_observation_step_v1`。模型输出相对固定 chunk 基底的目标，
客户端将其转换为相对当前观测的动作。服务器保留这帧观测的原始 TCP 位姿快照，
使用该快照还原世界坐标目标，避免在前一个接受的目标上重复累加。
服务器仍使用最新实测位姿检查跟踪误差，ACK 的 `reference_source=observation`。
`scheduled` 只表示控制器已接受调度，不表示机器人已经到达。

旧模型仍使用 `rdp_step_v3`；新客户端不能与不支持新协议的旧服务器混用。
机器人服务器改动分为观测基底、执行时间和运动校验参考三份补丁。本机
`/home/typhon/vb3_robot_server` 已应用三份；其他同版本服务器按未应用部分依次执行：

```bash
cd /path/to/vb3_robot_server
git apply --check /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_step_v1.patch
git apply /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_step_v1.patch
git apply --check /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_deadline_v1.patch
git apply /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_deadline_v1.patch
git apply --check /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_motion_limits.patch
git apply /path/to/FRS_Tact/deploy_RDP/server_patches/rdp_observation_motion_limits.patch
```

已经应用的服务器不要重复应用。其他版本需要先处理补丁上下文差异。

## 启动

先退出旧服务器进程，再重新运行原来的机器人端脚本以加载新代码：

```bash
bash /home/typhon/vb3_robot_server/scripts/bimanual_rdp.sh
```

确认右臂 YAML 的 LDP、AT、PCA 都指向同一任务后，在 FRS_Tact 目录运行：

```bash
bash deploy_RDP/scripts/start_pick_tube_rdp_right.sh
```

日志应出现 `Loaded baseline ... tactile input=15D` 和
`requires server rdp_observation_deadline_v1`，执行 ACK 应为 `reference=observation`。
当前配置仍在 warmup 后等待按 Enter 启动。

## 异步视觉规划与逐步触觉执行

baseline 默认采用 `planning_mode: asynchronous`。一个后台线程运行 LDP，主执行
线程每帧编码新触觉、运行 AT、发送一个动作并处理 ACK。CUDA 上 LDP 使用独立流，
主循环不再通过设备级同步等待全部 LDP 工作。GPU 计算资源仍共享，实际延迟需要实测。

```yaml
control:
  control_frequency: 30.0
  controller_frequency: 80.0
  planning_mode: asynchronous
  ldp_inference_frequency: 6.0
  slow_update_interval: 16
```

`ldp_inference_frequency` 表示后台规划请求的最高频率；慢推理时只保留最新请求，
不积压历史任务。`slow_update_interval` 表示采纳新计划的控制步间隔，当前约
0.53 秒。两者独立，不要求间隔整除。新计划必须连同采集时间、固定基底一起交接，
AT 按服务器指定的执行时间相对计划采集时间计算动作索引，交接时不把索引错误地清零。
如果旧计划即将耗尽，可提前采纳已经完成的有效新计划，不必等满该间隔。

首个计划需要等待推理完成；之后后台规划尚未完成时，AT 继续使用有效旧计划和新触觉。
旧计划达到模型有效动作长度而没有可用新计划时，报错停止，不继续重复末尾动作。
reset 会使前一轮规划结果失效，退出时关闭规划线程。

这里的协议字段 `action_horizon=1 / steps_per_inference=1` 仍表示一次通信调度一个
动作，模型有效预测长度仍为 29。原始 RDP 的同名 `steps_per_inference` 是 LDP 与
控制频率之比，不能直接对应本协议字段。新实现对齐其视觉规划与触觉执行分离的机制，
沿用本机 30Hz 时间基准，补偿至明确执行时刻，并非复刻原始 24Hz 和固定延迟步配置。

若需与此前同步部署做对照，将 `planning_mode` 改为 `synchronous`；此时
`slow_update_interval` 恢复为串行 LDP 推理间隔，`ldp_inference_frequency` 不参与调度。
旧 0902 默认继续使用同步运行时。

日志使用 `plan_updated` 表示本步是否采纳了计划，`execution_inference_ms` 表示前台
执行推理耗时，`planner` 表示后台状态，`adopted_ldp_ms` 是最近采纳计划的 LDP 耗时。
首次计划的等待时间会包含在首次前台耗时中。

## 执行时间修复与切换诊断

服务器发布真实执行观测时附加 `observation.action_target_timestamp`，设为发布前
服务器时刻加 80ms。这个时间和 obs_seq、位姿快照绑定，避免依赖客户端与服务器
墙钟同步。客户端按这个时间对应的最近模型控制步解码，服务器使用同一个时间
调度；ACK 必须确认同一个值，不允许收包后悄悄改为 `now + 50ms`。

观测采集到执行的总延迟还包含采集与传输时间，因此这不是固定只前瞻80ms。
例如已记录的215ms总延迟，会使执行索引比观测索引前进约6–7步。
若推理错过目标时间，该动作不入队，ACK 为 `rdp_execution_deadline_missed`，
两端继续获取下一帧观测。其他执行拒绝仍按原方式结束运行。

未来触觉未知，当前显式采用最后一个实际样本保持。历史触觉前缀与此前重采样
保持一致，不读取当前采集时间之后的记录。这是前瞻所需假设，并不是对未来接触的预测。

每次切换会额外解码旧计划，和新计划在同一执行时刻比较。若旧计划已越过有效
时域，旧目标记为 null，不复用末尾动作。默认保存到 `deploy_RDP/logs/rdp_*.jsonl`，
可通过 `runtime.trace_dir` 修改位置；启动日志打印实际文件路径。
记录包含 obs_seq、观测状态、触觉15D/30D、wire动作、ACK，以及采纳时旧/新 latent、
固定基底、计划采集时间与 episode-frame 目标。控制台另显示 capture_tick 和 lookahead_ticks。

这次修正确定的时序遗漏，未加入目标平滑、禁止反向或强行改基底的规则。
之前的真实权重对照显示，前瞻能改善部分回拉，但不能保证所有轴的切换差异消失；
需要用执行时刻一致后的目标日志继续评估模型计划一致性，不能直接宣称真机已修复。

## 观测相对动作的运动校验

观测/执行时间协议中的 wire 位移包含观测采集后机器人已经移动的距离，因此不能
把它直接当作从当前位置开始的运动量。服务器保留 wire 的形状、有限值、夹爪范围
及6D旋转有效性校验，先用对应观测还原绝对目标，再按新鲜实测检查配置的
3cm平移和0.5rad旋转上限。上限没有放宽；旧 `rdp_step_v3` 仍按原合同校验。

例如旧观测位置为0、实测已到3cm、目标为4cm时，待跟踪误差是1cm；若实测仍为0，
同一目标则因4cm误差被拒绝。两种情况均有回归测试。

151414运行暴露了这个遗漏；其最后被拒绝的obs15未被旧客户端写入JSONL，不能
精确声称该步在修正后一定通过。现在客户端会保存拒绝ACK及对应动作/计划数据，
报错也不再附加误导性的协议不匹配提示。
本次客户端178项、RDP服务器92项通过；共享校验76项通过，跳过1项修改前就存在的
非RDP调度时间断言失败（100.01与100.05）。

## 本次离线验证（2026-09-06）

使用本机 `rdp_0906/press` 的真实 LDP、AT、PCA 和 encoder0824，在 CPU 上对
`eval_obs_20260905_201734` 的第 0、1、2、20 帧完成编码、慢推理、快速解码和
动作基底转换，输出均为有限 10D 数值。这是接口验证，不是 press 任务成功率评估。
未连接或驱动机器人。

客户端相关回归测试 117 项通过；服务器相关测试 65 项通过。
扩大到部署测试目录时，旧 action_contract_v2 测试因已删除的训练函数 `append`
无法收集；其余测试 165 项通过、1 项失败。后者硬编码 0902 路径，
与用户当前选择的 0906 press 配置不符；本次保留用户配置。

追加异步验证：配置、线程、生命周期、基底和已有部署合同共 149 项测试通过。
真实 0906 press 权重在 CPU 上用上述记录的前三帧图片循环输入，以 30Hz 构造
45 步采样时间，完成第 16、32 步计划交接；LDP 工作期间 AT 持续返回有效动作。
该测试只验证异步链路，不代表实测 30Hz 吞吐量。当前执行环境无法访问 NVIDIA
驱动，尚未验证 CUDA 并发延迟或真机效果。

执行时间修复后：客户端相关176项、服务器79项测试通过；真实0906 press权重完成
45步CPU执行时间测试。144459的真实输入重放确认前瞻实现符合预期，但仍有部分轴
切换差异；详见[回拉特征与修复验证](../docs/analysis/rdp_20260906_144459/README.md)。
