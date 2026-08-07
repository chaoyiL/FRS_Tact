# modalities_eval 独立触觉评估实施方案

目标是在不修改模型结构和原始 v2.1 数据的前提下，让 `modalities_eval` 正确评估独立 tactile tensor 的 JAX VT-SmolVLA checkpoint，并依次完成 action-error、likelihood/Hutchinson 与辅助 t-SNE。

## 实施顺序

1. 先写测试覆盖 tactile tensor 保留/透传、tactile mask 消融、state mask 和 action padding；确认旧实现失败。
2. 最小修改共享 observation/context 和指标代码；运行聚焦测试及原有测试。
3. 将六路 v2.1 数据安全复制到工作区后，用 LeRobot 官方转换器生成独立 v3 数据；验证六路图像、episode/frame、索引和样本读取。
4. 固定相机 rename、normalization、episode/frame、noise seed：先跑 VT 单帧 `tactile` action-error。
5. 两个模型在同一帧清单上分别跑多帧 action-error。跨模型只把反归一化后的共同有效 horizon 作为系统级比较；VT 内部 full/no-tactile 才是主要触觉反事实。
6. action-error 验证通过后，先单帧再多帧运行 likelihood/Hutchinson。只解释每个模型内部的 original/ablated contribution；不同 chunk/normalizer 的 raw logL 不直接横比。
7. t-SNE 仅作辅助。若分别拟合，只解释各模型内部结构；跨模型坐标比较必须改为联合拟合。

## 固定路径

- 代码 worktree：`/home/yunjing/FRS/.worktrees/FRS_Tact-modalities-eval`
- 六路 v3 数据：`/home/yunjing/FRS/eval_data/KaiyueChen/pick_tube_02`
- VT checkpoint：`/home/yunjing/FRS/KaiyueChen/vtsmolvla_01_3w`
- 视觉 checkpoint：`/home/yunjing/FRS/FRS_Tact/checkpoints/pick_tube_02_3w_jax`

## 验收条件

- VT full 和 tactile-ablated 均能进入同一采样/velocity 路径，且共享 noise/probe。
- 纯视觉 checkpoint 的 tactile condition 明确为不适用，而非零贡献。
- episode 尾部 padding 不进入 action-error/likelihood 聚合。
- 两个 checkpoint 使用相同显式 RGB rename map、数据帧清单和独立输出目录。
- 所有命令、产物、运行环境和已知非公平因素写入 `CODEBASE_MEMORY.md`。
