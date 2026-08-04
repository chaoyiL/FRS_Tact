把触觉加入smolvla的训练，使用已经训练好的触觉encoder
主要链路：
observation.images.tactile_*
        ↓
tactile_encoder 的 tactile_resnet
        ↓
触觉 embedding / tactile tokens
        ↓
投影到 SmolVLA hidden size
        ↓
拼接 SmolVLA prefix
        ↓
tactile token 和 Image tokens + language tokens + state token 一起训练

第一阶段：冻结tactile encoder
创建新的config配置yaml：train_vtsmolvla_jax.yaml
图像增强（只针对RGB）

