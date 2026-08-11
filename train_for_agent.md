在开始前，询问以下事项：
1. 询问用户使用的数据集，并根据数据集名称修改 srcipts/download_data.sh 和 ./configs/train_smolvla_jax.yaml 中的数据配置。若用户未给出，则直接使用当前配置
2. 询问用户的训练配置。若用户未给出，则直接使用当前配置
3. 查看用户的训练环境，若为 runpod 服务器，则设置 srcipts/download_data.sh 中的 HF_DATASET_CACHE_DIR="${HF_DATASET_CACHE_DIR:-/workspace}"；否则使用 HF_DATASET_CACHE_DIR="${HF_DATASET_CACHE_DIR:-${HOME}/.cache/huggingface/dataset}"
4. 询问用户的 hf token
5. 询问用户的 wandb token
请执行如下管线，进行训练：
1. 下载uv, tmux, rsync
2. 执行 uv sync
3. 登录 hf, wandb
4. 执行脚本 srcipts/download_data.sh。若存在依赖缺失，请在补全依赖后按照脚本中的方式，从失败处开始执行，不要重新执行脚本导致重复下载。
5. 新建 tmux 后台
6. 请在 tmux 后台运行 ./tools/train_smolvla_jax.py
开始训练后告诉用户，保证用户能通过 tmux 后台实时监控
