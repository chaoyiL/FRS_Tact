在开始前：
1. 请询问用户使用的数据集，并根据数据集名称修改 download_data.sh 若用户未给出，则直接使用当前配置
2. 询问用户的 hf token
3. 询问用户的 wandb token
请执行如下管线，进行训练：
1. 下载uv, tmux, rsync
2. 执行 uv sync
3. 执行脚本 download_data.sh 若存在依赖缺失，请在补全依赖后按照脚本中的方式，从失败处开始执行，不要重新执行脚本、导致重复下载。
4. 新建 tmux 后台
5. 请在 tmux 后台运行 ./tools/train_smolvla_jax.py
开始训练后告诉用户，保证用户能通过 tmux 后台实时监控