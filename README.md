# ACT / ALOHA 仿真复现

复现 [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/abs/2304.13705)在 MuJoCo 上的两个双臂任务。训练与评估均在单卡 RTX 4090 D 上完成，论文配置 2000 epochs。

| 任务 | 结果 | 论文 | 视频 |
|---|---|---|---|
| `sim_transfer_cube_scripted` 双臂传递方块 | **49/50 = 98.0%** | 86% | [普通做法](results/sim_transfer_cube_scripted/final_ep2000/rollout00_no_temporal_agg_success_return685.mp4) ｜ [TE](results/sim_transfer_cube_scripted/final_ep2000/rollout00_temporal_agg_success_return640.mp4) |
| `sim_insertion_scripted` 双臂插入 | **15/50 = 30.0%** | 32% | [成功](results/sim_insertion_scripted/eval_ep2000/rollout12_success_return460.mp4) ｜ [失败](results/sim_insertion_scripted/eval_ep2000/rollout00_fail_return430.mp4) |

论文数字为 3 个 seed 平均，本仓库只跑了 1 个 seed（seed 0）的 50 次评估。

---

## 双臂传递方块 · 98.0%

训练损失（灰 train ／ 蓝 val，最低 0.0461 @ 第 1926 轮）：

![loss](assets/loss_transfer_cube.png)

第 0 条 rollout 的关键帧：

![frames](assets/cam_transfer_cube.png)

## 双臂插入 · 30.0%

训练损失（最低 0.0394 @ 第 1898 轮）：

![loss](assets/loss_insertion.png)

第 12 条 rollout（成功）：

![frames](assets/cam_insertion.png)

第 0 条 rollout（失败 —— 插片已对准插槽，但没顶到底）：

![frames](assets/cam_insertion_fail.png)

插入任务要求插片推进到槽底才算成功，所以失败录像看上去很像成功。

---

视频为评估时录制的完整 rollout（每条 8 秒、400 步）。

---

## 代码

`code/act-main/` 是官方 ACT 仿真代码（MIT），改了 5 个文件：

| 文件 | 改动 |
|---|---|
| `imitate_episodes.py` | **断点续训**：新增 `--resume` / `--save_every` / `--epochs_per_run` / `--archive_ckpts` 等；检查点 `train_state.pt` 持久化优化器动量与逐 epoch 损失历史，弱机器可反复续训到同一全局目标 |
| `detr/main.py` | 同步新增上述参数（该文件在 import 期就 strict parse `sys.argv`） |
| `sim_env.py`、`ee_sim_env.py` | 相机渲染可选（`ACT_SIM_CAMERAS`），只渲染 `top` 时采集提速约 3 倍 |
| `constants.py` | 数据集根目录自动解析为同级的 `data/`，不必手填 |

运行要点：

```bash
# detr 内部有顶层绝对导入，必须以可编辑模式装成包
cd code/act-main/detr && pip install -e .

# 采集演示数据（cwd 必须在 act-main）
cd .. && python record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir ../data/sim_transfer_cube_scripted --num_episodes 50

# 论文配置训练 2000 epochs
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted --ckpt_dir ../ckpt/paper \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

# 评估 50 rollouts（加 --temporal_agg 则为时间集成）
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted --ckpt_dir ../ckpt/paper \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 \
  --eval --num_rollouts 50
```

`code/act-main/LICENSE` 为官方 MIT 许可（作者 Tony Z. Zhao），已原样保留。完整日志与逐条评估数据不在本仓库。
