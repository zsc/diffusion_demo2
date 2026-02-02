# diffusion_demo2 重写 SPEC（Python + HTML Demo）

> 目标：将本仓库重写为一个“可一键跑起来、可在浏览器里操作”的 **Diffusion（DiT）训练/采样** Demo。  
> 受众：使用 `codex` / `gemini-cli` 进行自动化落地实现的工程脚手架与任务说明。  
> 语言：本文为中文规格说明（SPEC），代码/接口示例用英文 JSON/伪代码表达。

---

## 0. 现状盘点（基于当前仓库，作为可复用资产）

### 0.1 关键可复用点（尽量不动/少动）
- **数据源**：`data_loader.py`
  - 从 `IMAGE_URL = "http://zsc.github.io/widgets/celeba/48x48.png"` 下载 CelebA 48×48 sprite sheet
  - 解析为 `NUM_IMAGES=30000` 张小图（48×48×3）
  - 缓存：`cache/celeba_48x48_sprites.npy`
  - `get_dataloader(batch_size, data_limit)` 已提供标准训练输入（Tensor、归一化到 [-1, 1]）
- **模型**：`models/vae.py`（VAE/VQVAE/Discriminator）、`models/dit.py`（DiT）
- **训练函数**：`training/train_vae.py::train_vae_model(config)`、`training/train_dit.py::train_dit_model(config)`
- **采样函数**：`training/train_dit.py::p_sample_loop(...)`
- **输出目录约定**（可延续）：`checkpoints/`、`runs/`、`results/vae_samples/`、`results/dit_samples/`

### 0.2 当前痛点（重写要解决）
- Web 端交互偏“表单 + 控制台”，缺少 **作业列表 / 状态 / 日志 / 产物可视化**，需要手工复制模型路径。
- 训练/采样任务和 Web 进程耦合，缺少 **可恢复的元数据**（重启后不知道上次训练输出在哪里）。
- 配置 schema 分散在前端表单与 Python 代码里，缺少统一校验与默认值策略。

---

## 1. 重写后的 Demo 形态（MVP）

### 1.1 一句话定义
一个本地运行的 Web Demo：用户打开网页即可完成 **数据预览 → 训练 VAE/VQVAE（可选）→ 训练 DiT → 采样生成 → 查看日志/产物**。

### 1.2 非目标（明确不做）
- 不做生产级鉴权、部署、并发隔离、多用户、资源配额。
- 不追求训练质量/收敛到可用生成（MVP 以“跑得通、看得见”为准）。
- 不做复杂前端框架（MVP 仅原生 HTML + CSS + JS）。

---

## 2. 总体架构

### 2.1 组件划分
- **Web Server（Python）**
  - 提供页面（HTML/CSS/JS）与 JSON API
  - 启动/管理后台训练与采样作业（job）
- **Job Manager（Python）**
  - 统一创建 job、跟踪状态、落盘元数据、落盘日志、产物归档
  - 允许 Web UI 轮询查看进度
- **Core（复用现有）**
  - 数据：复用 `data_loader.py`
  - 模型/训练：复用 `models/*`、`training/*`（允许小幅重构为更易调用/更易记录产物）

### 2.2 线程 vs 进程（推荐策略）
MVP 推荐 **线程**（与当前 `app.py` 接近，改造成本最低）：
- 每个 job 由 `threading.Thread` 执行
- stdout/stderr 重定向到 job 的 log 文件

Stretch（可选）改为 **子进程**（更利于取消/隔离 GPU 内存）：
- Web 进程只负责写 config JSON 并 `subprocess.Popen(["python", "-m", "...", "--config", "..."])`
- job 通过 pid 管理

---

## 3. 目录结构（重写后建议）

目标：把“Web/Job/训练/模型/数据”解耦，避免所有逻辑堆在 `app.py`。

建议结构（可按需微调，但需要保留清晰边界）：

```text
.
├── CLAUDE.md
├── README.md                       # 简短运行说明（可后续补）
├── requirements.txt                # 新增：可一键安装依赖
├── src/
│   └── diffusion_demo/
│       ├── __init__.py
│       ├── server.py               # Web 入口：创建 app、注册路由、启动
│       ├── jobs.py                 # JobManager + Job 数据结构 + 落盘
│       ├── api.py                  # /api 路由（JSON）
│       ├── ui/
│       │   ├── templates/
│       │   │   └── index.html      # 单页 UI
│       │   └── static/
│       │       ├── app.js
│       │       └── style.css
│       ├── core/
│       │   ├── data_loader.py      # 复用/迁移自根目录 data_loader.py
│       │   ├── models/             # 复用/迁移自 models/
│       │   └── training/           # 复用/迁移自 training/
│       └── utils/
│           ├── io.py               # base64 图片、路径、json 读写
│           └── torch_env.py        # device、bf16 能力探测、seed
├── cache/                          # 复用：numpy cache
├── checkpoints/                    # 复用：模型权重
├── runs/                           # 复用：tensorboard logs
├── results/
│   ├── vae_samples/
│   └── dit_samples/
└── var/
    └── jobs/
        ├── <job_id>/
        │   ├── meta.json
        │   ├── log.txt
        │   └── artifacts.json
        └── index.json              # job 列表索引（可选）
```

说明：
- `var/jobs/<job_id>/meta.json` 用于重启后恢复 UI（至少能显示 job 历史与产物路径）。
- 训练产物仍写入 `checkpoints/`、`runs/`、`results/`，但 **job 元数据必须记录这些路径**，避免用户手动复制。

---

## 4. Job 设计（核心）

### 4.1 Job 状态机
`queued` → `running` → `succeeded` / `failed` / `canceled`

### 4.2 Job 元数据 schema（meta.json）

```json
{
  "id": "b3d7c6b7-...",
  "type": "vae|dit|sample|dataset",
  "status": "queued|running|succeeded|failed|canceled",
  "created_at": "2026-02-02T12:34:56Z",
  "started_at": null,
  "finished_at": null,
  "config": {},
  "artifacts": {
    "checkpoints": [],
    "images": [],
    "tensorboard_logdir": null
  },
  "error": {
    "message": null,
    "traceback": null
  }
}
```

### 4.3 日志策略
- 每个 job 独立 `log.txt`
- API 提供“增量拉取”能力（避免每次全量读文件）

---

## 5. API 设计（MVP）

统一前缀：`/api`

### 5.1 系统信息
`GET /api/system`
- 用于 UI 初始化展示环境能力与目录状态

响应示例：
```json
{
  "torch": {"version": "2.x", "cuda_available": true, "bf16_supported": true},
  "paths": {"cache": "cache", "checkpoints": "checkpoints", "runs": "runs", "results": "results"},
  "dataset": {"image_size": 48, "cached": true, "cached_file": "cache/celeba_48x48_sprites.npy"}
}
```

### 5.2 数据集

`POST /api/dataset/ensure`
- 触发下载与缓存（内部可直接调用 `CelebASpriteSheetDataset()`，若已缓存则快速返回）
- 返回 job_id（可选；也可同步执行并返回结果）

`GET /api/dataset/samples?count=16&seed=0`
- 返回若干张真实样本（base64 PNG），用于 UI “数据预览”

响应示例：
```json
{"images": ["<base64>", "<base64>"], "image_size": 48}
```

### 5.3 Job 列表与详情

`GET /api/jobs?type=vae|dit|sample&limit=50`
- 返回最近 job（按 created_at 倒序）

`GET /api/jobs/<job_id>`
- 返回单个 job 元数据（meta.json）

`GET /api/jobs/<job_id>/log?offset=0`
- 增量读日志

响应示例：
```json
{"text": "...\n", "next_offset": 12345, "eof": false}
```

### 5.4 启动训练：VAE / VQVAE

`POST /api/jobs/vae`

请求（标准 VAE 示例）：
```json
{
  "model": "vae",
  "lr": 0.0002,
  "batch_size": 128,
  "epochs": 5,
  "data_limit": 2048,
  "use_bfloat16": true,
  "latent_dim": 256,
  "kld_beta": 1.0,
  "use_gan": false
}
```

请求（VQVAE 示例）：
```json
{
  "model": "vqvae",
  "lr": 0.0002,
  "batch_size": 128,
  "epochs": 5,
  "data_limit": 2048,
  "use_bfloat16": true,
  "embedding_dim": 64,
  "num_embeddings": 512,
  "commitment_cost": 0.25,
  "vq_beta": 1.0,
  "vq_hidden_dims_enc": [128, 64],
  "vq_hidden_dims_dec": [64, 128],
  "use_gan": false
}
```

响应：
```json
{"job_id": "<uuid>"}
```

要求：
- 后端必须把 `training/train_vae.py::train_vae_model` 的返回路径写入 `job.meta.artifacts.checkpoints`
- 若 `use_gan=true`，要把 discriminator 权重路径也写入 artifacts（即便训练函数只返回 generator 路径，job 也应在已知命名规则下补齐记录）

### 5.5 启动训练：DiT

`POST /api/jobs/dit`

请求示例：
```json
{
  "vae_model_path": "checkpoints/....pth",
  "discriminator_model_path": null,
  "use_gan": false,
  "gan_loss_weight": 0.01,
  "lr_d": 0.0001,

  "ddpm_schedule": "linear",
  "ddpm_timesteps": 1000,

  "lr": 0.00001,
  "batch_size": 64,
  "epochs": 5,
  "data_limit": 2048,
  "use_bfloat16": true,

  "dit_patch_size": 2,
  "dit_hidden_size": 384,
  "dit_depth": 6,
  "dit_num_heads": 6,

  "vae_latent_is_flat": true,
  "vae_flat_latent_dim": 256,
  "vae_latent_channels": 16,
  "vae_latent_spatial_dim": 4,
  "vae_latent_clamp_range": [-2.0, 2.0]
}
```

要求：
- 后端必须校验 `vae_model_path` 存在
- 如果 `use_gan=true`，必须校验 `discriminator_model_path` 存在
- job artifacts 必须包含：DiT 权重路径、tensorboard logdir、每个 epoch 的采样图片路径（若产生）

### 5.6 启动采样（生成图片）

`POST /api/jobs/sample`

请求示例：
```json
{
  "vae_model_path": "checkpoints/....pth",
  "dit_model_path": "checkpoints/....pth",
  "num_samples": 16,
  "ddpm_timesteps": 1000,
  "use_bfloat16": true,

  "dit_patch_size": 2,
  "dit_hidden_size": 384,
  "dit_depth": 6,
  "dit_num_heads": 6,

  "vae_latent_channels": 16,
  "vae_latent_spatial_dim": 4,
  "vae_flat_latent_dim": 256,
  "vq_embedding_dim_for_dit": 64
}
```

响应：
```json
{"job_id": "<uuid>"}
```

产物要求：
- `job.meta.artifacts.images` 至少包含一个 `results/generated/<job_id>.png`（网格图）
- 同时提供 `GET /api/jobs/<job_id>/result` 直接返回 base64 图片（便于 UI 直接显示）

---

## 6. 前端页面（HTML）规格

### 6.1 页面布局（单页）
建议采用 Tab 或折叠区块：
1) **Dataset**
   - “Ensure dataset / Download & cache”按钮
   - 真实样本预览（默认 16 张，支持刷新、seed）
2) **Train VAE/VQVAE**
   - 表单：lr/batch/epochs/data_limit/bf16 + VAE/VQVAE 特有参数
   - “Start”按钮 → 创建 job
   - 展示：job 状态、日志（增量）、产物（最新重建图/采样图）、checkpoint 列表（可点击复制/自动填入下一步）
3) **Train DiT**
   - 表单：选择 VAE checkpoint（下拉列表来自 `/api/jobs` artifacts），以及 DiT 超参
   - “Start”按钮 → 创建 job
   - 展示：日志、epoch 采样图、最终 checkpoint
4) **Sample**
   - 表单：选择 VAE checkpoint + DiT checkpoint + num_samples/timesteps
   - “Generate”按钮 → 创建 job
   - 展示：生成结果图（base64）、可下载 PNG
5) **Jobs**
   - 列表：最近 jobs（type/status/时间）
   - 点击某 job → 右侧显示详情（meta、log、artifacts）

### 6.2 交互细节（必须）
- 创建 job 后，UI 立即进入“跟踪模式”：
  - 每 1s 轮询 `/api/jobs/<id>` 更新 status
  - 每 0.5~1s 增量拉取 `/api/jobs/<id>/log?offset=...`
- 当 status = `succeeded`：
  - 自动刷新 jobs 列表
  - 若 artifacts 里有 checkpoint：
    - 自动写入“最近一次训练输出”的选择器（便于下一阶段）
- 错误展示：当 status = `failed`，展示 `error.message` 并提供“展开 traceback”。

---

## 7. 配置与默认值（MVP 约定）

### 7.1 环境变量
- `PORT`：默认 5000
- `HOST`：默认 `127.0.0.1`
- `DEVICE`：默认 `auto`（cuda 优先，其次 cpu）
- `DATA_LIMIT_DEFAULT`：默认 2048（保证 CPU 也能跑）

### 7.2 目录与文件命名
- 模型输出沿用现有训练脚本的命名规则（run_name 拼接），但 job artifacts 要记录：
  - generator / dit / discriminator 的实际路径
  - `runs/...` 的 logdir
  - `results/...` 的 sample 图片路径

---

## 8. 依赖与运行方式（MVP）

### 8.1 requirements.txt（建议最小集）
- flask（或 fastapi/uvicorn 二选一；MVP 建议继续 Flask 以复用现有 `app.py` 经验）
- torch, torchvision
- pillow, numpy, requests
- tqdm
- tensorboard（用于训练可视化）
- einops（DiT 用到）

### 8.2 启动命令（示例）
```bash
python -m diffusion_demo.server
```

访问：
- `http://127.0.0.1:5000/`

---

## 9. 验收标准（MVP 必过）

1) **Dataset 预览可用**
   - 首次启动可触发下载并缓存 `cache/celeba_48x48_sprites.npy`
   - 页面可展示至少 16 张真实样本（base64 渲染）
2) **Job 系统可用**
   - 任意 job 创建后可在 Jobs 列表出现
   - status 能从 running → succeeded/failed 正确流转
   - 日志可在页面持续滚动更新（增量拉取）
3) **VAE 训练链路可跑通（低配）**
   - 使用 `data_limit=64~512`、`epochs=1` 在 CPU 也能完成一次训练并保存 checkpoint
   - 产物图片写入 `results/vae_samples/`（或 job 专属目录）并可在页面显示
4) **DiT 训练链路可跑通（低配）**
   - 使用 `ddpm_timesteps<=50`、`epochs=1` 能完成一次训练并保存 checkpoint
5) **采样链路可跑通**
   - 基于训练出的 VAE/DiT 或提供的 checkpoint，能生成网格图并在页面展示

---

## 10. 实施步骤（适合 codex / gemini-cli 分阶段落地）

建议按以下顺序实现，每步可独立验收：

### Phase A：工程化与可运行
- [ ] 增加 `requirements.txt`
- [ ] 创建 `src/diffusion_demo/server.py`，能启动并渲染最小 `index.html`
- [ ] 建立 `var/jobs/` 落盘目录与基础读写工具

### Phase B：JobManager（最关键）
- [ ] 实现 `JobManager.create_job(type, config) -> job_id`
- [ ] 实现 job 的 `meta.json` / `log.txt` 落盘
- [ ] 实现 API：`/api/jobs`、`/api/jobs/<id>`、`/api/jobs/<id>/log`

### Phase C：Dataset API + UI
- [ ] API：`/api/dataset/ensure`、`/api/dataset/samples`
- [ ] UI：Dataset Tab，能预览与刷新

### Phase D：接入 VAE 训练
- [ ] API：`POST /api/jobs/vae` 创建训练 job 并调用 `train_vae_model`
- [ ] job artifacts 正确记录 checkpoint、sample 图片、tensorboard logdir
- [ ] UI：VAE Tab 可启动、看日志、看产物、复制/选中 checkpoint

### Phase E：接入 DiT 训练
- [ ] API：`POST /api/jobs/dit` 调用 `train_dit_model`
- [ ] UI：DiT Tab 可启动、看日志、看 epoch sample、拿到 checkpoint

### Phase F：接入采样生成
- [ ] API：`POST /api/jobs/sample` + `GET /api/jobs/<id>/result`
- [ ] UI：Sample Tab 可生成并展示结果

---

## 11. Stretch（可选增强）
- Job 取消：`POST /api/jobs/<id>/cancel`（线程版可做“软取消标记”；进程版可 kill）
- SSE 实时日志（替代轮询）
- Checkpoint Registry：扫描 `checkpoints/` 自动建立索引与关联 config
- 生成过程可视化：保存 sampling 中间步并在前端做 slider 或 gif

