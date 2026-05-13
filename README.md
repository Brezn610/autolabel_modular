# autolabel_modular

LiDAR–多相机自动标注流水线：时间同步 → 标定 → **Grounding DINO 2D 检测** → **LiDAR 投影 / 视锥取点** → **DBSCAN + 3D OBB** → 置信度 / Top-K / **3D NMS** → JSON 与诊断报告。用于复现 OpenAnnotate3D 类「2D 开词汇检测 + 几何 lifting」思路的工程化版本。

---

## 目录架构（按数据流）

```
autolabel_modular/
├── __main__.py              # python -m autolabel_modular → 同 CLI 主流程
├── main.py                  # 兼容：python -m autolabel_modular.main
├── html_report.py           # 兼容：python -m autolabel_modular.html_report
├── dino_html_report.py      # 兼容：仅 DINO debug → HTML
├── settings/                # 配置与数据结构
│   ├── config.py            # AppConfig、DEFAULT_*、环境变量 DEMO_DATA_ROOT / DEMO_OUT
│   ├── dino_camera_config.py # Grounding DINO 按相机的 prompt / top_k
│   └── schemas.py           # FrameRecord、CalibrationBundle
├── ingest/                  # 原始数据读入
│   ├── lidar_io.py          # middle_lidar NPZ → (N,3)
│   └── timesync.py          # timesync_info.csv/xlsx → 帧表
├── calib/                   # 标定
│   └── calibration.json 解析、K/dist、T_lidar→cam (cTv @ vTl)
├── geom/                    # 几何
│   └── projection.py        # LiDAR→像素投影（float64）、视锥 mask
├── detect/                  # 2D
│   ├── detection.py         # Grounding DINO 加载与推理、标签映射
│   └── sam2_masks.py        # 可选 SAM2 mask 诊断（`--sam2-debug`）
├── lift3d/                  # 3D
│   └── lifting.py           # DBSCAN、OBB（默认仅 yaw + 竖直 z）
├── post/                    # 后处理
│   └── postprocess.py       # 置信度过滤、每相机 Top-K、3D NMS（IoU + 中心距辅助）
├── export/                  # 导出
│   └── exporter.py          # annotations JSON 组装与写盘
├── pipeline/                # 编排
│   └── runner.py            # run_pipeline：整链 + 可选 projection_vis
├── reports/                 # 报告
│   ├── console.py           # 终端分段诊断（与 raw_dino_debug 一致）
│   ├── html_report.py       # 单页 HTML（需 annotations_demo + debug）
│   └── dino_html_report.py  # 单页 HTML（仅需 raw_dino_debug + 数据根图）
└── cli/
    └── main.py              # argparse 入口
```

**依赖方向（自上而下）**：`cli` → `pipeline` → `detect` / `geom` / `lift3d` / `ingest` / `calib` / `post` / `export`；`reports` 读落盘 JSON，不进入训练环。

---

## 运行方式

在**包的上级目录**执行（例如 `/home/chase_610`），保证能解析到包名 `autolabel_modular`：

```bash
# 推荐
python -m autolabel_modular --max-frames 100 --save-projection-vis

# 与旧习惯等价
python -m autolabel_modular.main --max-frames 100 --save-projection-vis
```

**仅 Grounding DINO** 的 HTML 报告（**不需要** `annotations_demo.json`，只要 `raw_dino_debug.json` + 数据根下的 timesync 与相机图）：

```bash
python -m autolabel_modular.dino_html_report \
  --data-root /mnt/d/thesis/day \
  --output-root /home/chase_610/autolabel_modular/out \
  --max-frames 100
# 或
python -m autolabel_modular.reports.dino_html_report ...
```

默认写出 `dino_report.html` 与同名的 `dino_report_assets/`（每帧一张带 DINO 框的图 + 检测表）。`--display-camera` 指定优先展示的相机；若该帧该相机无检测，则自动换成本帧检测数最多的相机。

**完整流水线 HTML**（3D OBB 投影 + 2D 框，需先有 `out/annotations_demo.json` 与 `out/raw_dino_debug.json`）：

```bash
python -m autolabel_modular.html_report \
  --data-root /mnt/d/thesis/day \
  --output-root /home/chase_610/autolabel_modular/out
# 或
python -m autolabel_modular.reports.html_report ...
```

---

## 暴露的参数

### 环境变量

| 变量 | 含义 |
|------|------|
| `DEMO_DATA_ROOT` | 数据集根（`calibration.json`、`timesync_info.*`、各相机目录、`middle_lidar/`） |
| `DEMO_OUT` | 输出根目录；未设置时默认为 **包内** `autolabel_modular/out/` |

### 命令行（`cli/main.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data-root` | 见上 | 覆盖数据根 |
| `--output-root` | 见上 | 覆盖输出根 |
| `--conf-thr` | 0.18 | 写入 `annotations_demo.json` 前的分数阈值 |
| `--max-boxes-per-image` | 8 | 每帧×每相机保留 Top-K 个 3D 框 |
| `--nms-iou-thr` | 0.3 | 3D NMS：角点 AABB 近似体积 IoU |
| `--nms-center-dist-m` | 2.0 | NMS 辅助：中心距小于此且体积重叠达阈则抑制 |
| `--nms-size-overlap-min` | 0.3 | 同上，体积重叠比例下限 |
| `--save-projection-vis` | 关 | 写出 `projection_vis/*.jpg` |
| `--single-camera` | 空 | 仅单相机检测（如 `front_left_camera`） |
| `--max-frames` | 100 | 处理帧数上限 |
| `--no-obb-yaw-only` | 关 | 关闭「仅 yaw 竖直盒」，回到 Open3D 最小体积 OBB |
| `--no-dino-camera-specific` | 关 | 关闭按相机 DINO 配置，恢复全局 prompt + 固定阈值 |
| `--sam2-debug` | 关 | DINO 后对每框跑 **SAM2 Hiera-Small**（`facebook/sam2-hiera-small`），写入 `raw_dino_debug` 的 `sam_*` 字段与 `sam_debug/*.jpg`（**不改**视锥/DBSCAN/OBB） |

### `AppConfig`（未全部暴露 CLI，可在代码中改）

| 字段 | 默认 | 说明 |
|------|------|------|
| `cameras` / `cameras_for_detection` | 6 相机 | 投影列表 / 检测列表 |
| `grounding_dino_model_id` | IDEA-Research/grounding-dino-base | 2D 模型 |
| `dino_text_prompt` | 六类交通参与体 | 全局模式下的开词汇 prompt |
| `dino_box_threshold` / `dino_text_threshold` | 0.22 | **全局模式**（`dino_use_camera_specific=False`）下的 DINO 阈值 |
| `dino_use_camera_specific` | True | True：每相机独立 prompt + **低召回阈值 + Top-K**（见 `dino_camera_config.py`） |
| `dino_recall_box_threshold` / `dino_recall_text_threshold` | 0.01 | 相机专项模式下先召回候选框，再按每相机 `top_k` 截断 |
| `sam2_debug_enabled` | False（CLI `--sam2-debug`） | 仅诊断：SAM2 mask + JSON 字段 |
| `sam2_model_id` | `facebook/sam2-hiera-small` | Hugging Face 模型 id |
| `sam2_mask_threshold` | 0.5 | 将 SAM 输出 logits 二值化的阈值 |
| `sam2_debug_store_compressed_mask` | False | True 时在 JSON 中附加 zlib+base64 编码的整图 mask（体积大） |
| `max_points_projection` | 50000 | 投影可视化下采样 |
| `frustum_z_min` / `frustum_z_max` | 0.5 / 120 | 视锥深度（米，相机坐标） |
| `dbscan_eps` / `dbscan_min_samples` | 0.6 / 8 | 聚类 |
| `obb_yaw_only` | True（CLI 可关） | 竖直长方体 OBB |

### HTML 报告

- **`reports/html_report.py`**：`--demo-json`、`--debug-json`、`--output-html`、`--output-root`、`--display-camera`、`--difficulty-thr`、`--max-table-rows`、`--use-raw-annotations`（用 NMS 前的 `annotations_demo_raw.json`）。生成前会清空 `report_assets/frame_*.jpg`；画布对齐标定 **ImageSize**，并对 2D `xyxy` 做同比例缩放，使 3D 投影与黄框一致。
- **`reports/dino_html_report.py`**：`--debug-json`、`--output-html`、`--output-root`、`--data-root`、`--display-camera`、`--max-frames`、`--difficulty-thr`、`--max-table-rows`

---

## 输出产物（默认 `out/`）

| 文件 | 说明 |
|------|------|
| `annotations_demo_raw.json` | 后处理前、所有成功 OBB |
| `annotations_demo.json` | 置信度 + Top-K + NMS 后 |
| `raw_dino_debug.json` | 每条 2D 检测的 frustum 点数、OBB 是否成功 |
| `stage_stats.json` | 阶段计数 + 路径 |
| `projection_vis/` | 可选，LiDAR 彩色投影 |
| `report.html` + `report_assets/` | 由 `html_report` 生成（需 annotations_demo） |
| `dino_report.html` + `dino_report_assets/` | 由 `dino_html_report` 生成（仅需 raw_dino_debug） |
| `sam_debug/` | 可选，`--sam2-debug` 时每帧×相机 SAM mask 叠加图 |

---

## 最近一次跑批结果摘要（`out/stage_stats.json`，100 帧 × 多相机）

| 指标 | 数值 | 说明 |
|------|------|------|
| `raw_dets` | 3677 | 2D 检测总条数 |
| `frustum_nonempty` | 3297（**89.7%**） | 视锥内至少有一个 LiDAR 点 |
| `obb_success` | 2561（占 raw **69.7%**；占 frustum 非空 **77.7%**） | DBSCAN+OBB 成功 |

**Score（仅成功 OBB 的框）**：`median≈0.37`，`min` 贴近 DINO 阈值 0.22，说明不少框贴着阈值过线。

**按相机**：前向 / 右侧相机 frustum 与 OBB 通过率明显高于 **back_left / back_right**（后视更弱）。

**问题帧 Top-5**（frustum 点数 ≤3 占比高）：45、30、32、44、33。

---

## 原因分析（简要）

1. **3677 → 2561 的主因不是 NMS**：主要是 **视锥全空（约 10%）** + **视锥有点但聚类/OBB 失败（约 22% 相对非空视锥）**。NMS 与 Top-K 作用在已成功 OBB 之后，进一步减少 `annotations_demo.json` 中的条数。
2. **2D**：开词汇模型 + 固定阈值 → 弱分、框位偏差会直接推高「视锥空 / 点数极少」。
3. **几何**：整体 frustum 通过率较高，说明 **外参链路大体可用**；后视弱更像 **视角 + 距离 + 标定/同步** 叠加。
4. **3D**：DBSCAN 对点数与密度敏感；**仅 yaw OBB** 已改善「蓝框乱歪」的可视化，但不能单独解决「点太少」导致的失败。

---

## 后续改进步骤（建议顺序）

1. **2D**：调 `dino_*_threshold` / prompt；或换专用检测器；对 `raw_dino_debug` 里 `frustum_points≤3` 的样本做误差分析。  
2. **视锥**：按相机微调 `frustum_z_*`；试验略扩 2D 框（需在检测后处理层加）。  
3. **聚类**：对后视或远距离 **单独** `dbscan_eps` / `min_samples`，或点数极低时直接丢弃 / 弱监督 fallback。  
4. **多相机去重**：加强跨相机 3D 关联或在 NMS 前做全局合并。  
5. **与 GT 对比**（若需要）：先统一坐标系与单位，再谈定量指标。

---

## 依赖

Python 3.10+，`torch`、`transformers`（需含 **Grounding DINO** 与 **SAM2**，如 `Sam2Model` / `Sam2Processor`）、`open3d`、`numpy`、`opencv-python`、`pandas`（timesync xlsx/csv）、`scikit-learn`、`tqdm` 等。
