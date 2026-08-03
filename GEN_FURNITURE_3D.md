# gen_furniture_3d.py 使用教程

## 背景

`gen_furniture_3d.py` 将全景图和分割掩码（或边界框）输入到 Aroomy 的 SAM3D 3D API 中，生成 3D 家具模型（.glb 格式）及每个物体的位姿元数据。

这条工具处于全自动流水线的后半段：

```
全景图 -> GroundingDINO -> SAM Mask -> SAM3D 3D -> 3D 家具模型 (.glb)
         (物体检测)       (掩码分割)   (3D 重建)    + 位姿 metadata.json
```

与之配合的工具：
- `get_pano_masks.py` — 管前半段（全景图 -> masks）
- `gen_furniture_3d.py` — 管后半段（masks -> 3D 家具）
- `visualization/compare_layout_furniture.py` — 3D 对比查看（房间 + 家具）

## 环境准备

```bash
conda activate lgt-net
cd D:\Documents\LGT-Net
```

## 两种运行模式

### 模式一：from_masks — 已有 mask URL，直接生成 3D

适用于已经通过 SAM mask API 拿到 mask 文件 URL 的场景。

```bash
python gen_furniture_3d.py from_masks \
    --image_url "https://your-server.com/pano.jpg" \
    --mask_urls "https://your-server.com/mask_0.png" "https://your-server.com/mask_1.png" \
    --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \
    --output_dir ./furniture
```

| 参数 | 必须 | 说明 |
|---|---|---|
| `--image_url` | 是 | 原始全景图的公网可访问 URL |
| `--mask_urls` | 是 | 一个或多个 mask 图片 URL，空格分隔 |
| `--sam3d_base_url` | 是 | SAM3D 3D API 地址 |
| `--output_dir` | 是 | 输出目录，下载的 .glb 和 metadata 存到这里 |
| `--prompt` | 否 | 文本提示词回退（默认 `furniture`） |
| `--no_textured_glb` | 否 | 禁用纹理导出（速度更快，细节更少） |
| `--poll_interval` | 否 | 轮询间隔秒数（默认 15） |
| `--poll_max_attempts` | 否 | 最大轮询次数（默认 60，共 15 分钟） |

### 模式二：full_pipeline — 全自动，从图片直接到 3D 家具

一条命令完成：GroundingDINO 检测 -> SAM 掩码 -> SAM3D 3D 生成。

```bash
python gen_furniture_3d.py full_pipeline \
    --image_path src/demo/demo.png \
    --dino_base_url "http://localhost:8001" \
    --dino_api_key "dev-only-insecure-key" \
    --text_prompt "sofa. chair. table. window." \
    --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \
    --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \
    --image_url_for_sam "https://raw.githubusercontent.com/.../demo.png" \
    --output_dir src/output/furniture_test
```

| 参数 | 必须 | 说明 |
|---|---|---|
| `--image_path` | 是 | 本地全景图路径 |
| `--dino_base_url` | 是 | GroundingDINO 服务地址 |
| `--dino_api_key` | 是 | GroundingDINO API 密钥（X-API-KEY） |
| `--text_prompt` | 是 | 检测提示词，用英文句号分隔，如 `"sofa. chair. table."` |
| `--sam_base_url` | 是 | SAM mask API 地址 |
| `--sam3d_base_url` | 是 | SAM3D 3D API 地址 |
| `--output_dir` | 是 | 输出目录（自动建 `masks/` 和 `furniture/` 子目录） |
| `--image_url_for_sam` | **强烈建议** | 传给 SAM API 的公网图片 URL（SAM 需要能访问到） |
| `--box_threshold` | 否 | 边界框置信度阈值（默认 0.3） |
| `--text_threshold` | 否 | 文本匹配阈值（默认 0.25） |
| `--furniture_prompt` | 否 | 3D 生成的文本提示词（默认 `furniture`） |
| `--poll_interval` | 否 | 轮询间隔秒数（默认 15） |
| `--poll_max_attempts` | 否 | 最大轮询次数（默认 60） |

## 依赖服务

| 服务 | 默认端口 | 说明 |
|---|---|---|
| GroundingDINO | 8001 | 仅 `full_pipeline` 需要 |
| SAM3D mask API | 外部 | `https://ai-test.aroomy.com/api/sam3d/masks` |
| SAM3D 3D API | 外部 | `https://ai-test.aroomy.com/api/sam3d` |

> **注意**：`--image_url_for_sam` 必须是 SAM API 服务器能访问的公网 URL，本地路径或 `localhost` 不行。

## 输出

运行后会在 `--output_dir` 下自动创建两个子目录：

```
output_dir/
├── masks/                          # mask 步骤产物（full_pipeline 模式）
│   ├── mask_0.png                  # 物体 0 的分割掩码
│   ├── mask_1.png                  # 物体 1
│   ├── annotated.png               # 标注预览图
│   └── dino_<job_id>_boxes.json   # GroundingDINO 检测结果
└── furniture/                      # 3D 步骤产物
    ├── mesh_0.glb                  # 物体 0 的独立 3D 模型（带纹理）
    ├── mesh_1.glb                  # 物体 1
    ├── combined_scene.glb           # 所有物体的合并场景
    └── metadata.json               # 每个物体的位姿元数据
```

### metadata.json 结构

每个物体包含未校正的 3D 位姿信息，后续可用于校正矩阵计算：

```json
[
  {
    "object_index": 0,
    "rotation": [[x, y, z, w]],       // 四元数 (xyzw)
    "scale": [[sx, sy, sz]],           // 缩放
    "translation": [[tx, ty, tz]]      // 平移
  },
  ...
]
```

## 代码调用

也可以作为 Python 模块导入：

```python
from gen_furniture_3d import run_from_masks, run_full_pipeline

# 从已有 mask URL 生成 3D
result = run_from_masks(
    image_url="https://example.com/pano.jpg",
    mask_urls=["https://example.com/mask_0.png"],
    sam3d_base_url="https://ai-test.aroomy.com/api/sam3d",
    output_dir="./furniture",
)

# 全自动流水线
result = run_full_pipeline(
    image_path="pano.jpg",
    dino_base_url="http://localhost:8001",
    dino_api_key="dev-key",
    text_prompt="sofa. chair. table.",
    sam_base_url="https://ai-test.aroomy.com/api/sam3d/masks",
    sam3d_base_url="https://ai-test.aroomy.com/api/sam3d",
    image_url_for_sam="https://example.com/pano.jpg",
    output_dir="./output",
)
```

## 完整实验流程示例

```bash
# 终端 1：启动 GroundingDINO
cd GroundingDINO
set HF_HUB_OFFLINE=1
python -m uvicorn app:app --host 0.0.0.0 --port 8001

# 终端 2：运行全自动流水线
cd D:\Documents\LGT-Net
python gen_furniture_3d.py full_pipeline \
    --image_path src/demo/demo.png \
    --dino_base_url "http://localhost:8001" \
    --dino_api_key "dev-only-insecure-key" \
    --text_prompt "sofa. chair. table. window." \
    --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \
    --sam3d_base_url "https://ai-test.aroomy.com/api/sam3d" \
    --image_url_for_sam "https://raw.githubusercontent.com/Aroomyaroomy/LGT-Net/dev-victor/src/demo/demo.png" \
    --output_dir src/output/furniture_test

# 终端 3（可选）：启动 LGT-Net 生成房间 mesh 做对比
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 查看 3D 结果（仅家具）
python visualization/compare_layout_furniture.py \
    --furniture_globs "src/output/furniture_test/furniture/*.glb"

# 或：房间 + 家具对比
python visualization/compare_layout_furniture.py \
    --room_mesh room_mesh.obj \
    --furniture_globs "src/output/furniture_test/furniture/*.glb"
```

## 常见问题

**Q: 轮询时网络超时怎么办？**
脚本内置了网络容错机制，遇到 `Timeout` 或 `ConnectionError` 会自动重试 3 次。如果 3 次都失败才会报错退出。此时服务器上的任务通常还在运行，可以用 curl 手动查询状态并下载结果。

**Q: 3D 生成需要多长时间？**
通常 2-5 分钟，取决于物体数量和复杂度。脚本默认最多等待 15 分钟（60 次 × 15 秒）。

**Q: mask URL 从哪里来？**
- 运行 `get_pano_masks.py` 后，SAM API 返回的结果中包含 mask 文件的下载 URL
- 或者直接把 mask 图片上传到 S3/CDN 获取公网 URL
