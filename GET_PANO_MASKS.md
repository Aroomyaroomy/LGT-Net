# get_pano_masks.py 使用教程

## 背景

`get_pano_masks.py` 会自动把原始全景图和边界框（GroundingDINO 检测结果）输入到 Aroomy 的 SAM3D mask API 中，生成物体的分割掩码（masks）。

这条工具处于全景图到房间布局流水线的第二步：

```
全景图 → GroundingDINO → SAM Mask → Inpaint-Anything → LGT-Net
          (物体检测)      (掩码分割)   (物体擦除)         (房间布局)
```

## 环境准备

```bash
conda activate lgt-net
cd D:\Documents\LGT-Net
```

## 两种运行模式

### 模式一：本地模式（local）—— 已有 boxes.json 文件

适用于已经通过 GroundingDINO 获得 `boxes.json` 的场景，直接将文件喂入 SAM API。

```bash
python get_pano_masks.py local \
    --image_url "https://your-image-url/pano.jpg" \
    --boxes_json path/to/boxes.json \
    --sam_base_url "https://ai-test.aroomy.com/api/sam3d/masks" \
    --output_dir ./output/masks
```

| 参数 | 必须 | 说明 |
|---|---|---|
| `--image_url` | 是 | 全景图的可访问 URL（SAM API 需要远程下载） |
| `--boxes_json` | 是 | GroundingDINO 输出的 boxes.json 路径 |
| `--sam_base_url` | 是 | SAM3D mask API 地址 |
| `--output_dir` | 是 | 输出目录，下载的 mask 文件会存到这里 |
| `--min_score` | 否 | 最低置信度阈值，如 `0.5`，低于此的检测结果会被过滤 |
| `--poll_interval` | 否 | 轮询间隔秒数（默认 10） |
| `--poll_max_attempts` | 否 | 最大轮询次数（默认 30，共 5 分钟） |

### 模式二：远程模式（remote）—— 全自动，从图片直接到分割

将本地全景图上传到 GroundingDINO API 检测物体，自动拿到边界框后输入 SAM API。一条命令完成两步。

```bash
python get_pano_masks.py remote \
    --image_path src/demo/demo.png \
    --dino_base_url "http://localhost:8001" \
    --dino_api_key "dev-only-insecure-key" \
    --text_prompt "chair. sofa. table." \
    --sam_base_url "http://localhost:8080/api/sam3d/masks" \
    --image_url_for_sam "http://localhost:9000/demo.png" \
    --output_dir ./output/masks
```

| 参数 | 必须 | 说明 |
|---|---|---|
| `--image_path` | 是 | 本地全景图路径 |
| `--dino_base_url` | 是 | GroundingDINO 服务地址 |
| `--dino_api_key` | 是 | GroundingDINO API 密钥（X-API-KEY） |
| `--text_prompt` | 是 | 检测提示词，多个词语用英文句号分隔，如 `"sofa. chair. table."` |
| `--sam_base_url` | 是 | SAM3D mask API 地址 |
| `--output_dir` | 是 | 输出目录（同时保存 boxes.json + masks） |
| `--image_url_for_sam` | 否 | 传给 SAM 的图片 URL（SAM 需要能访问到。如果不传，会用 image_path） |
| `--box_threshold` | 否 | 边界框置信度阈值（默认 0.3） |
| `--text_threshold` | 否 | 文本匹配阈值（默认 0.25） |
| `--poll_interval` | 否 | 轮询间隔秒数（默认 10） |
| `--poll_max_attempts` | 否 | 最大轮询次数（默认 30） |

## 依赖服务

脚本运行前需要启动以下服务（如果用远程模式）：

| 服务 | 默认端口 | 启动命令 |
|---|---|---|
| GroundingDINO | 8001 | `cd GroundingDINO && python -m uvicorn app:app --host 0.0.0.0 --port 8001` |
| SAM3D mask API | 8080 | `cd aroomy-worker && uvicorn main:app --port 8080` |
| 图片静态服务 | 9000 | `cd src/demo && python -m http.server 9000` |

## 输出

- **mask 文件**：PNG 格式，保存到 `--output_dir` 下，如 `mask_0.png`、`mask_1.png`、`annotated.png`
- **boxes.json**（仅远程模式）：一份 `dino_<任务ID>_boxes.json` 保存到输出目录，方便追踪和复用

## 代码调用

也可以作为 Python 模块导入，在其他脚本中直接调用：

```python
from get_pano_masks import run_local_mode, run_remote_mode

# 本地模式
result = run_local_mode(
    image_url="https://example.com/pano.jpg",
    boxes_json_path="boxes.json",
    sam_base_url="https://ai-test.aroomy.com/api/sam3d/masks",
    output_dir="./masks",
)

# 远程模式
result = run_remote_mode(
    image_path="pano.jpg",
    dino_base_url="http://localhost:8001",
    dino_api_key="your-key",
    text_prompt="sofa. chair. table.",
    sam_base_url="http://localhost:8080/api/sam3d/masks",
    image_url_for_sam="http://localhost:9000/pano.jpg",
    output_dir="./masks",
)
```

## 数据格式约定

**boxes.json**（GroundingDINO 的输出，SAM API 的输入）：

```json
{
  "box_prompts": [
    {"xMin": 200, "yMin": 100, "xMax": 400, "yMax": 300},
    {"xMin": 500, "yMin": 200, "xMax": 800, "yMax": 450}
  ],
  "detections": [
    {
      "label": "沙发的",
      "score": 0.95,
      "box": {"xMin": 200, "yMin": 100, "xMax": 400, "yMax": 300}
    }
  ]
}
```

**SAM API 返回结果**（任务完成后）：

```json
{
  "data": {
    "masks": [
      {"url": "https://...", "file_name": "mask_0.png"},
      {"url": "https://...", "file_name": "mask_1.png"}
    ],
    "image": {"url": "https://...", "file_name": "annotated.png"}
  }
}
```

