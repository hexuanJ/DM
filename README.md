# Stable Diffusion — 潜空间扩散模型实验

*Stable Diffusion 由 [Stability AI](https://stability.ai/) 和 [Runway](https://runwayml.com/) 协作完成，基于以下研究工作：*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp),
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

---

## 一、项目简介

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)

Stable Diffusion 是一种**潜空间文本到图像的扩散模型**（Latent Text-to-Image Diffusion Model）。得益于 [Stability AI](https://stability.ai/) 提供的算力支持和 [LAION](https://laion.ai/) 的数据贡献，该模型在 [LAION-5B](https://laion.ai/blog/laion-5b/) 数据库的子集上完成了 512×512 分辨率的训练。

模型采用冻结的 **CLIP ViT-L/14** 文本编码器对文本提示进行条件化控制，���结构如图所示：

![模型架构](assets/modelfigure.png)

**核心组件**：

| 组件 | 参数量 | 作用 |
|------|--------|------|
| **UNet**（去噪网络） | 860M | 在潜空间中预测并去除噪声，是生成图像的核心。内含 ResBlock + Self-Attention + Cross-Attention |
| **CLIP 文本编码器** | 123M | 将文字描述编码为 768 维条件向量，通过交叉注意力注入 UNet |
| **VAE**（自编码器） | 83M | 图像 ↔ 潜空间的压缩/解压，下采样因子为 8（512×512 → 64×64×4） |

模型相对轻量，在 **10GB 显存以上**的 GPU 上即可运行。详见 [模型卡片](https://huggingface.co/CompVis/stable-diffusion)。

---

## 二、Stable Diffusion v1 权重

Stable Diffusion v1 采用**下采样因子 8 的自编码器 + 860M UNet + CLIP ViT-L/14 文本编码器**的架构。模型先以 256×256 分辨率预训练，再以 512×512 分辨率微调。

官方提供了以下四个版本的预训练权重，各版本之间为**逐步迭代**关系：

| 版本 | 训练数据 | 训练细节 |
|------|---------|---------|
| `sd-v1-1.ckpt` | [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) + [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) | 256×256 训练 237k 步 → 512×512 训练 194k 步 |
| `sd-v1-2.ckpt` | [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/)（美学评分>5.0，分辨率≥512，水印概率<0.5） | 从 v1-1 恢复，512×512 训练 515k 步 |
| `sd-v1-3.ckpt` | laion-aesthetics v2 5+ | 从 v1-2 恢复，训练 195k 步，**10% 文本条件丢弃**（提升 CFG 效果） |
| **`sd-v1-4.ckpt`**（本项目使用） | laion-aesthetics v2 5+ | 从 v1-2 恢复，训练 225k 步，**10% 文本条件丢弃** |

以下为不同 Classifier-Free Guidance Scale（1.5~8.0）、50 步 PLMS 采样下各版本的 FID/IS 评估对比：

![各版本权重评估对比](assets/v1-variants-scores.jpg)

> **本项目使用 `sd-v1-4.ckpt`**，它是目前质量最优的 v1 开源权重。

### 官方示例输出

以下图片均为 Stable Diffusion 原始模型生成的示例，展示了模型对不同 prompt 的理解能力：

**文字→图像示例**（prompt: "a photograph of an astronaut riding a horse" 等）：

![txt2img 示例1](assets/stable-samples/txt2img/merged-0005.png)
![txt2img 示例2](assets/stable-samples/txt2img/merged-0007.png)

**同一主题不同表达的生成对比**（prompt 均围绕 "fire" 主题）：

| prompt | 生成结果 |
|--------|---------|
| "fire" | ![fire](assets/fire.png) |
| "a photograph of a fire" | ![photograph](assets/a-photograph-of-a-fire.png) |
| "a painting of a fire" | ![painting](assets/a-painting-of-a-fire.png) |
| "a watercolor painting of a fire" | ![watercolor](assets/a-watercolor-painting-of-a-fire.png) |
| "the earth is on fire, oil on canvas" | ![earth-fire](assets/the-earth-is-on-fire,-oil-on-canvas.png) |
| "a shirt with a fire printed on it" | ![shirt-fire](assets/a-shirt-with-a-fire-printed-on-it.png) |

可以看到，模型能够区分**摄影（photograph）、油画（painting）、水彩（watercolor）、印花T恤（shirt with print）**等不同视觉风格。

**图像→图像示例**（草图转艺术画）：

输入草图：

![草图输入](assets/stable-samples/img2img/sketch-mountains-input.jpg)

输出效果：

![img2img 输出1](assets/stable-samples/img2img/mountains-3.png)
![img2img 输出2](assets/stable-samples/img2img/mountains-2.png)

**图像修复示例**：

![inpainting 示例](assets/inpainting.png)

---

## 三、Google Colab 环境配置

本项目已在 **Google Colab**（Python 3.12、PyTorch 2.x、CUDA T4 15GB）上完成验证。由于原项目依赖��旧版本的库，需要进行兼容性修复。

### Cell 1：克隆仓库

```python
!git clone https://github.com/hexuanJ/DM.git
%cd /content/DM
```

### Cell 2：安装依赖 + 一次性修复兼容性问题

> 本 Cell 具有**幂等性**（无论运行多少次结果相同），解决以下兼容性问题：
>
> | 问题 | 原因 | 修复方式 |
> |------|------|---------|
> | `torch._six` 不存在 | PyTorch 2.x 移除了该内部模块 | 替换为 `string_classes = (str,)` |
> | `AutoFeatureExtractor` 已弃用 | transformers 新版改名 | 仅在 `scripts/` 中替换为 `CLIPImageProcessor` |
> | `torch.load` 安全警告 | PyTorch 2.x 新增安全机制 | 添加 `weights_only=False` |
> | `Trainer.add_argparse_args` 不存在 | PyTorch Lightning 2.x 移除了该方法 | 锁定安装 `pytorch-lightning==1.9.5` |

```python
# ===== 安装依赖 =====
!pip install -q transformers diffusers invisible-watermark \
    omegaconf einops pytorch-lightning==1.9.5 \
    kornia torchmetrics open_clip_torch albumentations accelerate
!pip install -q -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
!pip install -q -e git+https://github.com/openai/CLIP.git@main#egg=clip
!pip install -q -e .

# ===== 一次性修复所有兼容性问题 =====
import glob

FIXES = [
    ('from torch._six import string_classes', 'string_classes = (str,)'),
    ('torch.load(ckpt, map_location="cpu")',  'torch.load(ckpt, map_location="cpu", weights_only=False)'),
    ('torch.load(path, map_location="cpu")',  'torch.load(path, map_location="cpu", weights_only=False)'),
    ('torch.load("models/ldm/inpainting_big/last.ckpt")',
     'torch.load("models/ldm/inpainting_big/last.ckpt", map_location="cpu", weights_only=False)'),
]
SCRIPT_FIXES = [
    ('from transformers import AutoFeatureExtractor', 'from transformers import CLIPImageProcessor'),
    ('AutoFeatureExtractor.from_pretrained',          'CLIPImageProcessor.from_pretrained'),
]
ENCODER_RESTORE = [
    ('from transformers import CLIPImageProcessor',   'from transformers import CLIPTokenizer, CLIPTextModel'),
]

for pyfile in glob.glob('**/*.py', recursive=True):
    with open(pyfile, 'r') as f:
        content = f.read()
    original = content
    for old, new in FIXES:
        content = content.replace(old, new)
    if pyfile.startswith('scripts/'):
        for old, new in SCRIPT_FIXES:
            content = content.replace(old, new)
    if pyfile == 'ldm/modules/encoders/modules.py':
        for old, new in ENCODER_RESTORE:
            content = content.replace(old, new)
    content = content.replace('weights_only=False, weights_only=False', 'weights_only=False')
    if content != original:
        with open(pyfile, 'w') as f:
            f.write(content)

print("✅ 完成")
```

### Cell 3：下载 Stable Diffusion v1-4 权重

```python
!mkdir -p models/ldm/stable-diffusion-v1/
!wget -q --show-progress -O models/ldm/stable-diffusion-v1/model.ckpt \
    "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
print("✅ SD v1-4 权重下载完成")
```

> 权重文件约 4GB。首次运行 txt2img 时还会自动下载 CLIP 文本编码器（~1.7GB）和 Safety Checker（~1.2GB），它们会被缓存到 `/root/.cache/huggingface/`，后续运行无需重新下载。

---

## 四、任务实现

本项目完成了以下四个任务，均在 Google Colab T4 GPU（15GB VRAM）上验证通过。

---

### 任务一：文字生成图像（Text-to-Image）

**任务内容**：输入一段自然语言文字描述（prompt），模型从纯高斯噪声出发，经过反复去噪，生成与描述语义匹配的 512×512 图像。

**工作流程**：

```
                    "a photograph of an astronaut riding a horse"
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  CLIP ViT-L/14       │
                           │  冻结文本编码器       │
                           └─────────┬───────────┘
                                     │ 768 维条件向量
                                     ▼
  纯噪声 z_T ──→ UNet 去噪（50步）←──交叉注意力──── 文本条件
                    │
                    ▼
              潜空间 z_0 ──→ VAE 解码器 ──→ 512×512 RGB 图像
```

**运行代码（Cell 4）**：

```python
%cd /content/DM
!python scripts/txt2img.py \
    --prompt "a photograph of an astronaut riding a horse" \
    --plms --ddim_steps 50 --n_samples 1 --scale 7.5 --seed 42 \
    --outdir outputs/txt2img-samples

from IPython.display import Image as IPImage, display
import glob
for img in sorted(glob.glob("outputs/txt2img-samples/samples/*.png")):
    display(IPImage(img))
```

**参数详解**：

| 参数 | 默认值 | 含义与建议 |
|------|--------|-----------|
| `--prompt` | 必填 | 文字描述。越具体效果越好，建议包含主体、风格、质量修饰词（如 "photorealistic, 8k, sharp focus"） |
| `--plms` | 关闭 | 启用 [PLMS 采样器](https://arxiv.org/abs/2202.09778)（推荐），比默认 DDIM 更快且质量更好 |
| `--dpm_solver` | 关闭 | 启用 DPM-Solver 采样器，可用更少步数（如 25 步）达到同等质量 |
| `--ddim_steps` | 50 | 去噪步数。越多质量越好但越慢。PLMS 推荐 50 步，DPM-Solver 推荐 25 步 |
| `--n_samples` | 3 | 每个 prompt 生成几张图。Colab T4 显存有限，建议设为 1-2 |
| `--n_iter` | 2 | 采样迭代次数。总生成图数 = n_samples × n_iter |
| `--scale` | 7.5 | Classifier-Free Guidance 引导系数。越大越严格遵循 prompt，但太大（>15）会导致图像过饱和。**推荐范围 5-12** |
| `--seed` | 42 | 随机种子。相同 seed + 相同参数 = 完全相同的输出，方便复现 |
| `--H` / `--W` | 512 | 输出图像高/宽（像素），必须是 64 的倍数。模型在 512×512 上训练，其他尺寸可能效果下降 |
| `--C` | 4 | 潜空间通道数（固定为 4，不建议修改） |
| `--f` | 8 | 下采样因子（固定为 8，与 VAE 架构对应） |
| `--from-file` | 无 | 从文件批量读取 prompt，每行一个。适合大量生成 |
| `--precision` | autocast | 推理精度。`autocast` 使用 FP16 混合精度加速，`full` 使用 FP32 |
| `--config` | `configs/stable-diffusion/v1-inference.yaml` | 模型配置文件路径 |
| `--ckpt` | `models/ldm/stable-diffusion-v1/model.ckpt` | 模型权重路径 |

**任务结果**：模型成功生成了宇航员骑马的照片风格图像，耗时约 15-20 秒（T4 GPU）。

---

### 任务二：图像到图像（Image-to-Image）

**任务内容**：输入一张参考图�� + 文字描述，模型以参考图为起点进行"重绘"，可实现风格迁移、内容修改、细节增强等效果。

**与 txt2img 的核心区别**：

```
txt2img：从 100% 纯噪声开始 ──→ 完全由文字控制
img2img：从 输入图 + 部分噪声 开始 ──→ 保留原图结构，文字引导修改

具体流程：
  输入图片 ──→ VAE 编码 ──→ 潜空间表示 z_0
                                │
                  加入 strength 比例的噪声 ──→ z_t（t = strength × T）
                                │
  prompt ──→ CLIP ──→ 交叉注意力 ──→ UNet 从 z_t 开始去噪
                                │
                        VAE 解码 ──→ 输出图像
```

**运行代码（Cell 5）**：

```python
%cd /content/DM
!python scripts/img2img.py \
    --prompt "a girl with black hair, beautiful face, photorealistic, 8k" \
    --init-img data/inpainting_examples/bench2.png \
    --strength 0.4 --ddim_steps 75 --n_samples 1 --scale 10.0 \
    --outdir outputs/img2img-samples

from IPython.display import Image as IPImage, display
import glob
for img in sorted(glob.glob("outputs/img2img-samples/samples/*.png")):
    display(IPImage(img))
```

**关键参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--init-img` | 必填 | 输入参考图片路径 |
| `--strength` | 0.75 | 噪声强度（0.0~1.0）。**这是 img2img 最重要的参数** |

**strength 参数效果对照**：

| strength | 效果描述 | 适用场景 |
|----------|---------|---------|
| 0.1~0.3 | 几乎不变，仅微调色调/亮度 | 色彩校正 |
| 0.3~0.5 | 保留主体结构，细节可被改变 | **人像美化、风格微调（推荐）** |
| 0.5~0.7 | 构图大致保留，内容明显变化 | 风格迁移 |
| 0.7~1.0 | 几乎完全重绘，仅保留大致色调 | 创意生成 |

**任务结果**：以 `bench2.png` 为输入，strength=0.4，成功生成了保留原图构图但符合 prompt 描述的人像图片。

---

### 任务三：Diffusers 集成（最简方式）

**任务内容**：使用 Hugging Face [diffusers](https://github.com/huggingface/diffusers) 库，以最少的代码完成 Stable Diffusion 推理。

**与原始脚本的对比**：

| 对比项 | 原始脚本 `txt2img.py` | Diffusers 集成 |
|--------|----------------------|----------------|
| 代码量 | ~350 行 | ~10 行 |
| 权重管理 | 手动下载 4GB `.ckpt` | 自动从 HuggingFace Hub 下载并缓存 |
| Safety Checker | 手动初始化 | 内置，自动运行 |
| 负面提示词 | 不支持 | ✅ `negative_prompt` 参数 |
| FP16 推理 | `--precision autocast` | `torch_dtype=torch.float16` |
| 返回值 | 保存到文件系统 | 直接返回 PIL Image 对象 |

> **注意**：原 README 中的 `pipe(prompt)["sample"][0]` 在新版 diffusers 中已改为 `pipe(prompt).images[0]`。

**运行代码（Cell 6）**：

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("diffusers_result.png")

from IPython.display import Image as IPImage, display
display(IPImage("diffusers_result.png"))
```

**任务结果**：成功生成火星上宇航员骑马的图像，代码极其简洁，首次运行需从 HuggingFace 下载模型（约 5GB，自动缓存）。

---

### 任务四：图像修复（Inpainting）

**任务内容**：给定一张图片和一个 mask（白色标记需要修复的区域），模型自动填充 mask 覆盖的区域，使其与周围内容自然融合。

![inpainting 效果示例](assets/inpainting.png)

**工作流程**：

```
  原始图片 ──────────┐
                    ├──→ masked_image = (1 - mask) × image
  mask（黑白图）────┘
                              │
                    VAE 编码 + mask 下采样 → 拼接为条件输入
                              │
                    UNet 在 mask 区域生成新内容
                              │
                    VAE 解码
                              │
                    最终输出 = 原图未mask部分 + 生成的mask部分
```

**输入数据格式**：`--indir` 目录下需要**成对文件**：

```
bench2.png           ← 原始图片（RGB）
bench2_mask.png      ← 对应 mask（白色=要修复，黑色=保留）
```

项目在 `data/inpainting_examples/` 下自带了 8 对示例图片。

> **注意**：图像修复使用**独立的预训练权重**（387M 参数，VQ-f4 架构，约 3GB），与 txt2img/img2img 使用的 SD v1-4 权重不同。运行前需释放其他模型占用的 GPU 显存。

**运行代码（Cell 7）**：

```python
import torch, gc, shutil
try: del pipe
except: pass
gc.collect(); torch.cuda.empty_cache()

%cd /content/DM

# 下载 inpainting 专用权重（约 3GB）
!mkdir -p models/ldm/inpainting_big
!wget -q --show-progress -O models/ldm/inpainting_big/model.zip \
    https://ommer-lab.com/files/latent-diffusion/inpainting_big.zip
!cd models/ldm/inpainting_big && unzip -o model.zip
shutil.copy("models/ldm/inpainting_big/model.zip", "models/ldm/inpainting_big/last.ckpt")

# 运行修复
!python scripts/inpaint.py \
    --indir data/inpainting_examples --outdir outputs/inpainting --steps 50

from IPython.display import Image as IPImage, display
import glob
for img in sorted(glob.glob("outputs/inpainting/*.png")):
    display(IPImage(img, width=512))
```

**参数说明**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--indir` | 必填 | 包含图片和对应 mask 的输入目录 |
| `--outdir` | 必填 | 修复结果的输出目录 |
| `--steps` | 50 | DDIM 采样步数 |

**任务结果**：模型成功处理了 8 对示例输入，对 mask 区域进行了语义合理的填充，边缘过渡自然。

---

## 五、项目结构

```
DM/
├── README.md                           # 本文档
├── LICENSE                             # CreativeML OpenRAIL M 许可证
├── Stable_Diffusion_v1_Model_Card.md   # 模型卡片（训练数据、偏见、限制等）
├── setup.py                            # Python 包安装配置
├── environment.yaml                    # Conda 环境配置（本项目使用 pip 替代）
├── main.py                             # 训练/微调入口脚本
├── notebook_helpers.py                 # Notebook 辅助函数
│
├── scripts/                            # 推理脚本
│   ├── txt2img.py                      #   任务一：文字 → 图像
│   ├── img2img.py                      #   任务二：图像 → 图像
│   └── inpaint.py                      #   任务四：图像修复
│
├── ldm/                                # 模型核心代码
│   ├── models/
│   │   ├── diffusion/
│   │   │   ├── ddpm.py                 #   LatentDiffusion 主类（训练+推理逻辑）
│   │   │   ├── ddim.py                 #   DDIM 采样器
│   │   │   ├── plms.py                 #   PLMS 采样器
│   │   │   └── dpm_solver/             #   DPM-Solver 采样器
│   │   └── autoencoder.py              #   VAE / VQ-VAE 编码器解码器
│   └── modules/
│       ├── attention.py                #   CrossAttention / SpatialTransformer
│       ├── diffusionmodules/
│       │   └── openaimodel.py          #   UNetModel（860M 参数去噪网络）
│       └── encoders/
���           └── modules.py              #   FrozenCLIPEmbedder（CLIP 文本编码器封装）
│
├── configs/
│   └── stable-diffusion/
│       └── v1-inference.yaml           # SD v1 推理配置
│
├── models/ldm/stable-diffusion-v1/     # 权重存放目录
│   └── model.ckpt                      #   SD v1-4 权重（需下载，约 4GB）
│
├── data/
│   └── inpainting_examples/            # 图像修复示例数据（8 对图片 + mask）
│
└── assets/                             # README 中引用的示例图片
    ├── modelfigure.png                 #   模型架构图
    ├── v1-variants-scores.jpg          #   各版本评估对比
    ├── stable-samples/txt2img/         #   txt2img 官方示例
    ├── stable-samples/img2img/         #   img2img 官方示例（含草图输入）
    ├── inpainting.png                  #   inpainting 效果示例
    ├── fire.png, a-painting-of-a-fire.png, ...  # 同主题不同风格对比
    └── rick.jpeg                       #   NSFW 替换占位图
```

---

## 六、致谢

- 扩散模型代码基于 [OpenAI 的 ADM 代码库](https://github.com/openai/guided-diffusion) 和 [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Transformer 编码器实现来自 [x-transformers](https://github.com/lucidrains/x-transformers)（[lucidrains](https://github.com/lucidrains)）

## 引用

```bibtex
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
