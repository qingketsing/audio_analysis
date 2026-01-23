# Hotel Video Auditory Analytics (IJHM-2025 Paper Reproduction)

本项目旨在复现论文 **"Auditory features in hotel short videos"** (IJHM 2025) 中的方法论与统计模型。

项目提供了一套完整的 Python 自动化工作流，能够批量处理短视频，提取关键的声学特征（音高、音强、语速、音质、情感），由于论文假设声学特征与观众参与度（Engagement）之间存在非线性关系，本项目最后使用二次回归模型（Quadratic Regression）来验证这一"倒U型"假设。

---

## 📂 目录结构

```text
Project/
├── vedios/                 # [输入] 存放原始 .mp4 短视频文件 (Git仅上传文件夹结构)
├── wavs/                   # [中间] 存放转换后的 .wav 音频文件 (由脚本自动生成)
├── features/               # [输出] 存放特征提取结果(.csv) 和 分析图表(.png)
├── wav2vec2_checkpoints/   # [缓存] 自动下载的 HuggingFace 模型权重 (Git已忽略)
│
├── transform.py            # [预处理] 批量将 MP4 视频转换为 WAV 音频
├── feature_pitch.py        # [特征 1] 提取音高 (Pitch: Mean, SD, Min, Max)
├── feature_intensity.py    # [特征 2] 提取音强 (Intensity: Mean, SD)
├── feature_speech_rate.py  # [特征 3] 提取语速 (Speech Rate, Syllable Count)
├── feature_voice_quality.py# [特征 4] 提取音质 (Voice Quality: Jitter, Shimmer, HNR)
├── feature_emotion.py      # [特征 5] 提取情感 (Arousal, Valence) - 基于 AI 模型
│
├── analysis_model.py       # [建模] 数据合并、清洗、二次回归分析与可视化
├── README.md               # 项目说明文档
└── .gitignore              # Git 版本控制忽略规则
```

---

## 🛠️ 环境安装

本项目依赖 Python 音频处理与统计分析库。推荐使用 **Conda** 进行环境管理。

### 1. 创建虚拟环境
```bash
# 创建名为 audio_analysis 的环境，Python 版本 3.9
conda create -n audio_analysis python=3.9 -y

# 激活环境
conda activate audio_analysis
```

### 2. 安装依赖库
混合使用 `conda` 和 `pip` 以获取最佳的兼容性（特别是音频驱动和深度学习框架）。

```bash
# 1. 安装基础科学计算与绘图库
conda install pandas numpy scipy statsmodels matplotlib seaborn -y

# 2. 安装 PyTorch (音频情感分析模型需要)
# 注意：即使没有 GPU，CPU 版本也足够处理音频任务
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. 安装音频处理专用库 (Praat 接口与视频处理)
pip install praat-parselmouth moviepy transformers
```

---

## 🚀 使用指南 (标准工作流)

请按照以下顺序执行脚本，完成从视频到统计结论的全过程。

### 步骤 0: 准备数据
*   将你的 **.mp4** 视频文件复制到项目根目录下的 `vedios/` 文件夹中。

### 步骤 1: 格式转换 (Pre-processing)
将视频中的音频流无损分离并转换为标准 WAV 格式。
```bash
python transform.py
```
> **结果**：`wavs/` 文件夹下将生成对应的 `.wav` 文件。

### 步骤 2: 特征提取 (Feature Extraction)
依次运行以下脚本来构建多维度的声学特征库。所有生成的 CSV 文件会自动保存在 `features/` 目录下。

```bash
# 1. 基础声学特征
python feature_pitch.py          # -> features/features_pitch.csv
python feature_intensity.py      # -> features/features_intensity.csv
python feature_voice_quality.py  # -> features/features_voice_quality.csv

# 2. 高级语言特征 (语速)
python feature_speech_rate.py    # -> features/features_speech_rate.csv

# 3. 情感特征 (AI判断)
# 注意：首次运行会自动下载约 300MB 的 Wav2Vec2 模型，请保持网络连接
python feature_emotion.py        # -> features/features_emotion.csv
```

### 步骤 3: 统计建模与可视化 (Analysis)
最后，运行分析脚本将所有特征表合并，并进行回归建模。

```bash
python analysis_model.py
```

> **输出结果**：
> 1.  控制台打印 **OLS Regression Results** (包含 R-squared, P值, 系数等)。
> 2.  `features/` 目录下生成 **`regression_plots.png`**，可视化展示"倒U型"曲线。

---

## ⚙️ 自定义与进阶修改

### 如何使用真实的业务数据 (Engagement)?
默认情况下，本项目的 `analysis_model.py` 为了演示目的，使用了**合成数据**来模拟互动量。如果你有真实数据，请修改 `analysis_model.py`：

1.  准备一个名为 `engagement.csv` 的文件，需要包含两列：
    *   `filename`: 视频对应的音频文件名 (例如 `video_01.wav`)
    *   `engagement`: 互动数值 (如：点赞数+评论数)
2.  打开 `analysis_model.py`：
    *   在 `load_and_merge_data` 函数中，添加读取 `engagement.csv` 的代码。
    *   找到 `synthesize_missing_data` 函数，**注释掉**生成 `df["engagement"] = ...` 的相关代码行。

### 如何训练新内容?
本项目并非"训练"一个从零开始的模型，而是"提取特征 + 统计回归"。如果你要分析一批新视频：
1.  **清空** `vedios/` 和 `wavs/` 文件夹中的旧文件（保留 `.gitkeep`）。
2.  放入新视频。
3.  **重新运行** 步骤 1 至 步骤 3 的所有脚本。

---

## 🔒 数据隐私与 Git 策略

本项目配置了严格的 `.gitignore` 规则，以防止私有数据意外泄露到公共代码仓库。

*   **允许上传**：所有的 `.py` 代码文件、`README.md`、`.gitignore` 以及文件夹的**目录结构**（通过 `.gitkeep` 实现）。
*   **禁止上传**：
    *   `vedios/` 文件夹内的具体视频文件。
    *   `wavs/` 文件夹内的音频文件。
    *   `features/` 文件夹内的 CSV 数据表和分析图表。
    *   `wav2vec2_checkpoints/` 自动下载的模型权重文件。

别人 Clone 你的项目后，会看到完整的文件夹结构，但文件夹是空的，他们需要填入自己的视频数据才能开始分析。
