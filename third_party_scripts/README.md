# 📦 第三方脚本与模型 (third_party_scripts)

<p align="center">
  <strong>audio_filter_pipeline / noise_filter_pipeline 的第三方模型、VAD、分类器资产说明</strong>
</p>

> 本仓库只包含 Audio Filter 与 Noise Filter 两条 pipeline，本文只列出它们实际消费的资产。
> 完整下载入口、目标路径和版本锚点见
> [`docs/guides/reference/Third_Party_Assets.md`](../docs/guides/reference/Third_Party_Assets.md)。

---

## 📑 目录

- [目录概览](#third-party-overview)
- [Audio Filter 音频质量模型](#audio-quality-models)
  - [DNSMOS](#dnsmos)
  - [SCOREQ](#scoreq)
  - [NISQA](#nisqa)
  - [UTMOS](#utmos)
  - [URGENT-PK](#urgent-pk)
- [VAD 模型](#vad-models)
  - [TENVAD](#tenvad)
  - [SileroVAD](#silerovad)
  - [FireRedVAD](#fireredvad)
  - [NeMo VAD ONNX](#nemo-vad-onnx)
- [Audio Filter 声纹运行时](#speaker-runtime)
- [Noise Filter 分类与增强](#noise-filter-models)
  - [Qwen2.5-Omni](#qwen25-omni)
  - [icefall AudioTag](#icefall-audiotag)
  - [Apollo](#apollo)
- [依赖安装汇总](#dependency-installation)
- [项目来源与致谢](#project-sources)

---

<a id="third-party-overview"></a>
## 📁 目录概览

| 目录 | 使用方 | 定位 |
|------|--------|------|
| **DNSMOS/** | Audio Filter Step3 | 默认 MOS，含 P.835 / P808 ONNX |
| **SCOREQ/** | Audio Filter Step3 | 默认 MOS，需单独下载 ONNX |
| **NISQA/** | Audio Filter Step3 | 可选 MOS，代码随仓库，权重需下载 |
| **UTMOS/** | Audio Filter Step3 | 可选 MOS，代码随仓库，checkpoint 需下载 |
| **URGENT-PK/** | Audio Filter Step3 | 可选成对质量排序，需上游 clone + checkpoint |
| **SileroVAD/** | Noise Filter 可选 VAD | `silero_vad.jit` 已随仓库提供 |
| **TENVAD/** | Audio Filter Step1 | 参考 ONNX；运行时使用 `ten_vad` pip 包 |
| **FireRedVAD/** | Noise Filter VAD | 推荐 VAD，权重缺失时可自动下载 |
| **NeMo-VAD-onnx/** | Noise Filter 可选 VAD | 需显式提供可信 ONNX / SHA-256 |
| **wespeaker/** | Audio Filter Step5/6 | 声纹运行时与用户侧 enrollment 资产 |
| **Qwen2.5-Omni-7B/** | Noise Filter Stage C1 | 模型目录约 15–20 GB |
| **icefall_audiotaq/** | Noise Filter Stage C2 | AudioTag 代码 + 标签表随仓库，权重需下载 |
| **icefall/** | Noise Filter Stage C2 | AudioTag 依赖的 upstream 代码，脚本可 clone |
| **Apollo/** | Noise Filter Stage D1 | daemon 脚本/配置随仓库，权重与 `apollo_repo` 需下载 |

---

<a id="audio-quality-models"></a>
## 🎧 Audio Filter 音频质量模型

### DNSMOS

**用途**: Audio Filter Step3 的默认 MOS 之一。P.835 主模型读取 `DNSMOS/DNSMOS/sig_bak_ovr.onnx`，P808 辅助分读取 `DNSMOS/DNSMOS/model_v8.onnx`。

**依赖**:

```bash
pip install "onnxruntime>=1.15" numpy soxr librosa pandas soundfile requests tqdm
```

**资产**: `third_party_scripts/DNSMOS/` 已随仓库提供。模型来源见 [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)。

---

### SCOREQ

**用途**: Audio Filter Step3 的默认 MOS 之一，支持 natural/synthetic 与 nr/ref 组合。

**依赖**:

```bash
pip install "onnxruntime>=1.15" torch torchaudio
```

**资产**: natural NR 默认加载 `third_party_scripts/SCOREQ/adapt_nr_telephone.onnx`。该 ONNX 体积约 360 MB，未随仓库提交；模型发布页为 [Zenodo record 15739280](https://zenodo.org/records/15739280)。

---

### NISQA

**用途**: Audio Filter Step3 的可选 MOS。默认 `SCOREQ + DNSMOS` route 不选择它。

**依赖**:

```bash
pip install torch numpy pandas tqdm pyyaml librosa scipy matplotlib
```

**源码与权重**: 推理代码 `third_party_scripts/NISQA/nisqa/` 已随仓库提供（MIT）；权重 `third_party_scripts/NISQA/weights/nisqa.tar` 需单独下载，权重许可证为 CC BY-NC-SA，许可证文件已随仓库提供。

---

### UTMOS

**用途**: Audio Filter Step3 的可选 MOS。默认 route 不选择它。

**依赖**:

```bash
pip install torch torchaudio
```

**源码与权重**: 推理代码 `third_party_scripts/UTMOS/SpeechMOS/speechmos/` 已随仓库提供（MIT）；权重 `third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt` 需单独下载。上游来源见 [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)。

---

### URGENT-PK

**用途**: Audio Filter Step3 的可选成对感知质量排序模型。默认 route 不选择它。

**资产边界**: 启用时必须显式配置 `urgent_pk_checkpoint`、`urgent_pk_source_dir`、`urgent_pk_encoder` 和 `urgent_pk_backbone`；当前配置使用 `third_party_scripts/URGENT-PK/exp/mel_resnet34_urgent24.ckpt`、`third_party_scripts/URGENT-PK`、`mel`、`resnet34`。运行时还会校验 checkpoint digest 与 source commit，不能只提供目录后依赖隐式猜测。

**详细资料**: [urgent-challenge/URGENT-PK](https://github.com/urgent-challenge/URGENT-PK)。

---

<a id="vad-models"></a>
## 🎤 VAD 模型

### TENVAD

**用途**: Audio Filter Step1 的 native CPU VAD。运行时 `from ten_vad import TenVad`，模型由 pip 包自带。

**安装**:

```bash
pip install git+https://github.com/TEN-framework/ten-vad.git
```

**参考资产**: `third_party_scripts/TENVAD/ten-vad.onnx` 已随仓库提供，仅作模型参考，不被 `ten_vad` runtime 直接读取。

---

### SileroVAD

**用途**: Noise Filter 的可选 VAD backend。Audio Filter 主链已固定为 TEN VAD，不以 Silero 为主 VAD。

**依赖**:

```bash
pip install torch torchaudio
```

**资产**: `third_party_scripts/SileroVAD/silero_vad.jit` 已随仓库提供；缺失时 Noise Filter 会尝试 Torch Hub 缓存或下载。

---

### FireRedVAD

**用途**: Noise Filter 推荐的 VAD backend。

**安装与资产**:

```bash
pip install "fireredvad[gpu]"   # 或 fireredvad
```

权重优先从 `third_party_scripts/FireRedVAD/VAD/` 读取；缺失时会自动下载到 `~/.cache/use_train/FireRedVAD`。上游：[FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)。

---

### NeMo VAD ONNX

**用途**: Noise Filter 的可选 NeMo MarbleNet ONNX VAD，不依赖 `nemo_toolkit` 执行推理。

**依赖**:

```bash
pip install torch torchaudio "onnxruntime>=1.15"
```

**资产**: 将 `vad_marblenet.onnx` 或 `vad.onnx` 放入 `third_party_scripts/NeMo-VAD-onnx/`，或通过配置显式传入 `nemo_onnx_path`。远端下载必须提供可信 SHA-256，入口见 `tools/noise_filter_pipeline/scripts/download_nemo_vad_onnx.py`。

---

<a id="speaker-runtime"></a>
## 🗣️ Audio Filter 声纹运行时

Audio Filter Step5/6 使用 WeSpeaker 计算 embedding，并与 enrollment bundle / exclusion manifest 比较。运行时需要安装 WeSpeaker 并准备用户侧资产：

```bash
pip install wespeaker
```

`config/spk_filter.yaml` 中的 `/REPLACE_ME/...`、全零 digest、enrollment bundle 与 exclusion manifest 都是 fail-closed 占位，必须替换为真实资产后才能运行 Step5/6。WeSpeaker 来源：[wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)。

---

<a id="noise-filter-models"></a>
## 🏷️ Noise Filter 分类与增强

### Qwen2.5-Omni

**用途**: Noise Filter Stage C1 的可懂度与人声/噪声判定。

**依赖与使用**: Qwen2.5 utilities 与模型依赖必须安装在实际运行 `tools/noise_filter_pipeline/main.py` 的同一 Python 环境中。模型目录：

```text
third_party_scripts/Qwen2.5-Omni-7B/
```

下载：

```bash
python tools/noise_filter_pipeline/scripts/download_hive_models.py --qwen
```

### icefall AudioTag

**用途**: Noise Filter Stage C2 的 AudioSet 标签分类。

**资产**:

- 代码与标签表：`third_party_scripts/icefall_audiotaq/` 已随仓库提供（含 `data/class_labels_indices.csv`）。
- 权重：`third_party_scripts/icefall_audiotaq/exp/pretrained.pt` 需下载。
- 模型代码：`third_party_scripts/icefall/` 由 `download_hive_models.py` 自动 clone。

**模型来源**: [marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12](https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12/tree/main)。

### Apollo

**用途**: Noise Filter 可选 Stage D1 的语音超分/修复。

**资产**:

- `third_party_scripts/Apollo/inference_apollo.py`、`config_apollo_uni.yaml` 已随仓库提供。
- `third_party_scripts/Apollo/apollo_model_uni.ckpt` 需下载，约 130 MB。
- `third_party_scripts/apollo_repo/` 需 clone，提供 `look2hear`：

```bash
git clone https://github.com/JusperLee/Apollo.git third_party_scripts/apollo_repo
# 或设置 APOLLO_REPO 指向已有 Apollo 仓库根目录
```

---

<a id="dependency-installation"></a>
## 📦 依赖安装汇总

```bash
# Audio Filter
pip install -r tools/audio_filter_pipeline/requirements.txt
pip install git+https://github.com/TEN-framework/ten-vad.git

# Noise Filter
pip install -r tools/noise_filter_pipeline/requirements.txt
pip install "fireredvad[gpu]"   # 可选 GPU VAD

# Noise Filter Qwen 运行时（安装在主环境）
pip install "qwen-omni-utils[decord]" -U accelerate
```

模型源码、权重、Qwen/AudioTag/Apollo runtime 仍需按上文单独准备。

---

<a id="project-sources"></a>
## 项目来源与致谢

| 项目 | 来源 |
|---|---|
| **DNSMOS** | [GitHub](https://github.com/microsoft/DNS-Challenge) |
| **NISQA** | [GitHub](https://github.com/gabrielmittag/NISQA) |
| **UTMOS** | [GitHub](https://github.com/sarulab-speech/UTMOS22) |
| **SCOREQ** | [GitHub](https://github.com/tjohnson1118/SCOREQ) |
| **URGENT-PK** | [GitHub](https://github.com/urgent-challenge/URGENT-PK) |
| **SileroVAD** | [GitHub](https://github.com/snakers4/silero-vad) |
| **TENVAD** | [GitHub](https://github.com/TEN-framework/ten-vad) |
| **FireRedVAD** | [GitHub](https://github.com/FireRedTeam/FireRedVAD) |
| **NeMo VAD** | [NVIDIA NeMo](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) |
| **Qwen2.5-Omni** | [GitHub](https://github.com/QwenLM/Qwen2.5-Omni) |
| **icefall AudioTag** | [GitHub](https://github.com/k2-fsa/icefall) |
| **Apollo** | [GitHub](https://github.com/JusperLee/Apollo) |
| **WeSpeaker** | [GitHub](https://github.com/wenet-e2e/wespeaker) |

使用相关组件时，请遵循对应项目的 License 与引用要求。
