# 第三方资产获取与放置

本文件只列出 `audio_filter_pipeline` 与 `noise_filter_pipeline` 实际消费的第三方源码、模型和权重。路径均相对仓库根目录。

用途、依赖和运行边界的详细说明见 [`third_party_scripts/README.md`](../../../third_party_scripts/README.md)。

## Audio Filter MOS 模型

| 资产 | 本地路径 | 用途 | 官方下载 | 目标文件 |
|---|---|---|---|---|
| DNSMOS | `third_party_scripts/DNSMOS/` | 默认 MOS（P.835 + P808） | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | `DNSMOS/sig_bak_ovr.onnx`、`DNSMOS/model_v8.onnx`（已随仓库提供） |
| NISQA | `third_party_scripts/NISQA/` | 可选 MOS | [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) | `weights/nisqa.tar`（代码随仓库，权重需下载） |
| UTMOS | `third_party_scripts/UTMOS/` | 可选 MOS | [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)、[权重直链](https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt) | `utmos22_strong_step7459_v1.pt`；代码在 `SpeechMOS/speechmos/`（代码随仓库） |
| SCOREQ | `third_party_scripts/SCOREQ/` | 默认 MOS | [Zenodo record 15739280](https://zenodo.org/records/15739280) | `adapt_nr_telephone.onnx`（约 360 MB） |
| URGENT-PK | `third_party_scripts/URGENT-PK/` | 可选成对质量排序 | [urgent-challenge/URGENT-PK](https://github.com/urgent-challenge/URGENT-PK)、[权重直链](https://huggingface.co/Twinkzzzzz/URGENT-PK/resolve/main/mel_resnet34_urgent24.ckpt) | `exp/mel_resnet34_urgent24.ckpt` |

NISQA 权重（只下载权重，不要覆盖已随仓库提供的 `nisqa/` 代码）：

```bash
mkdir -p third_party_scripts/NISQA/weights
wget -O third_party_scripts/NISQA/weights/nisqa.tar \
  https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar
```

UTMOS checkpoint：

```bash
wget -O third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt \
  https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt
```

URGENT-PK：

```bash
git clone https://github.com/urgent-challenge/URGENT-PK.git third_party_scripts/URGENT-PK
git -C third_party_scripts/URGENT-PK checkout 484269f
```

## VAD 与 Noise Filter

| 资产 | 本地路径 | 用途 | 官方下载 | 目标文件 |
|---|---|---|---|---|
| SileroVAD | `third_party_scripts/SileroVAD/` | Noise Filter 可选 VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `silero_vad.jit`（已随仓库提供） |
| TENVAD | `third_party_scripts/TENVAD/` | Audio Filter Step1 参考模型 | [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) | `ten-vad.onnx`（已随仓库提供；运行时用 `ten_vad` pip 包） |
| FireRedVAD | `third_party_scripts/FireRedVAD/` | Noise Filter 推荐 VAD | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) | `VAD/`；缺失时自动下载到 `~/.cache/use_train/FireRedVAD` |
| NeMo VAD ONNX | `third_party_scripts/NeMo-VAD-onnx/` | Noise Filter 可选 VAD | [NVIDIA NeMo](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) | `vad_marblenet.onnx` 或 `vad.onnx` |
| Qwen2.5-Omni-7B | `third_party_scripts/Qwen2.5-Omni-7B/` | Noise Filter Stage C1 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)、[ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B) | 完整模型目录（约 15–20 GB） |
| icefall AudioTag | `third_party_scripts/icefall_audiotaq/` | Noise Filter Stage C2 | [HF 模型](https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12) | `exp/pretrained.pt`；代码与 `data/class_labels_indices.csv` 已随仓库提供 |
| icefall 代码 | `third_party_scripts/icefall/` | AudioTag 模型代码 | [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | 由 `download_hive_models.py` 自动 clone |
| Apollo | `third_party_scripts/Apollo/` | Noise Filter 可选 Stage D1 | [JusperLee/Apollo](https://github.com/JusperLee/Apollo)、[HF 权重](https://huggingface.co/JusperLee/Apollo) | `apollo_model_uni.ckpt`；脚本与配置已随仓库提供 |
| Apollo `look2hear` | `third_party_scripts/apollo_repo/` | Apollo daemon 依赖 | [JusperLee/Apollo](https://github.com/JusperLee/Apollo) | 完整上游仓库 |

下载示例：

```bash
python tools/noise_filter_pipeline/scripts/download_hive_models.py --qwen

git clone https://github.com/JusperLee/Apollo.git third_party_scripts/apollo_repo

# NeMo VAD ONNX 远端下载必须提供可信 SHA-256
export NEMO_VAD_ONNX_URL="https://example.invalid/vad_marblenet.onnx"
export NEMO_VAD_ONNX_SHA256="<64 位十六进制 SHA-256>"
python tools/noise_filter_pipeline/scripts/download_nemo_vad_onnx.py
```

## Audio Filter 声纹运行时

Audio Filter Step5/6 需要 WeSpeaker 运行时与用户侧资产：

```bash
pip install wespeaker
```

`config/spk_filter.yaml` 中的 `model_dir`、enrollment bundle、exclusion manifest 和 digest 都是 `/REPLACE_ME` 占位，必须替换为真实资产。WeSpeaker 来源：[wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)。

## 需要 Git 克隆的上游仓库

| 本地路径 | 上游 | 本机锚定 commit |
|---|---|---|
| `third_party_scripts/URGENT-PK/` | https://github.com/urgent-challenge/URGENT-PK.git | `484269f` |
| `third_party_scripts/icefall/` | https://github.com/k2-fsa/icefall.git | `7a35ca2` |
| `third_party_scripts/apollo_repo/` | https://github.com/JusperLee/Apollo.git | `c46640f` |

```bash
git clone https://github.com/urgent-challenge/URGENT-PK.git third_party_scripts/URGENT-PK
git -C third_party_scripts/URGENT-PK checkout 484269f
```

commit 是本机验证过的版本，不是强制版本；上游更新后如果要复现已验证行为，优先回到该 commit。

## 说明

- DNSMOS、SileroVAD、TENVAD、UTMOS SpeechMOS 代码、NISQA 代码、Apollo 脚本/配置、icefall_audiotaq 代码与标签表已随仓库提供。
- 其余模型/权重因体积或独立许可证未提交，按本文下载。
- 下载脚本必须保持 HTTPS 校验；不要为单个资产引入关闭证书校验的下载逻辑。
