# speech-data-filter-pipeline

从 `use_train` 抽出的语音筛选与噪声筛选 pipeline。

## 目录

| 路径 | 用途 |
| --- | --- |
| `tools/audio_filter_pipeline/` | 语音筛选：VAD、切分、MOS 打分、声纹筛选与导出（v3 runner） |
| `tools/noise_filter_pipeline/` | 噪声筛选：ingestion、切分、分类与增强 |
| `tools/shared/audio_io.py` | 两条 pipeline 共用的音频 I/O helper |
| `core/` | 最小共享实现：安全 checkpoint 加载、事务文件写入、noise admission policy |
| `third_party_scripts/` | 第三方 VAD / MOS / 分类 / 增强代码与小体积模型资产 |

每条 pipeline 保留自己的文档和依赖：

- `tools/audio_filter_pipeline/README.md`
- `tools/noise_filter_pipeline/README.md`
- `tools/audio_filter_pipeline/requirements.txt`
- `tools/noise_filter_pipeline/requirements.txt`

## 环境

建议 Python 3.10+、PyTorch 2.x；部分音频后端还需要 `ffmpeg`，噪声 pipeline 的重采样可选 `sox`。

```bash
pip install -r tools/audio_filter_pipeline/requirements.txt
pip install -r tools/noise_filter_pipeline/requirements.txt
```

Audio Filter Step1 的 native VAD 还需要：

```bash
pip install git+https://github.com/TEN-framework/ten-vad.git
```

Audio Filter Step5/6 的声纹筛选还需要安装 WeSpeaker，并使用真实的 enrollment bundle / exclusion manifest；`spk_filter.yaml` 随附的是 fail-closed 占位模板。

## 第三方资产（完整运行必需）

完整运行不只依赖仓库内代码，还需要 `third_party_scripts/` 下的第三方代码和模型。完整清单、官方下载入口、目标路径和版本锚点见：

- [`third_party_scripts/README.md`](third_party_scripts/README.md)
- [`docs/guides/reference/Third_Party_Assets.md`](docs/guides/reference/Third_Party_Assets.md)

### 已随仓库提供

| 路径 | 用途 |
| --- | --- |
| `third_party_scripts/DNSMOS/` | DNSMOS 代码与 P.835 / P808 ONNX，Audio Filter 默认 MOS 之一 |
| `third_party_scripts/SileroVAD/silero_vad.jit` | Silero VAD 模型 |
| `third_party_scripts/TENVAD/ten-vad.onnx` | TEN VAD 参考模型；运行时使用 `ten_vad` pip 包 |
| `third_party_scripts/UTMOS/SpeechMOS/` | UTMOS 推理代码（不含 checkpoint） |
| `third_party_scripts/NISQA/` | NISQA 推理代码（MIT，不含权重） |
| `third_party_scripts/Apollo/inference_apollo.py`、`config_apollo_uni.yaml` | Noise Filter 可选 Stage D1 Apollo daemon 代码与配置（不含权重） |
| `third_party_scripts/icefall_audiotaq/` | AudioTag daemon/result 代码与 `class_labels_indices.csv`（不含 `exp/pretrained.pt`） |

### 需单独下载

以下资产因体积或独立许可证未提交，下载命令见上述两个文档：

| 路径 | 用途 | 备注 |
| --- | --- | --- |
| `third_party_scripts/SCOREQ/adapt_nr_telephone.onnx` | Audio Filter 默认 MOS 之一 | 约 360 MB |
| `third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt` | UTMOS 权重 | 约 390 MB |
| `third_party_scripts/NISQA/weights/nisqa.tar` | NISQA 权重 | 约 3 MB，CC BY-NC-SA |
| `third_party_scripts/Apollo/apollo_model_uni.ckpt` | Apollo 超分权重 | 约 130 MB |
| `third_party_scripts/apollo_repo/` | Apollo 上游 `look2hear` 代码 | `git clone https://github.com/JusperLee/Apollo.git`，或设置 `APOLLO_REPO` |
| `third_party_scripts/icefall_audiotaq/exp/pretrained.pt` | AudioTag 权重 | 约 240 MB |
| `third_party_scripts/icefall/` | AudioTag 模型代码 | `download_hive_models.py` 自动 clone |
| `third_party_scripts/Qwen2.5-Omni-7B/` | Noise Filter Stage C1 | 约 15–20 GB |
| `third_party_scripts/URGENT-PK/` | Audio Filter 可选 MOS | 需要上游 git 与 checkpoint |
| `third_party_scripts/wespeaker/{en,chs}/model.onnx` | 可选声纹 / spksim 模型 | 不属于默认主链 |
| `third_party_scripts/FireRedVAD/` | Noise Filter FireRedVAD 权重 | 首次运行可自动下载 |
| `third_party_scripts/NeMo-VAD-onnx/` | 可选 NeMo ONNX VAD | 需显式提供可信 SHA-256 |

Noise Filter 的 Apollo、AudioTag、icefall 与 Qwen 可一次准备：

```bash
python tools/noise_filter_pipeline/scripts/download_hive_models.py --qwen
```

## 使用

从仓库根目录运行，确保 `core` 可导入：

```bash
PYTHONPATH=tools python -m audio_filter_pipeline.cli --help
PYTHONPATH=tools python -m noise_filter_pipeline.cli --help
```

## 测试

```bash
PYTHONPATH=tools python -m pytest tools/audio_filter_pipeline/tests -q
PYTHONPATH=tools python -m pytest tools/noise_filter_pipeline/tests -q -m "pipeline or unit"
```

噪声 pipeline 建议使用独立 conda 环境；`pytest` 与 `pytest-timeout` 已写入其 `requirements.txt`。

## 未包含

- 训练 checkpoint、生成数据、运行输出和日志。
- 上表中需要单独下载的模型权重与大型第三方代码。
- 数据本体、enrollment bundle、exclusion manifest 等用户侧资产。
