# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse
import concurrent.futures
import glob
import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def _cuda_providers(mem_limit_mb: int):
    """CUDA ExecutionProvider 的 provider options + SessionOptions 双保险。

    - ``arena_extend_strategy=kSameAsRequested``：按请求大小扩展 arena，
      避免默认 kNextPowerOfTwo 按 2 的幂激进预留（预留不释放）；
    - ``gpu_mem_limit``（provider）与 ``gpu.mem_limit``（session config
      entry）同时设置：不同 onnxruntime 版本对两者的支持不同（1.17+ 倾向
      session config，部分版本仍认 provider options），双保险保证任一
      生效即可限制单 session 显存（含 arena 预留），防止长时间运行中
      不同 shape 的中间 buffer 在 arena 累积预留把卡占满——实测单进程
      曾膨胀到 81GB（DNSMOS 21 段推理只需 ~7GB），之后连几十 MB 的小
      buffer 都分配失败（BFCArena OOM 事故）。
    """
    return [
        ("CUDAExecutionProvider", {
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": mem_limit_mb * 1024 * 1024,
        }),
        "CPUExecutionProvider",
    ]


def _make_onnx_session(model_path: str, mem_limit_mb: int) -> "ort.InferenceSession":
    """创建 onnxruntime session：provider options + SessionOptions 双保险限制显存。

    1.23 实测 provider options 的 gpu_mem_limit 未生效（加载后仍预分配
    总显存 ~30%），故同时写 session config entry（gpu.mem_limit /
    gpu.arena_extend_strategy），任一被当前版本识别即可生效。
    """
    opts = ort.SessionOptions()
    try:
        opts.add_session_config_entry("gpu.mem_limit", str(mem_limit_mb * 1024 * 1024))
        opts.add_session_config_entry("gpu.arena_extend_strategy", "kSameAsRequested")
    except Exception:
        pass
    return ort.InferenceSession(
        model_path, opts, providers=_cuda_providers(mem_limit_mb)
    )


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, use_gpu: bool = False) -> None:
        if use_gpu:
            # 主模型 4 段（max_seg_per_run）峰值 ~2.8GB，arena 限 16GB（shape 固定后不累积，arena 大则 0.7GB 中间 buffer 请求更易满足）；
            # p808 为轻量模型限 4GB。总 onnx 预留 ≤ 18GB，80GB 卡剩余充足
            self.onnx_sess = _make_onnx_session(primary_model_path, 16 * 1024)
            self.p808_onnx_sess = _make_onnx_session(p808_model_path, 4 * 1024)
        else:
            self.onnx_sess = ort.InferenceSession(primary_model_path)
            self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

    def run_batch_audio(self, list_of_audio_sr, sampling_rate=16000, is_personalized_MOS=False):
        """True ONNX batch: one sess.run with shape (N, T).  *list_of_audio_sr*: list of (audio_1d, sr)."""
        fs = sampling_rate
        len_samples = int(INPUT_LENGTH * fs)
        segments_primary = []
        segments_p808 = []
        clip_segment_end = []

        for audio, sr in list_of_audio_sr:
            audio = np.asarray(audio, dtype=np.float64)
            if sr != fs:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=fs)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            actual_len = len(audio)
            while len(audio) < len_samples:
                audio = np.append(audio, audio)
            num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
            hop_len_samples = fs
            start = len(segments_primary)
            for idx in range(num_hops):
                st = int(idx * hop_len_samples)
                en = int((idx + INPUT_LENGTH) * hop_len_samples)
                audio_seg = audio[st:en]
                if len(audio_seg) < len_samples:
                    continue
                audio_seg = np.array(audio_seg, dtype=np.float32)
                segments_primary.append(audio_seg)
                mel = self.audio_melspec(audio=audio_seg[:-160], sr=fs)
                segments_p808.append(np.array(mel, dtype=np.float32))
            clip_segment_end.append(len(segments_primary))

        if not segments_primary:
            return [{'OVRL': 3.0, 'SIG': 3.0, 'BAK': 3.0, 'P808_MOS': 3.0, 'num_hops': 0} for _ in list_of_audio_sr]

        # 单次 run 的最大 segment 数：4。shape=(4,144160) 时 wav2vec2 特征提取
        # Conv 中间 buffer 峰值 ~0.7GB，低于实测 arena 可用连续空间下限（~0.9GB），
        # 不再出现 onnxruntime 软 OOM；8 段/run 的 1.4GB buffer 在 arena 碎片化时
        # 请求失败（BFCArena 只打 E 日志不抛异常，任务“看起来完成但全程 OOM”）。
        # 不足 4 段时补足到固定 4 段 shape：arena 只缓存单一 batch shape，
        # 不同 shape 的中间 buffer 不再累积。
        max_seg_per_run = 4
        out_primary_list = []
        out_p808_list = []
        for i in range(0, len(segments_primary), max_seg_per_run):
            sub = segments_primary[i : i + max_seg_per_run]
            sub_p = segments_p808[i : i + max_seg_per_run]
            n_real = len(sub)
            if n_real < max_seg_per_run:
                # 固定 shape：不足时重复最后一段补足（多算的段在下方截断丢弃）
                sub = sub + [sub[-1]] * (max_seg_per_run - n_real)
                sub_p = sub_p + [sub_p[-1]] * (max_seg_per_run - n_real)
            sub_primary = np.stack(sub, axis=0)
            sub_p808 = np.stack(sub_p, axis=0)
            out_primary_list.append(self.onnx_sess.run(None, {'input_1': sub_primary})[0][:n_real])
            out_p808_list.append(self.p808_onnx_sess.run(None, {'input_1': sub_p808})[0][:n_real])
        out_primary = np.concatenate(out_primary_list, axis=0)
        out_p808 = np.concatenate(out_p808_list, axis=0)

        seg_start = 0
        results = []
        for clip_idx, seg_end in enumerate(clip_segment_end):
            n_h = seg_end - seg_start
            if n_h == 0:
                results.append({'OVRL': 3.0, 'SIG': 3.0, 'BAK': 3.0, 'P808_MOS': 3.0, 'num_hops': 0})
                continue
            sig_raw = out_primary[seg_start:seg_end, 0]
            bak_raw = out_primary[seg_start:seg_end, 1]
            ovr_raw = out_primary[seg_start:seg_end, 2]
            sig_p, bak_p, ovr_p = self.get_polyfit_val(
                np.mean(sig_raw), np.mean(bak_raw), np.mean(ovr_raw), is_personalized_MOS
            )
            p808_vals = out_p808[seg_start:seg_end]
            if p808_vals.ndim >= 3:
                p808_mos = float(np.mean(p808_vals[:, 0, 0]))
            else:
                p808_mos = float(np.mean(p808_vals))
            results.append({
                'OVRL_raw': float(np.mean(ovr_raw)),
                'SIG_raw': float(np.mean(sig_raw)),
                'BAK_raw': float(np.mean(bak_raw)),
                'OVRL': float(ovr_p),
                'SIG': float(sig_p),
                'BAK': float(bak_p),
                'P808_MOS': float(p808_mos),
                'num_hops': n_h,
            })
            seg_start = seg_end

        return results

def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join('DNSMOS', 'model_v8.onnx')

    if args.personalized_MOS:
        primary_model_path = os.path.join('pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
    else:
        print(df.describe())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    args = parser.parse_args()

    main(args)
