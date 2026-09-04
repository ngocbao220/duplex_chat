from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import get_token

from duplexchat_pipe.audio import load_wav_tensor
from duplexchat_pipe.model_options import DIARIZATION_MODELS, infer_diarization_backend, resolve_model_alias

if TYPE_CHECKING:
    from pyannote.audio import Pipeline


class FileDiarizationAdapter:
    backend: str

    def diarize_file(self, wav_path: Path) -> list[dict]:
        raise NotImplementedError


class SortformerDiarizationAdapter(FileDiarizationAdapter):
    backend = "sortformer"

    def __init__(self, model: Any):
        self.model = model

    def diarize_file(self, wav_path: Path) -> list[dict]:
        predicted = self.model.diarize(audio=str(wav_path), batch_size=1)
        return _segments_from_sortformer_output(predicted)


class DiariZenDiarizationAdapter(FileDiarizationAdapter):
    backend = "diarizen"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def diarize_file(self, wav_path: Path) -> list[dict]:
        output = self.pipeline(str(wav_path), sess_name=wav_path.stem)
        return _segments_from_annotation(output)


def load_diarization_pipeline(
    model: str,
    device: str = "cuda",
    backend: str = "auto",
) -> "Pipeline | FileDiarizationAdapter":
    model = resolve_model_alias(model, DIARIZATION_MODELS) or model
    backend_norm = backend.strip().lower()
    if backend_norm == "auto":
        backend_norm = infer_diarization_backend(model)
    if backend_norm == "sortformer":
        return _load_sortformer_pipeline(model, device)
    if backend_norm == "diarizen":
        return _load_diarizen_pipeline(model, device)
    if backend_norm != "pyannote":
        raise ValueError("Unsupported diarization backend '%s'. Use auto, pyannote, sortformer, or diarizen." % backend)
    return _load_pyannote_pipeline(model, device)


def _resolve_device(device: str) -> str:
    return device if (device != "cuda" or torch.cuda.is_available()) else "cpu"


def _load_pyannote_pipeline(model: str, device: str = "cuda") -> "Pipeline":
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Hugging Face token not found. Please login via huggingface-cli or set a token."
        )

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError("pyannote.audio is required for diarization") from exc

    try:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, token=token)

    resolved_device = _resolve_device(device)
    pipeline.to(torch.device(resolved_device))
    return pipeline


def _load_sortformer_pipeline(model: str, device: str = "cuda") -> SortformerDiarizationAdapter:
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Sortformer diarization requires NVIDIA NeMo. Install the sortformer "
            "dependency profile before using nvidia/diar_sortformer_4spk-v1."
        ) from exc

    diar_model = SortformerEncLabelModel.from_pretrained(model)
    if hasattr(diar_model, "eval"):
        diar_model.eval()
    resolved_device = _resolve_device(device)
    if hasattr(diar_model, "to"):
        diar_model.to(torch.device(resolved_device))
    return SortformerDiarizationAdapter(diar_model)


def _load_diarizen_pipeline(model: str, device: str = "cuda") -> DiariZenDiarizationAdapter:
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "DiariZen diarization requires the BUTSpeechFIT/DiariZen package and "
            "its pyannote-compatible environment. Install the diarizen profile in "
            "a separate environment before using BUT-FIT/diarizen-wavlm-large-s80-md."
        ) from exc

    pipeline = DiariZenPipeline.from_pretrained(model)
    resolved_device = _resolve_device(device)
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device(resolved_device))
    return DiariZenDiarizationAdapter(pipeline)


def _segments_from_annotation(output: Any) -> list[dict]:
    segments = []
    diarization = output.speaker_diarization if hasattr(output, "speaker_diarization") else output
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
        )
    segments.sort(key=lambda x: x["start"])
    return segments


def _segments_from_sortformer_output(output: Any) -> list[dict]:
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], (list, tuple)):
        output = output[0]
    segments = []
    for item in output:
        parsed = _parse_sortformer_segment(item)
        if parsed is not None:
            segments.append(parsed)
    segments.sort(key=lambda x: x["start"])
    return segments


def _parse_sortformer_segment(item: Any) -> dict | None:
    if isinstance(item, dict):
        start = item.get("start", item.get("begin", item.get("start_time")))
        end = item.get("end", item.get("stop", item.get("end_time")))
        speaker = item.get("speaker", item.get("label", item.get("speaker_id")))
        if start is not None and end is not None and speaker is not None:
            return {"speaker": str(speaker), "start": float(start), "end": float(end)}
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {"speaker": str(item[2]), "start": float(item[0]), "end": float(item[1])}
    if isinstance(item, str):
        parts = re.split(r"[\s,]+", item.strip())
        numbers = []
        speaker = None
        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                if part:
                    speaker = part
        if len(numbers) >= 2:
            if speaker is None and len(numbers) >= 3:
                speaker = str(int(numbers[2]))
            return {
                "speaker": str(speaker or "SPEAKER_00"),
                "start": numbers[0],
                "end": numbers[1],
            }
    return None


import numpy as np
from collections.abc import Callable

def run_diarization(
    pipeline: "Pipeline | FileDiarizationAdapter",
    wav_path: Path,
    max_chunk_dur: float = 60.0,
    progress_callback: Callable[[str, int], None] | None = None,
) -> list[dict]:
    """
    Chạy diarization bằng cách dùng VAD để cắt audio thành các chunk <= max_chunk_dur,
    sau đó so sánh embedding để gán nhãn speaker globally (giúp tránh OOM).
    """
    if isinstance(pipeline, FileDiarizationAdapter):
        if progress_callback is not None:
            progress_callback("start", 1)
        segments = pipeline.diarize_file(wav_path)
        if progress_callback is not None:
            progress_callback("advance", 1)
            progress_callback("close", 0)
        return segments

    waveform, sample_rate = load_wav_tensor(wav_path)
    dur_sec = waveform.shape[1] / sample_rate
    
    # 1. Dùng Silero VAD để lấy các phân đoạn có giọng nói
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        get_speech_timestamps = utils[0]
        # Silero VAD yêu cầu tensor 1D và sample_rate=16000
        speech_ts = get_speech_timestamps(waveform[0], vad_model, sampling_rate=sample_rate)
        vad_segments = [(ts["start"]/sample_rate, ts["end"]/sample_rate) for ts in speech_ts]
    except Exception as e:
        print(f"Warning: Silero VAD failed ({e}), falling back to full audio.")
        vad_segments = [(0.0, dur_sec)]
        
    if not vad_segments:
        return []

    # 2. Gom nhóm các segment VAD thành các chunk sao cho (end_N - start_1) <= max_chunk_dur
    chunks = []
    curr_start = vad_segments[0][0]
    curr_end = vad_segments[0][1]
    
    for st, en in vad_segments[1:]:
        if en - curr_start <= max_chunk_dur:
            curr_end = en
        else:
            chunks.append((curr_start, curr_end))
            curr_start = st
            curr_end = en
    chunks.append((curr_start, curr_end))
    if progress_callback is not None:
        progress_callback("start", len(chunks))
    
    # 3. Chạy diarization trên từng chunk & trích xuất embedding
    all_segments = []
    global_speaker_embs = {}  # {global_spk_id: list_of_embeddings}
    global_spk_idx = 0
    
    # Lấy model embedding từ pipeline (nếu có)
    embedding_model = getattr(pipeline, "_embedding", None)
    
    for c_start, c_end in chunks:
        # Mở rộng nhẹ chunk để không cắt gắt
        pad = 0.5
        s_pad = max(0.0, c_start - pad)
        e_pad = min(dur_sec, c_end + pad)
        
        s_idx = int(s_pad * sample_rate)
        e_idx = int(e_pad * sample_rate)
        chunk_wav = waveform[:, s_idx:e_idx]
        
        try:
            output = pipeline({"waveform": chunk_wav, "sample_rate": sample_rate})
        except Exception as e:
            print(f"Warning: Diarization failed on chunk {s_pad}-{e_pad}: {e}")
            if progress_callback is not None:
                progress_callback("advance", 1)
            continue
            
        local_segs = []
        iterator = (
            output.speaker_diarization.itertracks(yield_label=True)
            if hasattr(output, "speaker_diarization")
            else output.itertracks(yield_label=True)
        )

        for turn, _, speaker in iterator:
            local_segs.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })
            
        if not local_segs:
            if progress_callback is not None:
                progress_callback("advance", 1)
            continue
            
        # 4. Gắn speaker globally bằng cách so sánh Cosine Similarity của embedding
        local_to_global = {}
        for spk in set(s["speaker"] for s in local_segs):
            spk_segs = [s for s in local_segs if s["speaker"] == spk]
            spk_embs = []
            
            if embedding_model is not None:
                for s in spk_segs:
                    seg_s_idx = int(s["start"] * sample_rate)
                    seg_e_idx = int(s["end"] * sample_rate)
                    seg_wav = chunk_wav[:, seg_s_idx:seg_e_idx]
                    
                    if seg_wav.shape[1] > 160: # Tránh segment quá ngắn
                        with torch.no_grad():
                            try:
                                emb = embedding_model(seg_wav.unsqueeze(0))
                                spk_embs.append(emb.squeeze(0).cpu().numpy())
                            except:
                                pass
                                
            if spk_embs:
                local_emb = np.mean(spk_embs, axis=0)
                local_emb = local_emb.flatten()
                
                # Tìm global speaker phù hợp nhất
                best_sim = -1.0
                best_g_spk = None
                for g_spk, g_embs in global_speaker_embs.items():
                    g_emb = np.mean(g_embs, axis=0).flatten()
                    sim = np.dot(local_emb, g_emb) / (np.linalg.norm(local_emb) * np.linalg.norm(g_emb) + 1e-8)
                    if sim > best_sim:
                        best_sim = sim
                        best_g_spk = g_spk
                        
                # Ngưỡng cosine similarity
                if best_sim > 0.7 and best_g_spk is not None:
                    local_to_global[spk] = best_g_spk
                    global_speaker_embs[best_g_spk].append(local_emb)
                else:
                    new_g_spk = f"SPEAKER_{global_spk_idx:02d}"
                    global_spk_idx += 1
                    local_to_global[spk] = new_g_spk
                    global_speaker_embs[new_g_spk] = [local_emb]
            else:
                # Fallback nếu không tính được embedding
                new_g_spk = f"SPEAKER_UNK_{global_spk_idx:02d}"
                global_spk_idx += 1
                local_to_global[spk] = new_g_spk
                
        # Cập nhật thời gian thực tế và append
        for s in local_segs:
            all_segments.append({
                "speaker": local_to_global[s["speaker"]],
                "start": s["start"] + s_pad,
                "end": s["end"] + s_pad,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback("advance", 1)

    all_segments.sort(key=lambda x: x["start"])
    if progress_callback is not None:
        progress_callback("close", 0)
    return all_segments
