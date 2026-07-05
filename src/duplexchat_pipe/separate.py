from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as F_audio
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

REPO_ID = "sarulab-speech/DialogueSidon"
MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 10.0

_cache: dict = {}


def load_separation_models(device: str = "cuda") -> dict:
    """Download and load DialogueSidon models. Cached per device."""
    resolved = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if resolved in _cache:
        return _cache[resolved]

    paths = {f: hf_hub_download(repo_id=REPO_ID, filename=f) for f in MODEL_FILES}

    with open(paths["metadata.json"]) as fp:
        meta = json.load(fp)

    torch_device = torch.device(resolved)

    # Enable TF32 for faster float32 matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ssl_encoder = torch.export.load(paths["ssl_encoder.pt2"]).module().to(torch_device)
    diffusion_head = torch.export.load(paths["diffusion_head.pt2"]).module().to(torch_device)
    vae_decoder = torch.export.load(paths["vae_decoder.pt2"]).module().to(torch_device)

    latent_norm_mean = torch.tensor(
        meta["latent_norm_mean"], dtype=torch.float32, device=torch_device
    ).view(1, 1, -1)
    latent_norm_std = torch.tensor(
        meta["latent_norm_std"], dtype=torch.float32, device=torch_device
    ).view(1, 1, -1)

    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"],
        algorithm_type="dpmsolver++",
        timestep_spacing="linspace",
    )

    models = {
        "ssl_encoder": ssl_encoder,
        "diffusion_head": diffusion_head,
        "vae_decoder": vae_decoder,
        "latent_norm_mean": latent_norm_mean,
        "latent_norm_std": latent_norm_std,
        "latent_norm_initialized": meta["latent_norm_initialized"],
        "scheduler": scheduler,
        "latent_dim": meta["latent_dim"],
        "sample_rate": meta["sample_rate"],
        "device": torch_device,
    }
    _cache[resolved] = models
    return models


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad_batch(
    features: list[torch.Tensor],
    pad_to_multiple_of: int = 2,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = (
            (target_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full(
        (batch_size, target_length, feature_dim), padding_value,
        dtype=torch.float32, device=device,
    )
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask


def _extract_fbank(waveforms: list[torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        # kaldi.fbank requires CPU
        feat = kaldi.fbank(
            wav.cpu().unsqueeze(0),
            sample_frequency=SAMPLE_RATE_IN,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))

    stride = 2
    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}


def _normalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return ((latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]).to(latents.dtype)


def _denormalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return (latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]).to(latents.dtype)


def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / denom) if float(denom) > 1e-8 else 0.0


def _maybe_swap(
    prev_overlap: torch.Tensor, curr_chunk: torch.Tensor, overlap_samples: int
) -> tuple[torch.Tensor, bool]:
    if overlap_samples <= 0 or prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False
    curr_ov = curr_chunk[:, :overlap_samples]
    direct = (
        _channel_similarity(prev_overlap[0], curr_ov[0])
        + _channel_similarity(prev_overlap[1], curr_ov[1])
    )
    swapped = (
        _channel_similarity(prev_overlap[0], curr_ov[1])
        + _channel_similarity(prev_overlap[1], curr_ov[0])
    )
    if swapped > direct:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False


@torch.inference_mode()
def _separate_chunk(wav: torch.Tensor, num_steps: int, models: dict) -> torch.Tensor:
    """Separate a single chunk. Input: (1, T) at 16 kHz. Output: (2, T_out) at model sr."""
    # Re-enable TF32 in case pyannote disabled it
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = models["device"]
    latent_dim = models["latent_dim"]

    noisy_ssl = _extract_fbank([wav.view(-1)], device)

    features, pred0, pred1 = models["ssl_encoder"](
        noisy_ssl["input_features"], noisy_ssl["attention_mask"]
    )

    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)

    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(
        (1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype
    )
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(
            models["diffusion_head"](latents, t_batch, conditioning), t, latents
        ).prev_sample

    latents = _denormalize(latents, models)
    spk0 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(0)
    spk1 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    return torch.cat([spk0, spk1], dim=0)  # (2, T)


# ── public API ────────────────────────────────────────────────────────────────

def run_separation(
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
    models: dict,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Separate a (1, T) mono waveform into two speaker tracks.

    Returns (spk0, spk1, out_sr) where each track is a (1, T) CPU float32 tensor.
    """
    device = models["device"]
    out_sr: int = models["sample_rate"]

    if sample_rate != SAMPLE_RATE_IN:
        wav = F_audio.resample(wav, sample_rate, SAMPLE_RATE_IN)
    wav = wav.to(device)

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE_IN)
    total_samples = wav.shape[-1]

    if total_samples <= chunk_samples:
        max_val = wav.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * wav / max_val, (160, 160))
        separated = _separate_chunk(wav_norm, num_steps, models)
    else:
        overlap_samples_in = int(OVERLAP_SECONDS * SAMPLE_RATE_IN)
        hop_samples = chunk_samples - overlap_samples_in
        starts = list(range(0, total_samples, hop_samples))
        stitched: torch.Tensor | None = None
        prev_end_in = 0

        for start in starts:
            end = min(start + chunk_samples, total_samples)
            chunk = wav[:, start:end]
            max_val = chunk.abs().max().clamp_min(1e-6)
            chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))
            pred = _separate_chunk(chunk_norm, num_steps, models)

            target_out = max(1, round((end - start) * out_sr / SAMPLE_RATE_IN))
            if pred.shape[-1] > target_out:
                pred = pred[:, :target_out]
            elif pred.shape[-1] < target_out:
                pred = torch.cat(
                    [pred, torch.zeros(2, target_out - pred.shape[-1], device=device)], dim=-1
                )

            if stitched is None:
                stitched = pred
                prev_end_in = end
                continue

            overlap_in = max(0, prev_end_in - start)
            overlap_out = max(0, min(
                round(overlap_in * out_sr / SAMPLE_RATE_IN),
                stitched.shape[-1],
                pred.shape[-1],
            ))
            if overlap_out > 0:
                pred, _ = _maybe_swap(stitched[:, -overlap_out:], pred, overlap_out)
                fade = torch.linspace(0.0, 1.0, overlap_out, device=device).unsqueeze(0)
                blended = stitched[:, -overlap_out:] * (1 - fade) + pred[:, :overlap_out] * fade
                stitched = torch.cat(
                    [stitched[:, :-overlap_out], blended, pred[:, overlap_out:]], dim=-1
                )
            else:
                stitched = torch.cat([stitched, pred], dim=-1)
            prev_end_in = end

        separated = stitched  # type: ignore[assignment]

    spk0 = separated[0:1].cpu()
    spk1 = separated[1:2].cpu()
    return spk0, spk1, out_sr
